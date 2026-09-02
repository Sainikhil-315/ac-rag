"""
pipeline/nodes/retrieval_planner.py
Retrieval Planner Node — the "agent" that decides HOW to retrieve.

Responsibilities:
  - Read complexity_score and intent from state
  - Decide: k, fetch_k, lambda_mult, modality_filter, use_multi_query, retrieval_depth
  - Write a RetrievalPlan to state

Design:
  Two-tier approach:
    1. Rule-based defaults from complexity score (fast, deterministic, ablation-safe)
    2. LLM refinement pass that can override defaults for edge cases
       (e.g. a simple query about a table needs modality_filter="table")

  This keeps the planner both fast for simple queries and smart for complex ones.
  Ablation: set USE_RETRIEVAL_PLANNER=False in settings to skip and use defaults.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import (
    RETRIEVAL_K_MIN,
    RETRIEVAL_K_MAX,
    RETRIEVAL_K_DEFAULT,
    MMR_FETCH_K_MULTIPLIER,
    MMR_LAMBDA_MULT,
    USE_RETRIEVAL_PLANNER,
)
from pipeline.state import ACRagState, RetrievalPlan
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


# ── Rule-based planner (always runs) ─────────────────────────────────────────

def _rule_based_plan(complexity: float, intent: str) -> RetrievalPlan:
    """
    Deterministic plan based on complexity score thresholds.
    Used as the default and as the fallback if LLM planner fails.
    """
    if complexity < 0.4:
        k = RETRIEVAL_K_MIN          # 4 — simple query, few passages needed
        depth = "shallow"
        multi_query = False
        lambda_mult = 0.7            # lean relevance for simple factual queries

    elif complexity < 0.7:
        k = RETRIEVAL_K_DEFAULT      # 6 — moderate
        depth = "standard"
        multi_query = False
        lambda_mult = MMR_LAMBDA_MULT  # balanced

    else:
        k = min(10, RETRIEVAL_K_MAX)  # 10 — complex, need broad coverage
        depth = "deep"
        multi_query = True            # use sub-queries for multi-hop
        lambda_mult = 0.3             # lean diversity — cover different aspects

    # Analytical and comparative queries benefit from diversity
    if intent in ("analytical", "comparative"):
        lambda_mult = max(0.3, lambda_mult - 0.1)
        k = min(k + 2, RETRIEVAL_K_MAX)

    return RetrievalPlan(
        k=k,
        fetch_k=k * MMR_FETCH_K_MULTIPLIER,
        lambda_mult=lambda_mult,
        modality_filter="all",          # LLM refiner may override this
        use_multi_query=multi_query,
        retrieval_depth=depth,
    )


# ── Modality guard ────────────────────────────────────────────────────────────

# Only accept a specific modality when the query explicitly mentions it.
# Without this guard, the LLM planner tends to over-specify "table" or "figure"
# for general queries — which limits retrieval to a narrow slice of chunks and
# causes important text passages to be missed.
_TABLE_KEYWORDS  = frozenset(["table", "chart", "statistics", "statistic", "percentage"])
_FIGURE_KEYWORDS = frozenset(["figure", "image", "diagram", "plot", "graph", "visual", "picture"])

def _sanitize_modality(query: str, modality: str) -> str:
    """
    Return 'all' unless the query explicitly mentions a modality keyword.
    Prevents the LLM planner from choosing 'figure'/'table' for generic questions.
    """
    if modality in ("all", "text"):
        return modality
    q_lower = query.lower()
    if modality == "table" and not any(kw in q_lower for kw in _TABLE_KEYWORDS):
        logger.info("[RetrievalPlanner] Modality 'table' overridden → 'all' (no table keyword in query)")
        return "all"
    if modality == "figure" and not any(kw in q_lower for kw in _FIGURE_KEYWORDS):
        logger.info("[RetrievalPlanner] Modality 'figure' overridden → 'all' (no figure keyword in query)")
        return "all"
    return modality


# ── LLM-based refinement (runs when USE_RETRIEVAL_PLANNER=True) ───────────────

class PlanRefinement(BaseModel):
    modality_filter: str = Field(
        description="Which modality to retrieve: 'text', 'table', 'figure', or 'all'"
    )
    k_override: int = Field(
        ge=4, le=12,
        description="Override k value if the default is clearly wrong for this query"
    )
    override_reason: str = Field(
        description="Why you changed from the rule-based defaults (or 'no change')"
    )


_PLANNER_SYSTEM = """You are a retrieval planning agent for a document QA system.
Given a query, its intent, and a rule-based retrieval plan, refine the plan.

Your job is limited to:
1. Setting modality_filter: if the query asks about a table/chart/figure explicitly,
   set modality to "table" or "figure". Otherwise keep "all".
2. Optionally adjusting k if the rule-based value is clearly wrong.

Do NOT change fetch_k, lambda_mult, or use_multi_query — those are set by rules."""

_PLANNER_HUMAN = """Query: {query}
Intent: {intent}
Complexity: {complexity}
Rule-based k: {k}
Rule-based depth: {depth}

Refine the plan."""

_planner_prompt = ChatPromptTemplate.from_messages([
    ("system", _PLANNER_SYSTEM),
    ("human", _PLANNER_HUMAN),
])


def _llm_refine_plan(plan: RetrievalPlan, state: ACRagState) -> RetrievalPlan:
    """Apply LLM refinement on top of the rule-based plan."""
    try:
        chain = _planner_prompt | get_llm().with_structured_output(PlanRefinement)

        refinement: PlanRefinement = chain.invoke({
            "query": state.get("rewritten_query") or state["query"],
            "intent": state.get("intent", "unknown"),
            "complexity": state.get("complexity_score", 0.5),
            "k": plan["k"],
            "depth": plan["retrieval_depth"],
        })

        logger.info(
            "[RetrievalPlanner] LLM refinement → modality=%s, k=%d (%s)",
            refinement.modality_filter, refinement.k_override, refinement.override_reason
        )

        raw_modality = _sanitize_modality(
            state.get("rewritten_query") or state["query"],
            refinement.modality_filter,
        )
        plan["modality_filter"] = raw_modality
        plan["k"] = refinement.k_override
        plan["fetch_k"] = refinement.k_override * MMR_FETCH_K_MULTIPLIER

    except Exception as e:
        logger.warning("[RetrievalPlanner] LLM refinement failed, using rule-based plan: %s", e)

    return plan


# ── Node function ─────────────────────────────────────────────────────────────

def retrieval_planner_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Retrieval Planner.
    Reads:  state["complexity_score"], state["intent"], state["query"]
    Writes: state["retrieval_plan"]
    """
    complexity = state.get("complexity_score") or 0.5
    intent = state.get("intent") or "factual"

    log_entry = {
        "stage": "retrieval_planner",
        "status": "started",
        "details": {"complexity": complexity, "intent": intent},
    }
    logger.info("[RetrievalPlanner] Planning retrieval (complexity=%.2f, intent=%s)", complexity, intent)

    # Step 1: Rule-based plan (always)
    plan = _rule_based_plan(complexity, intent)

    # Step 2: LLM refinement (if enabled)
    if USE_RETRIEVAL_PLANNER:
        plan = _llm_refine_plan(plan, state)
    else:
        logger.info("[RetrievalPlanner] Ablation: LLM planner disabled, using rule-based defaults")

    logger.info(
        "[RetrievalPlanner] Final plan → k=%d, modality=%s, multi_query=%s, depth=%s",
        plan["k"], plan["modality_filter"], plan["use_multi_query"], plan["retrieval_depth"]
    )

    log_entry["status"] = "completed"
    log_entry["details"].update(dict(plan))

    return {
        **state,
        "retrieval_plan": plan,
        "stage_logs": state["stage_logs"] + [log_entry],
    }
