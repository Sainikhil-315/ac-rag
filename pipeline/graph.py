"""
pipeline/graph.py
Assembles the full AC-RAG LangGraph StateGraph.

Graph topology:

  START ──► entry_router ──► (unknown) ──► direct_responder ──► END
                    │
                    └──► (rag) ──► query_analyzer ──► retrieval_planner ──► retriever
                                        ▲                                      │
                                        │             (0 docs retry)           │
                                        │                  ▼                   │
                                        │             validator                │
                                        │           pass │  │ fail             │
                                        │               ▼  └──► retriever (retry)
                                        │       context_refiner                │
                                        │               │                      │
                                        │           generator ◄────────────────┘
                                        │               │         (format retry)
                                        │            critic
                                        │         pass │  │ fail
                                        │            END  └──► (retry)
                                        └──────────────────────────────────────┘

Entry Router:
  Similarity ≥ threshold  → "rag"     (full RAG pipeline)
  Similarity < threshold  → "unknown" (return "I don't know")

Terminal nodes:
  end_success      : answer accepted, pipeline done
  end_error        : unrecoverable error
  end_max_retries  : retry budget exhausted

Usage:
    from pipeline.graph import build_pipeline
    pipeline = build_pipeline()
    result = pipeline.invoke({"query": "What is...", ...})
"""

import logging

from langgraph.graph import StateGraph, END

from pipeline.state import ACRagState, initial_state
from pipeline.nodes.query_analyzer import query_analyzer_node
from pipeline.nodes.retrieval_planner import retrieval_planner_node
from pipeline.nodes.validator import validator_node
from pipeline.nodes.targeted_retrieval import make_targeted_retrieval_node
from pipeline.nodes.context_refiner import context_refiner_node
from pipeline.nodes.generator import generator_node
from pipeline.nodes.claim_extractor import claim_extractor_node
from pipeline.nodes.claim_verifier import claim_verifier_node
from pipeline.nodes.contradiction_detector import contradiction_detector_node
from pipeline.nodes.critic import critic_node
from pipeline.router import (
    route_after_entry_router,
    route_after_query_analyzer,
    route_after_retrieval_planner,
    route_after_retriever,
    route_after_validator,
    route_after_targeted_retrieval,
    route_after_context_refiner,
    route_after_generator,
    route_after_claim_extractor,
    route_after_claim_verifier,
    route_after_contradiction_detector,
    route_after_critic,
    route_after_max_retries,
)
from vectorstore.store import VectorStoreManager
from pipeline.nodes.retriever import make_retriever_node
from pipeline.nodes.entry_router import make_entry_router_node
from pipeline.nodes.direct_responder import direct_responder_node

logger = logging.getLogger(__name__)


# ── Retry counter wrapper ──────────────────────────────────────────────────────

def _increment_retry(state: ACRagState) -> ACRagState:
    """
    Thin pass-through node placed between the critic/validator and any retry target.
    """
    new_count = state.get("retry_count", 0) + 1
    new_control_steps = state.get("total_control_steps", 0) + 1
    logger.info("[Pipeline] Retry #%d (total_control_steps=%d)", new_count, new_control_steps)
    return {
        **state,
        "retry_count": new_count,
        "total_control_steps": new_control_steps,
    }


# ── Terminal node stubs ────────────────────────────────────────────────────────

def _end_success_node(state: ACRagState) -> ACRagState:
    logger.info("[Pipeline] ✓ Pipeline completed successfully (retries=%d)", state.get("retry_count", 0))
    return state


def _end_error_node(state: ACRagState) -> ACRagState:
    logger.error("[Pipeline] ✗ Pipeline terminated with error: %s", state.get("error"))
    return state


def _end_max_retries_node(state: ACRagState) -> ACRagState:
    logger.warning("[Pipeline] ✗ Pipeline terminated: max retries (%d) exhausted", state.get("retry_count", 0))

    if not state.get("answer"):
        from pipeline.nodes.direct_responder import _UNKNOWN_RESPONSE
        return {
            **state,
            "answer": _UNKNOWN_RESPONSE,
            "error": state.get("error") or "Max retries exhausted without a passing answer.",
        }

    return {
        **state,
        "error": state.get("error") or "Max retries exhausted without a passing answer.",
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_pipeline(vsm: VectorStoreManager = None) -> StateGraph:
    """
    Build and compile the full Evidence-Driven Adaptive AC-RAG LangGraph pipeline.
    """
    if vsm is None:
        logger.info("[Pipeline] Loading vector store...")
        vsm = VectorStoreManager()
        vsm.load()

    retriever_node = make_retriever_node(vsm)
    entry_router_node = make_entry_router_node(vsm)
    targeted_retrieval_node = make_targeted_retrieval_node(vsm)

    # ── Define graph ──────────────────────────────────────────────────────────
    graph = StateGraph(ACRagState)

    # Entry routing nodes
    graph.add_node("entry_router",     entry_router_node)
    graph.add_node("direct_responder", direct_responder_node)

    # RAG pipeline nodes
    graph.add_node("query_analyzer",         query_analyzer_node)
    graph.add_node("retrieval_planner",      retrieval_planner_node)
    graph.add_node("retriever",              retriever_node)
    graph.add_node("validator",              validator_node)
    graph.add_node("targeted_retrieval",    targeted_retrieval_node)
    graph.add_node("context_refiner",        context_refiner_node)
    graph.add_node("generator",              generator_node)
    graph.add_node("claim_extractor",        claim_extractor_node)
    graph.add_node("claim_verifier",         claim_verifier_node)
    graph.add_node("contradiction_detector", contradiction_detector_node)
    graph.add_node("critic",                 critic_node)
    graph.add_node("increment_retry",        _increment_retry)

    # Terminal nodes
    graph.add_node("end_success",            _end_success_node)
    graph.add_node("end_error",              _end_error_node)
    graph.add_node("end_max_retries",        _end_max_retries_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("entry_router")

    # ── Edges ─────────────────────────────────────────────────────────────────

    graph.add_conditional_edges(
        "entry_router",
        route_after_entry_router,
        {
            "rag":     "query_analyzer",
            "unknown": "direct_responder",
        },
    )

    graph.add_edge("direct_responder", "end_success")

    graph.add_conditional_edges(
        "query_analyzer",
        route_after_query_analyzer,
        {
            "retrieval_planner": "retrieval_planner",
            "end_error": "end_error",
        },
    )

    graph.add_conditional_edges(
        "retrieval_planner",
        route_after_retrieval_planner,
        {"retriever": "retriever"},
    )

    graph.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "validator":        "validator",
            "query_analyzer":   "increment_retry",
            "end_error":        "end_error",
            "end_max_retries":  "end_max_retries",
        },
    )

    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "context_refiner":     "context_refiner",
            "targeted_retrieval":  "targeted_retrieval",
            "retriever":           "increment_retry",
            "end_max_retries":     "end_max_retries",
            "end_error":           "end_error",
        },
    )

    graph.add_conditional_edges(
        "targeted_retrieval",
        route_after_targeted_retrieval,
        {"validator": "validator"},
    )

    graph.add_conditional_edges(
        "context_refiner",
        route_after_context_refiner,
        {"generator": "generator"},
    )

    graph.add_conditional_edges(
        "generator",
        route_after_generator,
        {
            "claim_extractor": "claim_extractor",
            "end_error":       "end_error",
        },
    )

    graph.add_conditional_edges(
        "claim_extractor",
        route_after_claim_extractor,
        {
            "claim_verifier": "claim_verifier",
            "end_error":      "end_error",
        },
    )

    graph.add_conditional_edges(
        "claim_verifier",
        route_after_claim_verifier,
        {
            "contradiction_detector": "contradiction_detector",
            "end_error":              "end_error",
        },
    )

    graph.add_conditional_edges(
        "contradiction_detector",
        route_after_contradiction_detector,
        {
            "critic":      "critic",
            "end_success": "end_success",
            "end_error":   "end_error",
        },
    )

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end_success":        "end_success",
            "targeted_retrieval": "targeted_retrieval",
            "query_analyzer":     "increment_retry",
            "generator":          "increment_retry",
            "end_max_retries":    "end_max_retries",
        },
    )

    graph.add_conditional_edges(
        "increment_retry",
        _route_after_retry_increment,
        {
            "query_analyzer":     "query_analyzer",
            "retriever":          "retriever",
            "targeted_retrieval": "targeted_retrieval",
            "generator":          "generator",
            "end_max_retries":    "end_max_retries",
        },
    )

    graph.add_edge("end_success", END)
    graph.add_edge("end_error",   END)

    graph.add_conditional_edges(
        "end_max_retries",
        route_after_max_retries,
        {
            "end": END,
        },
    )

    return graph.compile()


def _route_after_retry_increment(state: ACRagState) -> str:
    from config.settings import (
        MAX_RETRIES,
        MAX_RETRIEVAL_ROUNDS,
        MAX_GENERATION_REPAIRS,
        MAX_TOTAL_CONTROL_STEPS,
    )

    retry = state.get("retry_count", 0)
    retrieval_attempts = state.get("retrieval_attempts", 0)
    repair_attempts = state.get("repair_attempts", 0)
    total_control_steps = state.get("total_control_steps", 0)

    if retry > MAX_RETRIES or total_control_steps >= MAX_TOTAL_CONTROL_STEPS:
        return "end_max_retries"

    reason = state.get("retry_reason") or "content"

    if reason == "format":
        if repair_attempts >= MAX_GENERATION_REPAIRS:
            return "end_max_retries"
        return "generator"

    if reason in ("unsupported_claim", "coverage"):
        if retrieval_attempts >= MAX_RETRIEVAL_ROUNDS:
            return "end_max_retries"
        return "targeted_retrieval"

    if state.get("validation_passed") is False:
        if retrieval_attempts >= MAX_RETRIEVAL_ROUNDS:
            return "end_max_retries"
        return "retriever"

    return "query_analyzer"



# ── Convenience run function ───────────────────────────────────────────────────

def run_pipeline(
    query: str,
    vsm: VectorStoreManager = None,
    conversation_history=None,
) -> ACRagState:
    """
    High-level entry point: build pipeline, run query, return final state.

    Args:
        query : the user's question
        vsm   : optional pre-loaded VectorStoreManager (avoids reloading index)
        conversation_history : optional prior turns [{"role", "content"}, ...] for
            follow-up/pronoun resolution in query_analyzer (see GAPS_AND_IMPROVEMENTS.md
            item C.15). None for single-shot queries — CLI and eval runs never pass this.

    Returns:
        Final ACRagState after pipeline completes. Includes token_usage (cost/latency
        tracking, see GAPS_AND_IMPROVEMENTS.md item C.18) when LLM_PROVIDER="openai" —
        a query typically makes 4-7 LLM calls across nodes (analyzer, planner refinement,
        generator, critic, refiner compression, plus retries), and this had no
        instrumentation at all before.
    """
    pipeline = build_pipeline(vsm)
    state = initial_state(query, conversation_history=conversation_history)
    logger.info("[Pipeline] Running query: '%s'", query)
    # recursion_limit = (MAX_RETRIES+1) nodes per retry cycle * number of nodes + buffer

    from config.settings import LLM_PROVIDER
    if LLM_PROVIDER == "openai":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from langchain_community.callbacks import get_openai_callback
            with get_openai_callback() as cb:
                result = pipeline.invoke(state, config={"recursion_limit": 100})
                result = {
                    **result,
                    "token_usage": {
                        "total_tokens": cb.total_tokens,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_cost_usd": round(cb.total_cost, 6),
                        "llm_calls": cb.successful_requests,
                    },
                }
        logger.info(
            "[Pipeline] Token usage: %d total (%d prompt + %d completion) | "
            "$%.6f | %d LLM calls",
            cb.total_tokens, cb.prompt_tokens, cb.completion_tokens,
            cb.total_cost, cb.successful_requests,
        )
    else:
        # Cost tracking currently only implemented for OpenAI — langchain_community's
        # callback pricing table doesn't cover Google/Anthropic/Groq. token_usage stays
        # None for those providers rather than silently reporting zero/wrong numbers.
        result = pipeline.invoke(state, config={"recursion_limit": 50})

    return result
