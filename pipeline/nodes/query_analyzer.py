"""
pipeline/nodes/query_analyzer.py
Query Analyzer Node — first stage of the AC-RAG pipeline.

Responsibilities:
  1. Intent detection    : classify query as factual / analytical / comparative / summarization
  2. Complexity scoring  : 0.0 (simple lookup) → 1.0 (multi-hop reasoning)
  3. Query decomposition : if complexity ≥ 0.6, split into focused sub-queries

Design:
  Uses structured LLM output (Pydantic model) to guarantee parseable JSON.
  Pydantic validation acts as a contract — downstream nodes can trust field types.

  Complexity thresholds:
    0.0 – 0.39 : simple   → k=4,  no decomposition
    0.4 – 0.69 : moderate → k=6,  optional decomposition
    0.7 – 1.0  : complex  → k=10–12, decomposition enabled
"""

import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pipeline.state import ACRagState
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

# ── Pydantic output schema ─────────────────────────────────────────────────────

class QueryAnalysis(BaseModel):
    rewritten_query: str = Field(
        description=(
            "A clarified, retrieval-optimised version of the original query. "
            "Fix grammar, resolve vague pronouns, expand abbreviations, and make the "
            "question self-contained and specific. If the original is already clear, "
            "return it unchanged."
        )
    )
    intent: str = Field(
        description="Query intent: 'factual', 'analytical', 'comparative', or 'summarization'"
    )
    complexity_score: float = Field(
        ge=0.0, le=1.0,
        description="Query complexity from 0.0 (simple) to 1.0 (complex multi-hop)"
    )
    complexity_reason: str = Field(
        description="One-sentence explanation of the complexity score"
    )
    decomposed_queries: List[str] = Field(
        default_factory=list,
        description="Sub-queries if complex (complexity >= 0.6), else empty list"
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a query analysis expert for a document QA system.
Analyse the user's query and return a structured JSON response.

Query rewriting rules (rewritten_query field):
  - Fix spelling mistakes, grammar errors, and incomplete sentences
  - Resolve vague pronouns ("it", "that", "they") using context clues in the query,
    OR using the conversation history if provided below — e.g. if the previous turn
    discussed "the validator" and the current query asks "what's its threshold?",
    rewrite to "what is the validator's threshold?"
  - Expand abbreviations if their meaning is inferable
  - Make the question self-contained and specific enough for vector search
  - Do NOT add information that isn't implied by the original query or prior turns
  - If the original query is already clear and specific, return it unchanged
  - If conversation history is provided but the current query is already
    self-contained (doesn't reference prior turns), ignore the history

Intent categories:
  - factual       : asking for a specific fact, number, date, or definition
  - analytical    : asking to explain, analyse, or reason about something
  - comparative   : asking to compare two or more things
  - summarization : asking for a summary or overview

Complexity scoring guide:
  0.0–0.3 : Single fact lookup, one document needed
  0.4–0.6 : Requires synthesising a few passages, some reasoning
  0.7–0.9 : Multi-hop reasoning, cross-document synthesis
  1.0     : Extremely complex, requires full document understanding

Query decomposition rules:
  - Only decompose if complexity_score >= 0.6
  - Base sub-queries on the rewritten_query, not the original
  - Each sub-query must be self-contained and answerable independently
  - Maximum 4 sub-queries
  - If simple, return an empty list for decomposed_queries"""

_HUMAN_PROMPT = "{history_block}Query: {query}"

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_PROMPT),
])

_MAX_HISTORY_TURNS = 3  # most recent N turns only — older context rarely matters and costs tokens


def _format_history(history) -> str:
    """
    Render the last few conversation turns as a text block for the prompt.
    Returns "" if there's no history — keeps the prompt identical to the
    pre-multi-turn behavior for single-shot queries (CLI, eval runs).
    """
    if not history:
        return ""
    recent = history[-_MAX_HISTORY_TURNS:]
    lines = ["Conversation history (most recent last):"]
    for turn in recent:
        role = turn.get("role", "user")
        content = (turn.get("content") or "")[:500]  # cap length per turn
        lines.append(f"  {role}: {content}")
    lines.append("")  # blank line separating history from the current query
    return "\n".join(lines) + "\n"


# ── LLM with structured output ────────────────────────────────────────────────

def _get_analyzer_chain():
    return _prompt | get_llm().with_structured_output(QueryAnalysis)


# ── Node function ─────────────────────────────────────────────────────────────

def query_analyzer_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Query Analyzer.
    Reads:  state["query"]
    Writes: state["rewritten_query"], state["intent"], state["complexity_score"], state["decomposed_queries"]
    """
    query = state["query"]
    history_block = _format_history(state.get("conversation_history"))
    log_entry = {"stage": "query_analyzer", "status": "started", "details": {"query": query}}
    logger.info("[QueryAnalyzer] Analysing query: '%s'", query)

    try:
        chain = _get_analyzer_chain()
        result: QueryAnalysis = chain.invoke({"query": query, "history_block": history_block})

        # If not complex enough, clear any decomposed queries
        if result.complexity_score < 0.6:
            result.decomposed_queries = []

        # Log if the query was actually changed
        if result.rewritten_query != query:
            logger.info("[QueryAnalyzer] Query rewritten: '%s' → '%s'", query, result.rewritten_query)
        else:
            logger.info("[QueryAnalyzer] Query unchanged after rewrite")

        logger.info(
            "[QueryAnalyzer] intent=%s | complexity=%.2f | sub-queries=%d",
            result.intent, result.complexity_score, len(result.decomposed_queries)
        )

        log_entry["status"] = "completed"
        log_entry["details"].update({
            "rewritten_query": result.rewritten_query,
            "intent": result.intent,
            "complexity_score": result.complexity_score,
            "complexity_reason": result.complexity_reason,
            "num_sub_queries": len(result.decomposed_queries),
        })

        return {
            **state,
            "rewritten_query": result.rewritten_query,
            "intent": result.intent,
            "complexity_score": result.complexity_score,
            "decomposed_queries": result.decomposed_queries,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[QueryAnalyzer] Failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)
        return {
            **state,
            "error": f"QueryAnalyzer failed: {e}",
            "stage_logs": state["stage_logs"] + [log_entry],
        }
