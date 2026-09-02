"""
pipeline/nodes/generator.py
Generator Node — Answer Generation (full implementation).

Formal objective: a = argmax P(a | q, R(q, D))
Constraint:       Faith(a, R(q, D)) ≥ τ  (τ = FAITHFULNESS_THRESHOLD)

Algorithm:
  1. Build a generation prompt with the structured context and query
  2. Instruct the LLM to:
     - Answer strictly from the provided context
     - Cite sources inline as [N] for each claim
     - Never add information not present in the context
  3. Parse the structured response (answer text + cited sources)
  4. Build evidence attribution map via utils/attribution.py
  5. Return answer, clean answer (no citations), and attribution list

Design:
  Uses Pydantic structured output so the answer and citation list are
  always parseable — no fragile regex on free-form LLM text.

  The "no hallucination" constraint is enforced at TWO levels:
    1. System prompt: explicit instruction with examples
    2. Critic node: faithfulness dimension scores hallucination on 1–5 scale
"""

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import LLM_MAX_TOKENS
from pipeline.state import ACRagState
from utils.attribution import build_attribution, clean_citations_from_answer
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


# ── Pydantic output schema ─────────────────────────────────────────────────────

class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description=(
            "The answer to the question, strictly grounded in the provided context. "
            "Cite sources inline using [N] notation, e.g. 'The model achieved 94% accuracy [1][3].' "
            "If the context does not contain enough information, say: "
            "'The provided documents do not contain sufficient information to answer this question.'"
        )
    )
    is_answerable: bool = Field(
        description="True if the context contains enough information to answer the question."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that the answer is complete and faithful to the context (0–1)."
    )
    key_sources_used: List[int] = Field(
        default_factory=list,
        description="List of source numbers [N] that were primarily used to construct the answer."
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise document question-answering assistant.

STRICT RULES — you MUST follow all of these:
1. Answer ONLY using information explicitly present in the provided context.
2. Do NOT add any knowledge from outside the context, even if you are certain it is correct.
3. Cite EVERY claim with its source number in brackets, e.g. [1] or [2][3].
4. If the context contains tables or figures (marked with modality=table/figure), read them carefully.
5. If the answer spans multiple sources, cite all of them for the relevant claim.
6. If the context does not contain enough information, state this clearly — do NOT guess.
7. Be concise but complete. Do not pad the answer with unnecessary text.

Context format:
  [SOURCE N | file=... | page=... | section=... | chunk=... | score=... | modality=...]
  <passage text>

Each SOURCE tag identifies a passage. Use [N] inline to cite it."""

_HUMAN_PROMPT = """Context:
{context}

---

Question: {query}

Answer the question using ONLY the context above. Cite sources inline with [N]."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_PROMPT),
])


# ── Node function ─────────────────────────────────────────────────────────────

def generator_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Generator (full implementation).
    Reads:  state["query"], state["refined_context"]
    Writes: state["answer"], state["answer_with_attribution"]
    """
    query = state["query"]
    # NOTE: use `or ""`, not a .get() default — initial_state() sets refined_context=None
    # explicitly, so the key always exists and .get(key, "") would return None anyway.
    context = state.get("refined_context") or ""

    log_entry: Dict[str, Any] = {
        "stage": "generator",
        "status": "started",
        "details": {"query": query, "context_chars": len(context)},
    }

    if not context:
        logger.warning("[Generator] Empty context — cannot generate grounded answer.")
        log_entry["status"] = "completed"
        log_entry["details"]["is_answerable"] = False
        return {
            **state,
            "answer": "The provided documents do not contain sufficient information to answer this question.",
            "answer_with_attribution": [],
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    try:
        chain = _prompt | get_llm(max_tokens=LLM_MAX_TOKENS).with_structured_output(GeneratedAnswer)

        result: GeneratedAnswer = chain.invoke({
            "query": query,
            "context": context,
        })

        logger.info(
            "[Generator] answerable=%s | confidence=%.2f | sources_used=%s",
            result.is_answerable,
            result.confidence,
            result.key_sources_used,
        )

        # Build per-sentence attribution
        attribution = build_attribution(result.answer, context)

        # Clean version for display (no inline [N] markers)
        clean_answer = clean_citations_from_answer(result.answer)

        log_entry["status"] = "completed"
        log_entry["details"].update({
            "is_answerable": result.is_answerable,
            "confidence": result.confidence,
            "key_sources": result.key_sources_used,
            "answer_chars": len(result.answer),
            "attributed_sentences": len(attribution),
        })

        return {
            **state,
            "answer": clean_answer,
            "answer_with_attribution": attribution,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[Generator] Failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)
        return {
            **state,
            "answer": None,
            "answer_with_attribution": [],
            "error": f"Generator failed: {e}",
            "stage_logs": state["stage_logs"] + [log_entry],
        }
