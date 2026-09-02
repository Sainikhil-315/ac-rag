"""
pipeline/nodes/critic.py
Critic Node — Self-Reflection (full implementation).

Evaluates the generated answer on 5 dimensions (1–5 scale):
  1. faithfulness     : Is every claim in the answer supported by the context?
  2. completeness     : Does the answer address all aspects of the question?
  3. table_accuracy   : Are table values/statistics cited correctly? (5 if no tables)
  4. figure_accuracy  : Are figure descriptions accurate? (5 if no figures)
  5. conciseness      : Is the answer free of padding and repetition?

Accept condition: ALL dimensions ≥ CRITIC_MIN_SCORE (4)

Retry classification:
  "content" → faithfulness/completeness/table/figure failed
               → graph routes back to query_analyzer (full restart)
  "format"  → only conciseness failed
               → graph routes back to generator (regenerate, same context)

Ablation:
  USE_CRITIC=False in settings → graph router bypasses this node entirely.
"""

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.settings import CRITIC_MIN_SCORE, USE_CRITIC
from pipeline.state import ACRagState, CriticScores
from utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


# ── Pydantic output schema ─────────────────────────────────────────────────────

class CriticEvaluation(BaseModel):
    faithfulness: int = Field(
        ge=1, le=5,
        description=(
            "Score 1–5: Is every claim in the answer explicitly supported by the context? "
            "5=fully grounded, 1=contains hallucinated or unsupported claims."
        )
    )
    completeness: int = Field(
        ge=1, le=5,
        description=(
            "Score 1–5: Does the answer address all key aspects of the question? "
            "5=comprehensive, 1=major gaps."
        )
    )
    table_accuracy: int = Field(
        ge=1, le=5,
        description=(
            "Score 1–5: Are table values, statistics, and structured data cited correctly? "
            "Score 5 if the question/context involves no tables."
        )
    )
    figure_accuracy: int = Field(
        ge=1, le=5,
        description=(
            "Score 1–5: Are figure descriptions and visual data interpreted accurately? "
            "Score 5 if the question/context involves no figures."
        )
    )
    conciseness: int = Field(
        ge=1, le=5,
        description=(
            "Score 1–5: Is the answer concise and free of unnecessary padding or repetition? "
            "5=perfectly concise, 1=severely padded or repetitive."
        )
    )
    feedback: str = Field(
        description=(
            "Concise explanation of the lowest-scoring dimensions. "
            "Be specific about what is wrong so the generator can fix it on retry."
        )
    )
    retry_reason: str = Field(
        description=(
            "If any score < 4, classify the primary failure as: "
            "'content' (faithfulness/completeness/table/figure issue) or "
            "'format' (only conciseness issue). "
            "Use 'none' if all scores >= 4."
        )
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a rigorous quality critic for a document question-answering system.
Your role is to evaluate the quality of a generated answer against the source context.

Scoring scale:
  5 = Excellent — fully meets the criterion
  4 = Good — minor issues but acceptable
  3 = Fair — noticeable problems
  2 = Poor — significant problems
  1 = Unacceptable — criterion completely fails

Be strict. The system is research-grade and incorrect or unfaithful answers are unacceptable.
If even one factual claim in the answer cannot be traced to the context, faithfulness must be ≤ 3."""

_HUMAN_PROMPT = """Question:
{query}

Source Context:
{context}

Generated Answer:
{answer}

Evaluate the answer on all five dimensions. Be specific in your feedback."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_PROMPT),
])


# ── retry_reason sanitiser ────────────────────────────────────────────────────

_VALID_RETRY_REASONS = {"content", "format", "none"}

def _sanitize_retry_reason(raw: str) -> str:
    """
    Normalize the LLM-produced retry_reason to one of: 'content', 'format', 'none'.

    Some models (Groq/Llama, older GPT) occasionally leak partial JSON tokens or
    free-text into structured output fields.  This guard prevents those garbage values
    from confusing the router and causing infinite retry loops.

    Decision logic:
      - exact match to the three valid values → return as-is
      - contains "format" but not "content" → "format"
      - contains any content/quality keyword  → "content"
      - contains "none" or looks empty/junk   → "none"
      - anything else                         → "content" (safest default: retry)
    """
    if not raw:
        return "content"
    clean = raw.strip().lower()
    if clean in _VALID_RETRY_REASONS:
        return clean
    if "format" in clean and "content" not in clean:
        return "format"
    if any(w in clean for w in ("content", "faith", "complete", "table", "figure", "accuracy")):
        return "content"
    if "none" in clean:
        return "none"
    # junk like "}", "><|python_tag|>", ", and " — treat as content issue
    return "content"


# ── Scoring helper ────────────────────────────────────────────────────────────

def _all_pass(eval_result: CriticEvaluation) -> bool:
    dims = [
        eval_result.faithfulness,
        eval_result.completeness,
        eval_result.table_accuracy,
        eval_result.figure_accuracy,
        eval_result.conciseness,
    ]
    return all(d >= CRITIC_MIN_SCORE for d in dims)


def _compute_overall(eval_result: CriticEvaluation) -> float:
    dims = [
        eval_result.faithfulness,
        eval_result.completeness,
        eval_result.table_accuracy,
        eval_result.figure_accuracy,
        eval_result.conciseness,
    ]
    return round(sum(dims) / len(dims), 2)


# ── Node function ─────────────────────────────────────────────────────────────

def critic_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Critic (full implementation).
    Reads:  state["query"], state["refined_context"], state["answer"]
    Writes: state["critic_scores"], state["critic_passed"], state["critic_feedback"],
            state["retry_reason"]
    """
    query = state["query"]
    context = state.get("refined_context", "")
    answer = state.get("answer", "")

    log_entry: Dict[str, Any] = {
        "stage": "critic",
        "status": "started",
        "details": {"query": query},
    }

    # ── Ablation bypass ───────────────────────────────────────────────────────
    if not USE_CRITIC:
        logger.info("[Critic] Ablation: critic disabled — auto-passing answer.")
        scores = CriticScores(
            faithfulness=5, completeness=5, table_accuracy=5,
            figure_accuracy=5, conciseness=5, overall=5.0,
            feedback="Critic disabled (ablation mode).",
        )
        log_entry["status"] = "completed"
        log_entry["details"].update({"ablation_skip": True, "passed": True})
        return {
            **state,
            "critic_scores": scores,
            "critic_passed": True,
            "critic_feedback": scores["feedback"],
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    if not answer:
        logger.warning("[Critic] No answer to evaluate.")
        log_entry["status"] = "completed"
        log_entry["details"]["passed"] = False
        scores = CriticScores(
            faithfulness=1, completeness=1, table_accuracy=5,
            figure_accuracy=5, conciseness=5, overall=2.4,
            feedback="No answer was generated.",
        )
        return {
            **state,
            "critic_scores": scores,
            "critic_passed": False,
            "critic_feedback": "No answer was generated.",
            "retry_reason": "content",
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    try:
        chain = _prompt | get_llm().with_structured_output(CriticEvaluation)

        # Truncate context to avoid exceeding context window in critic call
        context_preview = context[:8000] if len(context) > 8000 else context

        eval_result: CriticEvaluation = chain.invoke({
            "query": query,
            "context": context_preview,
            "answer": answer,
        })

        passed = _all_pass(eval_result)
        overall = _compute_overall(eval_result)

        logger.info(
            "[Critic] faith=%d complete=%d table=%d figure=%d concise=%d | overall=%.1f | passed=%s",
            eval_result.faithfulness,
            eval_result.completeness,
            eval_result.table_accuracy,
            eval_result.figure_accuracy,
            eval_result.conciseness,
            overall,
            passed,
        )

        if not passed:
            logger.info(
                "[Critic] Answer rejected. retry_reason=%s | feedback: %s",
                eval_result.retry_reason,
                eval_result.feedback[:200],
            )

        scores = CriticScores(
            faithfulness=eval_result.faithfulness,
            completeness=eval_result.completeness,
            table_accuracy=eval_result.table_accuracy,
            figure_accuracy=eval_result.figure_accuracy,
            conciseness=eval_result.conciseness,
            overall=overall,
            feedback=eval_result.feedback,
        )

        # Sanitise retry_reason — guards against garbage LLM output like "}", "><|python_tag|>"
        clean_reason = _sanitize_retry_reason(eval_result.retry_reason) if not passed else "none"

        log_entry["status"] = "completed"
        log_entry["details"].update({
            "scores": {k: v for k, v in scores.items() if k != "feedback"},
            "passed": passed,
            "retry_reason": clean_reason,
        })

        return {
            **state,
            "critic_scores": scores,
            "critic_passed": passed,
            "critic_feedback": eval_result.feedback,
            "retry_reason": clean_reason if not passed else None,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[Critic] Evaluation failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)
        # On critic failure, fail the answer so the router can retry
        return {
            **state,
            "critic_passed": False,
            "retry_reason": "content",
            "critic_feedback": f"Critic evaluation failed: {e}",
            "stage_logs": state["stage_logs"] + [log_entry],
        }
