"""
evaluation/metrics.py
Individual metric functions for AC-RAG evaluation.

Metrics implemented:
  1. faithfulness_score   : embedding cosine similarity (answer sentences ↔ context)
                            Quantifies Faith(a, R(q,D)) from the formal objective
  2. answer_relevance     : cosine similarity between answer and query
  3. rouge_l              : ROUGE-L F1 (lexical overlap vs reference answer)
  4. completeness_score   : LLM-judge (1–5) assessing coverage of the question
  5. context_utilisation  : fraction of retrieved passages cited in the answer

Design:
  - Metrics 1, 2, 5 are reference-free (no gold answer needed) → can run on any query
  - Metrics 3, 4 require a reference answer → used when a test set is available
  - All scores normalised to [0, 1] for uniform comparison in the ablation report
  - BERTScore is excluded: too heavy for research iteration; ROUGE-L + embeddings
    cover the semantic similarity ground adequately
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from utils.scoring import score_passages_against_query
from utils.attribution import build_source_registry

logger = logging.getLogger(__name__)


# ── 1. Faithfulness Score ─────────────────────────────────────────────────────

def faithfulness_score(answer: str, context: str) -> float:
    """
    Measures how grounded the answer is in the retrieved context.
    Implements Faith(a, R(q,D)) from the formal objective.

    Method:
      - Split answer into sentences
      - For each sentence, compute max cosine similarity to any context passage
      - Faithfulness = mean of per-sentence max similarities

    Returns float in [0, 1]. Values < FAITHFULNESS_THRESHOLD (0.80) are unacceptable.
    """
    if not answer or not context:
        return 0.0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0

    # Extract passage texts from context (split on SOURCE tag separators)
    passages = [p.strip() for p in re.split(r"\[SOURCE \d+[^\]]*\]", context) if p.strip()]
    if not passages:
        return 0.0

    # For each sentence, find max similarity to any passage
    per_sentence_scores = []
    for sentence in sentences:
        try:
            scores = score_passages_against_query(sentence, passages)
            per_sentence_scores.append(max(scores) if scores else 0.0)
        except Exception as e:
            logger.warning("[Metrics] Faithfulness scoring failed for sentence: %s", e)
            per_sentence_scores.append(0.0)

    faith = float(np.mean(per_sentence_scores))
    logger.debug("[Metrics] faithfulness=%.4f (%d sentences)", faith, len(sentences))
    return round(faith, 4)


# ── 2. Answer Relevance ───────────────────────────────────────────────────────

def answer_relevance_score(query: str, answer: str) -> float:
    """
    Measures how directly the answer addresses the query.
    Uses cosine similarity between query embedding and answer embedding.

    Returns float in [0, 1].
    """
    if not query or not answer:
        return 0.0

    try:
        scores = score_passages_against_query(query, [answer])
        relevance = scores[0] if scores else 0.0
        logger.debug("[Metrics] answer_relevance=%.4f", relevance)
        return round(relevance, 4)
    except Exception as e:
        logger.warning("[Metrics] Answer relevance failed: %s", e)
        return 0.0


# ── 3. ROUGE-L ────────────────────────────────────────────────────────────────

def rouge_l_score(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 between generated answer and reference answer.
    Requires a reference answer — used in test-set evaluation.

    Returns float in [0, 1].
    """
    if not hypothesis or not reference:
        return 0.0

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(reference, hypothesis)
        score = result["rougeL"].fmeasure
        logger.debug("[Metrics] rouge_l=%.4f", score)
        return round(float(score), 4)
    except ImportError:
        logger.warning("[Metrics] rouge_score not installed. Run: pip install rouge-score")
        return 0.0
    except Exception as e:
        logger.warning("[Metrics] ROUGE-L failed: %s", e)
        return 0.0


# ── 4. Completeness (LLM Judge) ───────────────────────────────────────────────

def completeness_score(query: str, answer: str) -> float:
    """
    LLM-judge completeness score on 1–5 scale, normalised to [0, 1].
    Assesses whether the answer covers all key aspects of the question.

    Reference-free: does not require a gold answer.
    Returns float in [0, 1] (score 5 → 1.0, score 1 → 0.2).
    """
    if not query or not answer:
        return 0.0

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field
        from utils.llm_factory import get_llm

        class CompletenessJudge(BaseModel):
            score: int = Field(ge=1, le=5, description="Completeness score 1–5")
            reason: str = Field(description="Brief reason for the score")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an evaluation judge. Score how completely the answer addresses "
             "the question on a scale of 1–5.\n"
             "5=fully complete, addresses all key aspects\n"
             "4=mostly complete, minor gaps\n"
             "3=partially complete, notable gaps\n"
             "2=incomplete, major aspects missing\n"
             "1=does not address the question"),
            ("human", "Question: {query}\n\nAnswer: {answer}\n\nScore the completeness."),
        ])

        chain = prompt | get_llm().with_structured_output(CompletenessJudge)
        result: CompletenessJudge = chain.invoke({"query": query, "answer": answer})

        normalised = round(result.score / 5.0, 4)
        logger.debug("[Metrics] completeness=%d/5 (%.4f)", result.score, normalised)
        return normalised

    except Exception as e:
        logger.warning("[Metrics] Completeness LLM judge failed: %s", e)
        return 0.0


# ── 5. Context Utilisation ────────────────────────────────────────────────────

def context_utilisation(
    answer_with_attribution: List[Dict[str, Any]],
    total_retrieved: int,
    answer: str = "",
    retrieved_docs: Optional[List[Dict]] = None,
) -> float:
    """
    Fraction of retrieved passages that contributed to the answer.

    Primary method (citation-based):
      Counts distinct chunk_ids cited via [N] markers in the answer.
      Requires the generator to produce inline citations.

    Fallback method (semantic overlap):
      When no citations are present — e.g. the model skipped [N] markers —
      this counts a passage as "used" if its cosine similarity to the full
      answer exceeds 0.35.  This keeps the metric meaningful regardless of
      whether the LLM follows citation instructions.

    Returns float in [0, 1].
    """
    if not total_retrieved:
        return 0.0

    # ── Primary: citation-based ───────────────────────────────────────────────
    cited_sources = set()
    for entry in (answer_with_attribution or []):
        for source in entry.get("sources", []):
            cited_sources.add(source.get("chunk_id"))
    cited_sources.discard(None)

    if cited_sources:
        utilisation = round(len(cited_sources) / total_retrieved, 4)
        logger.debug(
            "[Metrics] context_utilisation=%.4f (citation-based, %d/%d passages cited)",
            utilisation, len(cited_sources), total_retrieved,
        )
        return utilisation

    # ── Fallback: semantic overlap ────────────────────────────────────────────
    # Fires when the generator omitted [N] citations (model-dependent behaviour).
    if answer and retrieved_docs:
        try:
            _SEMANTIC_THRESHOLD = 0.35
            used_count = 0
            for doc in retrieved_docs:
                content = doc.get("content", "") if isinstance(doc, dict) else ""
                if not content:
                    continue
                scores = score_passages_against_query(answer, [content])
                if scores and scores[0] >= _SEMANTIC_THRESHOLD:
                    used_count += 1
            utilisation = round(used_count / total_retrieved, 4)
            logger.debug(
                "[Metrics] context_utilisation=%.4f (semantic fallback, %d/%d passages above %.2f)",
                utilisation, used_count, total_retrieved, _SEMANTIC_THRESHOLD,
            )
            return utilisation
        except Exception as e:
            logger.warning("[Metrics] Semantic utilisation fallback failed: %s", e)

    return 0.0


# ── 6. Retrieval Quality: Recall@k and MRR ────────────────────────────────────
#
# These are the metrics that directly evaluate the Retrieval Planner's adaptive-k
# design — none of the answer-side metrics (faithfulness, relevance) can tell you
# whether retrieving 4 vs 12 passages is the right call, but Recall@k and MRR can.
#
# Ground truth format: for each query, a list of "relevant chunk_ids" (or any
# string IDs) the system SHOULD have retrieved. Annotated manually on the test
# set, or auto-derived from the test set's reference answer (e.g. any chunk whose
# text contains ≥40% of the reference answer's tokens is considered relevant).
#
# For a research-grade claim about adaptive retrieval, you want:
#   - Recall@4  (does the planner's minimum-k still find the answer for simple queries?)
#   - Recall@12 (does the planner's maximum-k recover the answer for complex queries?)
#   - MRR       (how high does the FIRST relevant chunk rank? — latency-sensitive)
#   - Precision@k (fraction of retrieved chunks that are actually relevant)

def recall_at_k(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int,
) -> float:
    """
    Recall@k = (relevant chunks in top-k) / (total relevant chunks).

    Returns 1.0 if the query has no annotated relevant chunks (vacuously true —
    the test set entry is treated as un-evaluable for retrieval, not scored 0).
    Returns 0.0 if there are relevant chunks but none are in the top-k.

    Args:
        retrieved_chunk_ids: ordered list of chunk_ids the system actually returned
                             (top of list = highest rank).
        relevant_chunk_ids:  ground-truth list of chunk_ids that answer the query.
        k:                   cut-off (we use the first k of retrieved_chunk_ids).
    """
    if not relevant_chunk_ids:
        return 1.0
    if not retrieved_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    top_k = retrieved_chunk_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return round(hits / len(relevant_set), 4)


def mrr(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
) -> float:
    """
    Mean Reciprocal Rank: 1 / (rank of first relevant chunk).
    Returns 0.0 if no relevant chunk is retrieved.
    Returns 1.0 (not 0.0) if there are no annotated relevant chunks — same
    convention as recall_at_k; an un-evaluable query is not punished.
    """
    if not relevant_chunk_ids:
        return 1.0
    if not retrieved_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def precision_at_k(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int,
) -> float:
    """
    Precision@k = (relevant chunks in top-k) / k.
    Measures how much of what the retriever returned was actually useful —
    a high-precision retriever is one that doesn't waste context-window budget
    on irrelevant passages (relevant for context_utilisation downstream).
    """
    if k <= 0 or not retrieved_chunk_ids:
        return 0.0
    if not relevant_chunk_ids:
        return 1.0  # no ground truth to falsify against

    relevant_set = set(relevant_chunk_ids)
    top_k = retrieved_chunk_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return round(hits / k, 4)


def retrieval_metrics(
    retrieved_docs: List[Dict[str, Any]],
    relevant_chunk_ids: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Recall@k, MRR, and Precision@k for a single query's retrieval.

    Args:
        retrieved_docs:       list of dicts with a "chunk_id" key (the retriever
                              node's output shape).
        relevant_chunk_ids:   ground-truth list of relevant chunk_ids. If None
                              or empty, the function returns an empty dict
                              (caller decides whether to skip the query or
                              report None — the eval runner reports None).
        k_values:             cut-offs to compute Recall@k / Precision@k for.
                              Defaults to [4, 6, 12] — the planner's min,
                              default, and max k.

    Returns:
        Dict of metric_name → float, e.g. {"recall@4": 0.5, "recall@12": 1.0,
        "mrr": 0.5, "precision@4": 0.5, "precision@6": 0.33, "precision@12": 0.25}.
        Returns {} when relevant_chunk_ids is None/empty (no ground truth).
    """
    if not relevant_chunk_ids:
        return {}

    k_values = k_values or [4, 6, 12]
    retrieved_ids = [
        d.get("chunk_id", "") for d in (retrieved_docs or []) if d.get("chunk_id")
    ]

    out: Dict[str, float] = {"mrr": mrr(retrieved_ids, relevant_chunk_ids)}
    for k in k_values:
        out[f"recall@{k}"]    = recall_at_k(retrieved_ids, relevant_chunk_ids, k)
        out[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_chunk_ids, k)
    return out


# ── Composite score ───────────────────────────────────────────────────────────

def compute_all_metrics(
    query: str,
    answer: str,
    context: str,
    answer_with_attribution: List[Dict[str, Any]],
    total_retrieved: int,
    reference_answer: Optional[str] = None,
    retrieved_docs: Optional[List[Dict]] = None,
    relevant_chunk_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all available metrics for a single query-answer pair.
    Returns a dict of metric_name → score (all in [0, 1]).

    When `relevant_chunk_ids` is provided, retrieval-quality metrics
    (Recall@k, MRR, Precision@k) are included — these directly evaluate the
    Retrieval Planner's adaptive-k design and are the headline numbers for
    the paper's central novelty claim.
    """
    results: Dict[str, float] = {}

    results["faithfulness"]        = faithfulness_score(answer, context)
    results["answer_relevance"]    = answer_relevance_score(query, answer)
    results["completeness"]        = completeness_score(query, answer)
    results["context_utilisation"] = context_utilisation(
        answer_with_attribution,
        total_retrieved,
        answer=answer,
        retrieved_docs=retrieved_docs,
    )

    if reference_answer:
        results["rouge_l"] = rouge_l_score(answer, reference_answer)

    # Retrieval-quality metrics — only if ground truth provided
    if relevant_chunk_ids:
        results.update(retrieval_metrics(retrieved_docs, relevant_chunk_ids))

    # Composite: mean of all available scores (excludes retrieval metrics
    # if no ground truth was provided, so the composite stays comparable
    # across queries with and without relevance annotations)
    if results:
        results["composite"] = round(float(np.mean(list(results.values()))), 4)

    return results
