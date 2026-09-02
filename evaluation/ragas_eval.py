"""
evaluation/ragas_eval.py
RAGAS-equivalent evaluation for AC-RAG pipeline.

Implements the 4 core RAGAS metrics without the ragas library
(which requires Python 3.10+). All metrics use the same LLM
that the pipeline uses — resolved via llm_factory (ollama → groq → google → openai).

Metrics:
  1. faithfulness        — are all answer claims supported by the retrieved context?
  2. answer_relevancy    — does the answer address the question?
  3. context_precision   — are the retrieved chunks relevant to the question?
  4. context_recall      — does the retrieved context cover the ground truth? (needs reference)

Output:
  evaluation/results/ragas_results_<timestamp>.json
  evaluation/results/ragas_summary_<timestamp>.json

Usage:
    python -m evaluation.ragas_eval --test-set evaluation/test_set_sample.json
    python -m evaluation.ragas_eval --test-set evaluation/test_set.json
"""

import json
import sys
import time

# See evaluation/ablation.py for why: Windows consoles default to cp1252, which
# can't encode the arrow characters this module's print statements use.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger
from utils.llm_factory import get_llm
from config.settings import BASE_DIR
from pipeline.graph import run_pipeline
from vectorstore.store import VectorStoreManager

logger = setup_logger("ac_rag.ragas")

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

# ── LLM judge prompt helpers ──────────────────────────────────────────────────

def _llm_judge(prompt: str) -> str:
    """Call the pipeline LLM and return text response."""
    llm = get_llm()
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        logger.warning("[RAGAS] LLM judge failed: %s", e)
        return ""


def _parse_score(text: str, scale: int = 5) -> float:
    """Extract first integer 1–scale from LLM text, normalise to [0,1]."""
    import re
    matches = re.findall(r"\b([1-" + str(scale) + r"])\b", text)
    if matches:
        return round(int(matches[0]) / scale, 4)
    return 0.0


# ── Metric 1: Faithfulness ────────────────────────────────────────────────────

def ragas_faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    """
    RAGAS Faithfulness: fraction of answer claims that are supported by contexts.
    LLM-judge approach — same logic as the ragas library.
    Score in [0, 1].
    """
    if not answer or not contexts:
        return 0.0

    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""You are an evaluation judge for a RAG system.

Given the following CONTEXTS retrieved from documents:
{context_block}

And the following ANSWER to the question "{question}":
{answer}

Rate how faithful the answer is to the contexts on a scale of 1 to 5:
5 = All claims in the answer are fully supported by the contexts
4 = Most claims are supported, minor details may not be
3 = About half the claims are supported
2 = Few claims are supported by the contexts
1 = The answer contains mostly unsupported claims

Reply with ONLY a single digit (1, 2, 3, 4, or 5)."""

    result = _llm_judge(prompt)
    score = _parse_score(result)
    logger.debug("[RAGAS] faithfulness=%.4f (raw: %s)", score, result[:30])
    return score


# ── Metric 2: Answer Relevancy ────────────────────────────────────────────────

def ragas_answer_relevancy(question: str, answer: str) -> float:
    """
    RAGAS Answer Relevancy: does the answer directly address the question?
    LLM-judge approach. Score in [0, 1].
    """
    if not question or not answer:
        return 0.0

    prompt = f"""You are an evaluation judge for a RAG system.

Question: {question}

Answer: {answer}

Rate how relevant and directly the answer addresses the question on a scale of 1 to 5:
5 = The answer directly and completely addresses the question
4 = The answer mostly addresses the question with minor irrelevance
3 = The answer partially addresses the question
2 = The answer is mostly irrelevant to the question
1 = The answer does not address the question at all

Reply with ONLY a single digit (1, 2, 3, 4, or 5)."""

    result = _llm_judge(prompt)
    score = _parse_score(result)
    logger.debug("[RAGAS] answer_relevancy=%.4f (raw: %s)", score, result[:30])
    return score


# ── Metric 3: Context Precision ───────────────────────────────────────────────

def ragas_context_precision(question: str, contexts: List[str]) -> float:
    """
    RAGAS Context Precision: fraction of retrieved contexts that are
    actually relevant to answering the question.
    Score in [0, 1].
    """
    if not contexts:
        return 0.0

    relevant = 0
    for i, ctx in enumerate(contexts):
        prompt = f"""You are an evaluation judge.

Question: {question}

Retrieved passage:
{ctx[:600]}

Is this passage relevant to answering the question?
Reply with ONLY 'yes' or 'no'."""
        result = _llm_judge(prompt).lower()
        if "yes" in result:
            relevant += 1
        logger.debug("[RAGAS] context_precision chunk %d: %s", i, result[:10])

    score = round(relevant / len(contexts), 4)
    logger.debug("[RAGAS] context_precision=%.4f (%d/%d relevant)", score, relevant, len(contexts))
    return score


# ── Metric 4: Context Recall ──────────────────────────────────────────────────

def ragas_context_recall(question: str, contexts: List[str], ground_truth: str) -> float:
    """
    RAGAS Context Recall: does the retrieved context contain enough
    information to produce the ground truth answer?
    Requires a reference answer. Score in [0, 1].
    """
    if not contexts or not ground_truth:
        return 0.0

    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""You are an evaluation judge for a RAG system.

Question: {question}

Ground truth answer: {ground_truth}

Retrieved contexts:
{context_block[:2000]}

Rate how well the retrieved contexts cover the information needed to produce
the ground truth answer, on a scale of 1 to 5:
5 = The contexts contain all information needed to produce the ground truth
4 = The contexts contain most of the needed information
3 = The contexts contain about half the needed information
2 = The contexts contain little of the needed information
1 = The contexts do not contain information relevant to the ground truth

Reply with ONLY a single digit (1, 2, 3, 4, or 5)."""

    result = _llm_judge(prompt)
    score = _parse_score(result)
    logger.debug("[RAGAS] context_recall=%.4f (raw: %s)", score, result[:30])
    return score


# ── Per-sample evaluation ─────────────────────────────────────────────────────

def evaluate_sample(
    test_case: Dict[str, Any],
    vsm: VectorStoreManager,
) -> Dict[str, Any]:
    """Run the pipeline on one test case and compute all RAGAS metrics."""
    question = test_case["query"]
    ground_truth = test_case.get("reference_answer")

    logger.info("[RAGAS] Evaluating: '%s'", question[:80])
    t0 = time.perf_counter()

    try:
        state = run_pipeline(question, vsm=vsm)
        elapsed = round(time.perf_counter() - t0, 3)

        answer = state.get("answer") or ""
        retrieved_docs = state.get("retrieved_docs") or []
        contexts = [d["content"] for d in retrieved_docs if isinstance(d, dict) and d.get("content")]

        metrics: Dict[str, float] = {}
        metrics["faithfulness"]     = ragas_faithfulness(question, answer, contexts)
        metrics["answer_relevancy"] = ragas_answer_relevancy(question, answer)
        metrics["context_precision"] = ragas_context_precision(question, contexts)
        if ground_truth:
            metrics["context_recall"] = ragas_context_recall(question, contexts, ground_truth)
        metrics["composite"] = round(float(np.mean(list(metrics.values()))), 4)

        return {
            "id": test_case.get("id", "?"),
            "query": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "contexts_retrieved": len(contexts),
            "latency_seconds": elapsed,
            "metrics": metrics,
            "error": state.get("error"),
        }

    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 3)
        logger.error("[RAGAS] Pipeline failed: %s", e)
        return {
            "id": test_case.get("id", "?"),
            "query": question,
            "ground_truth": ground_truth,
            "generated_answer": None,
            "contexts_retrieved": 0,
            "latency_seconds": elapsed,
            "metrics": {},
            "error": str(e),
        }


# ── Batch runner + output ─────────────────────────────────────────────────────

def run_ragas_evaluation(test_set_path: str) -> List[Dict[str, Any]]:
    with open(test_set_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    vsm = VectorStoreManager()
    vsm.load()
    logger.info("[RAGAS] Loaded %d test cases", len(test_cases))

    results = []
    for i, case in enumerate(test_cases, 1):
        logger.info("[RAGAS] [%d/%d] %s", i, len(test_cases), case.get("id", "?"))
        result = evaluate_sample(case, vsm)
        results.append(result)
        # Print live progress
        m = result.get("metrics", {})
        print(f"  [{i}/{len(test_cases)}] {case.get('id','?'):8s} | "
              f"faith={m.get('faithfulness',0):.2f} "
              f"rel={m.get('answer_relevancy',0):.2f} "
              f"prec={m.get('context_precision',0):.2f} "
              f"comp={m.get('composite',0):.2f} "
              f"| {result['latency_seconds']:.1f}s")

    return results


def save_ragas_results(results: List[Dict[str, Any]]) -> Dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"ragas_results_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    metric_keys = set()
    for r in results:
        metric_keys.update(r.get("metrics", {}).keys())

    summary: Dict[str, Any] = {
        "total_queries": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "avg_latency_s": round(float(np.mean([r["latency_seconds"] for r in results])), 2),
        "metrics": {},
    }
    for key in sorted(metric_keys):
        vals = [r["metrics"][key] for r in results if key in r.get("metrics", {})]
        if vals:
            summary["metrics"][key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
                "min":  round(float(np.min(vals)), 4),
                "max":  round(float(np.max(vals)), 4),
            }

    summary_path = RESULTS_DIR / f"ragas_summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print final table
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary["metrics"].items():
        print(f"  {k:<22} mean={v['mean']:.4f}  std={v['std']:.4f}  "
              f"min={v['min']:.4f}  max={v['max']:.4f}")
    print(f"\n  Total queries : {summary['total_queries']}")
    print(f"  Errors        : {summary['errors']}")
    print(f"  Avg latency   : {summary['avg_latency_s']}s")
    print("=" * 60)
    print(f"\n  Results → {json_path.name}")
    print(f"  Summary → {summary_path.name}")

    return {"json": json_path, "summary": summary_path}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AC-RAG RAGAS-equivalent Evaluator")
    parser.add_argument("--test-set", required=True, help="Path to test set JSON")
    args = parser.parse_args()

    results = run_ragas_evaluation(args.test_set)
    save_ragas_results(results)
