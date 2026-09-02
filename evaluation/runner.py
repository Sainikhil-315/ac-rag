"""
evaluation/runner.py
Batch evaluation runner for the AC-RAG pipeline.

Test set format (JSON):
  [
    {
      "id": "q1",
      "query": "What is the accuracy of the proposed model?",
      "reference_answer": "The model achieved 94.3% accuracy.",  # optional
      "notes": "factual, single-doc"                             # optional
    },
    ...
  ]

Output (saved to evaluation/results/):
  - results_<timestamp>.json  : full per-query results with all metrics and pipeline state
  - results_<timestamp>.csv   : flat table for spreadsheet / pandas analysis
  - summary_<timestamp>.json  : aggregate statistics (mean, std per metric)

Usage:
    python -m evaluation.runner --test-set evaluation/test_set.json
    python -m evaluation.runner --test-set evaluation/test_set.json --config ablation_no_critic
"""

import csv
import json
import sys
import time
from datetime import datetime

# See evaluation/ablation.py for why: Windows consoles default to cp1252, which
# can't encode the arrow characters ("->") this module's log messages use.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger
from config.settings import BASE_DIR
from pipeline.graph import run_pipeline
from vectorstore.store import VectorStoreManager
from evaluation.metrics import compute_all_metrics

logger = setup_logger("ac_rag.eval")

RESULTS_DIR = BASE_DIR / "evaluation" / "results"


# ── Test set loader ───────────────────────────────────────────────────────────

def load_test_set(path: str) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Test set must be a JSON array.")
    # Normalise field names: support both legacy format ("query"/"reference_answer")
    # and the sample test set format ("question"/"answer").
    normalised = []
    for item in data:
        entry = dict(item)
        if "question" in entry and "query" not in entry:
            entry["query"] = entry.pop("question")
        if "answer" in entry and "reference_answer" not in entry:
            entry["reference_answer"] = entry.pop("answer")
        normalised.append(entry)
    logger.info("Loaded %d test cases from '%s'", len(normalised), path.name)
    return normalised


# ── Single query evaluation ───────────────────────────────────────────────────

def evaluate_single(
    test_case: Dict[str, Any],
    vsm: VectorStoreManager,
    config_name: str = "default",
) -> Dict[str, Any]:
    """
    Run the full pipeline for one test case and compute all metrics.
    Returns a result dict with pipeline outputs + metrics.
    """
    query = test_case["query"]
    reference = test_case.get("reference_answer")

    logger.info("[Runner] Evaluating: '%s'", query[:80])
    t0 = time.perf_counter()

    try:
        state = run_pipeline(query, vsm=vsm)
        elapsed = round(time.perf_counter() - t0, 3)

        answer = state.get("answer") or ""
        context = state.get("refined_context") or ""
        attribution = state.get("answer_with_attribution") or []
        retrieved_docs = state.get("retrieved_docs") or []
        critic_scores = state.get("critic_scores") or {}

        # Compute evaluation metrics
        metrics = compute_all_metrics(
            query=query,
            answer=answer,
            context=context,
            answer_with_attribution=attribution,
            total_retrieved=len(retrieved_docs),
            reference_answer=reference,
            retrieved_docs=retrieved_docs,
        )

        return {
            "id": test_case.get("id", "unknown"),
            "config": config_name,
            "query": query,
            "reference_answer": reference,
            "generated_answer": answer,
            "intent": state.get("intent"),
            "complexity_score": state.get("complexity_score"),
            "retry_count": state.get("retry_count", 0),
            "docs_retrieved": len(retrieved_docs),
            "docs_after_validation": len(state.get("scored_docs") or []),
            "critic_scores": dict(critic_scores),
            "critic_passed": state.get("critic_passed"),
            "error": state.get("error"),
            "latency_seconds": elapsed,
            "metrics": metrics,
            "token_usage": state.get("token_usage"),
            "stage_logs": state.get("stage_logs", []),
        }

    except Exception as e:
        elapsed = round(time.perf_counter() - t0, 3)
        logger.error("[Runner] Pipeline failed for query '%s': %s", query[:60], e)
        return {
            "id": test_case.get("id", "unknown"),
            "config": config_name,
            "query": query,
            "reference_answer": reference,
            "generated_answer": None,
            "error": str(e),
            "latency_seconds": elapsed,
            "metrics": {},
            "stage_logs": [],
        }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_evaluation(
    test_set: List[Dict[str, Any]],
    vsm: VectorStoreManager,
    config_name: str = "default",
) -> List[Dict[str, Any]]:
    """Run evaluation for all test cases. Returns list of result dicts."""
    results = []
    for i, case in enumerate(test_set, 1):
        logger.info("[Runner] [%d/%d] %s", i, len(test_set), case.get("id", "?"))
        result = evaluate_single(case, vsm, config_name)
        results.append(result)
    return results


# ── Output writers ────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(
    results: List[Dict[str, Any]],
    config_name: str = "default",
) -> Dict[str, Path]:
    """Save results to JSON and CSV. Returns paths of saved files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()

    # ── JSON (full detail) ──────────────────────────────────────────────────
    json_path = RESULTS_DIR / f"results_{config_name}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("[Runner] Full results saved → %s", json_path.name)

    # ── CSV (flat table for analysis) ──────────────────────────────────────
    csv_path = RESULTS_DIR / f"results_{config_name}_{ts}.csv"
    if results:
        flat_rows = []
        for r in results:
            row = {
                "id": r.get("id"),
                "config": r.get("config"),
                "query": r.get("query", "")[:120],
                "intent": r.get("intent"),
                "complexity": r.get("complexity_score"),
                "retry_count": r.get("retry_count", 0),
                "docs_retrieved": r.get("docs_retrieved", 0),
                "docs_validated": r.get("docs_after_validation", 0),
                "critic_passed": r.get("critic_passed"),
                "latency_s": r.get("latency_seconds"),
                "error": r.get("error"),
                "cost_usd": (r.get("token_usage") or {}).get("total_cost_usd"),
                "llm_calls": (r.get("token_usage") or {}).get("llm_calls"),
            }
            # Flatten metrics
            for k, v in (r.get("metrics") or {}).items():
                row[f"metric_{k}"] = v
            # Flatten critic scores
            for k, v in (r.get("critic_scores") or {}).items():
                if k != "feedback":
                    row[f"critic_{k}"] = v
            flat_rows.append(row)

        fieldnames = list(flat_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    logger.info("[Runner] CSV results saved → %s", csv_path.name)

    # ── Summary JSON ────────────────────────────────────────────────────────
    summary = _compute_summary(results, config_name)
    summary_path = RESULTS_DIR / f"summary_{config_name}_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("[Runner] Summary saved → %s", summary_path.name)

    return {"json": json_path, "csv": csv_path, "summary": summary_path}


def _compute_summary(
    results: List[Dict[str, Any]], config_name: str
) -> Dict[str, Any]:
    """Compute aggregate statistics over all results."""
    import numpy as np

    metric_keys = set()
    for r in results:
        metric_keys.update((r.get("metrics") or {}).keys())

    agg: Dict[str, Any] = {
        "config": config_name,
        "total_queries": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "critic_pass_rate": round(
            sum(1 for r in results if r.get("critic_passed")) / max(len(results), 1), 4
        ),
        "avg_latency_s": round(
            float(np.mean([r.get("latency_seconds", 0) for r in results])), 3
        ),
        "avg_retry_count": round(
            float(np.mean([r.get("retry_count", 0) for r in results])), 3
        ),
        "metrics": {},
    }

    costs = [
        r["token_usage"]["total_cost_usd"]
        for r in results
        if r.get("token_usage") and r["token_usage"].get("total_cost_usd") is not None
    ]
    if costs:
        agg["avg_cost_usd_per_query"] = round(float(np.mean(costs)), 6)
        agg["total_cost_usd"] = round(float(sum(costs)), 6)

    for key in sorted(metric_keys):
        values = [
            r["metrics"][key]
            for r in results
            if r.get("metrics") and key in r["metrics"]
        ]
        if values:
            agg["metrics"][key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
            }

    return agg


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AC-RAG Batch Evaluator")
    parser.add_argument("--test-set", required=True, help="Path to test set JSON file")
    parser.add_argument("--config", default="default", help="Config name tag for output files")
    args = parser.parse_args()

    vsm = VectorStoreManager()
    vsm.load()

    test_cases = load_test_set(args.test_set)
    results = run_evaluation(test_cases, vsm, config_name=args.config)
    paths = save_results(results, config_name=args.config)

    logger.info("Evaluation complete. Files written:")
    for k, p in paths.items():
        logger.info("  %s → %s", k, p)
