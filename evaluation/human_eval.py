"""
evaluation/human_eval.py
Human evaluation export/scoring for AC-RAG.

Gap fix (GAPS_AND_IMPROVEMENTS.md item B.12): all existing metrics are automated
(embedding cosine similarity, LLM-judge, ROUGE-L). A small human-rated sample adds
a grounding check the automated metrics can't provide on their own, and reporting
correlation between human and automated scores is itself evidence for (or against)
trusting the automated metrics at scale.

Workflow:
  1. Export a sample of queries from an existing results JSON to a rating sheet:
       python -m evaluation.human_eval export --results evaluation/results/results_full_<ts>.json --n 15
     Produces evaluation/results/human_eval_sheet_<ts>.csv with blank rating columns
     for two independent raters.
  2. Two people independently fill in the *_faithfulness and *_usefulness columns
     (1-5 scale, see column headers) without seeing each other's ratings or the
     automated scores.
  3. Score the filled-in sheet:
       python -m evaluation.human_eval score --sheet evaluation/results/human_eval_sheet_<ts>.csv
     Reports: mean human scores, inter-rater agreement (Pearson correlation between
     the two raters), and correlation between mean human faithfulness and the
     automated faithfulness metric already recorded in the results JSON.

Rating scale (both dimensions, 1-5):
  Faithfulness — is every claim in the answer actually supported by the source
    document (independent of what the automated metric says)?
  Usefulness   — would this answer actually satisfy someone who asked the question?
    (Faithfulness and usefulness are deliberately separate: an answer can be fully
    faithful but unhelpfully vague, or useful-sounding but subtly wrong.)
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from config.settings import BASE_DIR

RESULTS_DIR = BASE_DIR / "evaluation" / "results"

RATING_COLUMNS = [
    "rater1_faithfulness", "rater1_usefulness",
    "rater2_faithfulness", "rater2_usefulness",
]


def export_sheet(results_path: str, n: int, seed: int = 42) -> Path:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    scoreable = [r for r in results if r.get("generated_answer")]
    if len(scoreable) < n:
        print(f"Warning: only {len(scoreable)} answered queries available, requested {n}.")
        n = len(scoreable)

    random.Random(seed).shuffle(scoreable)
    sample = scoreable[:n]

    ts = Path(results_path).stem.replace("results_", "")
    out_path = RESULTS_DIR / f"human_eval_sheet_{ts}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "query", "generated_answer", "reference_answer",
            "automated_faithfulness", "automated_composite",
            *RATING_COLUMNS, "notes",
        ])
        for r in sample:
            m = r.get("metrics") or {}
            writer.writerow([
                r.get("id"),
                r.get("query"),
                r.get("generated_answer"),
                r.get("reference_answer") or "",
                m.get("faithfulness", ""),
                m.get("composite", ""),
                "", "", "", "",  # blank rating columns for the two raters
                "",
            ])

    print(f"Exported {len(sample)} queries -> {out_path}")
    print("Rating scale (1-5) for both faithfulness and usefulness - see module docstring.")
    print("Have two people fill in rater1_*/rater2_* columns independently, then run:")
    print(f"  python -m evaluation.human_eval score --sheet {out_path}")
    return out_path


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 if undefined (zero variance)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    return round(cov / denom, 4) if denom else 0.0


def score_sheet(sheet_path: str) -> Dict[str, Any]:
    with open(sheet_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def _floats(col: str) -> List[float]:
        return [float(r[col]) for r in rows if r.get(col, "").strip()]

    r1_faith = _floats("rater1_faithfulness")
    r2_faith = _floats("rater2_faithfulness")
    r1_use = _floats("rater1_usefulness")
    r2_use = _floats("rater2_usefulness")
    auto_faith = [
        float(r["automated_faithfulness"]) for r in rows
        if r.get("rater1_faithfulness", "").strip() and r.get("automated_faithfulness", "").strip()
    ]
    mean_human_faith_for_corr = [
        (float(r["rater1_faithfulness"]) + float(r["rater2_faithfulness"])) / 2
        for r in rows
        if r.get("rater1_faithfulness", "").strip() and r.get("rater2_faithfulness", "").strip()
        and r.get("automated_faithfulness", "").strip()
    ]

    n_rated = len(r1_faith)
    if n_rated == 0:
        print("No rated rows found — fill in rater1_*/rater2_* columns first.")
        return {}

    report = {
        "n_rated": n_rated,
        "mean_rater1_faithfulness": round(sum(r1_faith) / len(r1_faith), 3) if r1_faith else None,
        "mean_rater2_faithfulness": round(sum(r2_faith) / len(r2_faith), 3) if r2_faith else None,
        "mean_rater1_usefulness": round(sum(r1_use) / len(r1_use), 3) if r1_use else None,
        "mean_rater2_usefulness": round(sum(r2_use) / len(r2_use), 3) if r2_use else None,
        "inter_rater_correlation_faithfulness": _pearson(r1_faith, r2_faith) if len(r1_faith) == len(r2_faith) and r1_faith else None,
        "inter_rater_correlation_usefulness": _pearson(r1_use, r2_use) if len(r1_use) == len(r2_use) and r1_use else None,
        "human_vs_automated_faithfulness_correlation": (
            _pearson(mean_human_faith_for_corr, auto_faith) if len(mean_human_faith_for_corr) == len(auto_faith) and auto_faith else None
        ),
    }

    print("\n" + "=" * 50)
    print("  HUMAN EVALUATION REPORT")
    print("=" * 50)
    for k, v in report.items():
        print(f"  {k:<45}: {v}")
    print("=" * 50)
    print("\nInterpretation notes:")
    print("- Inter-rater correlation < 0.4 suggests the rating criteria need tightening")
    print("  (raters disagreeing this much means the rubric is ambiguous, not that the")
    print("  system is inconsistent).")
    print("- Human-vs-automated correlation tells you how much to trust the automated")
    print("  faithfulness metric at scale - low correlation is a real finding worth")
    print("  reporting, not something to hide.")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AC-RAG human evaluation export/scoring")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export a rating sheet from a results JSON")
    p_export.add_argument("--results", required=True, help="Path to results_<config>_<ts>.json")
    p_export.add_argument("--n", type=int, default=15, help="Number of queries to sample")
    p_export.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    p_score = sub.add_parser("score", help="Score a filled-in rating sheet")
    p_score.add_argument("--sheet", required=True, help="Path to the filled human_eval_sheet_*.csv")

    args = parser.parse_args()
    if args.cmd == "export":
        export_sheet(args.results, args.n, args.seed)
    elif args.cmd == "score":
        score_sheet(args.sheet)
