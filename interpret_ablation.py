"""
Interpretation helper: reads the v2 ablation summaries + per-query results
and produces a markdown report with the headline claims the paper can make.

Run after analyze_ablation.py produces ablation_v2_summary.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    results_dir = Path("evaluation/results")
    summary_file = results_dir / "ablation_v2_summary.json"
    if not summary_file.exists():
        print(f"Run analyze_ablation.py first to produce {summary_file}")
        return

    summary = json.load(open(summary_file))
    summaries = summary["summaries"]

    full = summaries.get("full", {})
    baseline = summaries.get("baseline_rag", {})

    if not full or not baseline:
        print("Need both 'full' and 'baseline_rag' configs to interpret.")
        return

    # Headline numbers
    full_metrics = full.get("metrics", {})
    base_metrics = baseline.get("metrics", {})

    lines = []
    lines.append("# AC-RAG Ablation v2 — Interpretation\n")
    lines.append(f"Test set: 46 questions (annotated, 40+ with relevant chunk ground truth)\n")
    lines.append(f"Configs compared: {', '.join(summaries.keys())}\n")
    lines.append("\n## Headline numbers (full AC-RAG vs naive baseline RAG)\n")
    lines.append("| Metric | Baseline RAG | AC-RAG full | Δ |")
    lines.append("|---|---:|---:|---:|")
    for m in ["faithfulness", "answer_relevance", "completeness",
              "context_utilisation", "rouge_l", "composite",
              "mrr", "recall@4", "recall@6", "recall@12", "precision@4", "precision@12"]:
        bv = base_metrics.get(m, {}).get("mean")
        fv = full_metrics.get(m, {}).get("mean")
        if bv is None and fv is None:
            continue
        bv_s = f"{bv:.3f}" if bv is not None else "—"
        fv_s = f"{fv:.3f}" if fv is not None else "—"
        d_s = f"{(fv - bv):+.3f}" if (fv is not None and bv is not None) else "—"
        lines.append(f"| {m} | {bv_s} | {fv_s} | {d_s} |")

    # What the data says we can claim
    lines.append("\n## What this data supports (or doesn't)\n")
    full_pass = full.get("critic_pass_rate", 0) * 100
    base_pass = baseline.get("critic_pass_rate", 0) * 100
    lines.append(f"- **Critic pass rate**: baseline={base_pass:.1f}%, full={full_pass:.1f}% "
                 f"({'higher' if full_pass > base_pass else 'lower'} in full)")

    # Find which config wins on each key metric
    if "mrr" in full_metrics:
        lines.append("\n### Best config per metric (mean)\n")
        lines.append("| Metric | Best config | Value |")
        lines.append("|---|---|---:|")
        for m in ["composite", "faithfulness", "answer_relevance", "mrr", "recall@4", "recall@12"]:
            best_cfg, best_val = None, -1
            for cfg, s in summaries.items():
                v = s.get("metrics", {}).get(m, {}).get("mean")
                if v is not None and v > best_val:
                    best_val, best_cfg = v, cfg
            if best_cfg:
                lines.append(f"| {m} | {best_cfg} | {best_val:.3f} |")

    # Latency / cost
    lines.append("\n### Latency and cost (per query)\n")
    lines.append("| Config | Avg latency | Avg cost |")
    lines.append("|---|---:|---:|")
    for cfg, s in summaries.items():
        lat = s.get("avg_latency_s", 0)
        cost = s.get("avg_cost_usd_per_query")
        cost_s = f"${cost:.5f}" if cost is not None else "—"
        lines.append(f"| {cfg} | {lat:.2f}s | {cost_s} |")

    # Save
    out = results_dir / "ablation_v2_interpretation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print()
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
