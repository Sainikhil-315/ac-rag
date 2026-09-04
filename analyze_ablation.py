"""
Post-ablation analysis: read summary_*.json files, compute paired t-tests
between configs, and print a paper-ready comparison.

Used to produce the v2 ablation results table once the 8-config run completes.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_all_summaries(results_dir: Path) -> dict:
    """Load every summary_*.json in the directory, keyed by config name."""
    out = {}
    for f in sorted(results_dir.glob("summary_*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = data.get("config", f.stem)
        out[cfg] = data
    return out


def load_per_query_metrics(results_dir: Path) -> dict:
    """Load per-query metric dicts for each config (for paired tests)."""
    out = {}
    for f in sorted(results_dir.glob("results_*.json")):
        if "summary" in f.name:
            continue
        with open(f, encoding="utf-8") as fh:
            rows = json.load(fh)
        if not rows:
            continue
        cfg = rows[0].get("config", f.stem)
        # Align by id — paired t-test needs same queries
        per_id = {}
        for r in rows:
            per_id[r.get("id")] = r.get("metrics") or {}
        out[cfg] = per_id
    return out


def paired_test(values_a: list, values_b: list) -> tuple:
    """Paired t-test. Returns (mean_a, mean_b, mean_diff, p_value, n)."""
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return (float(a.mean()) if len(a) else 0,
                float(b.mean()) if len(b) else 0, 0, 1.0, n)
    diffs = a[:n] - b[:n]
    t, p = stats.ttest_rel(a[:n], b[:n])
    return (float(a.mean()), float(b.mean()), float(diffs.mean()), float(p), n)


def main():
    results_dir = Path("evaluation/results")
    summaries = load_all_summaries(results_dir)
    per_query = load_per_query_metrics(results_dir)

    if not summaries:
        print("No summary files found.")
        return

    # Order: full first, baseline_rag second, then the rest
    order = ["full", "baseline_rag"] + [k for k in summaries if k not in ("full", "baseline_rag")]

    # ── Table 1: aggregate metrics ──────────────────────────────────────────
    metric_keys = ["faithfulness", "answer_relevance", "completeness",
                   "context_utilisation", "rouge_l", "composite",
                   "mrr", "recall@4", "recall@6", "recall@12",
                   "precision@4", "precision@12"]

    short = {
        "faithfulness": "Faith", "answer_relevance": "AnsRel",
        "completeness": "Compl", "context_utilisation": "CtxUtl",
        "rouge_l": "ROUGE-L", "composite": "Compos",
        "mrr": "MRR", "recall@4": "R@4", "recall@6": "R@6",
        "recall@12": "R@12", "precision@4": "P@4", "precision@12": "P@12",
    }

    print("=" * 110)
    print("AC-RAG ABLATION — v2 (annotated test set, 46 Qs, retrieval metrics included)")
    print("=" * 110)
    hdr = f"{'Config':<18}"
    for k in metric_keys:
        if any(k in (s.get("metrics") or {}) for s in summaries.values()):
            hdr += f"{short.get(k, k)[:8]:>9}"
    hdr += f"{'Pass%':>8}{'Lat(s)':>8}{'$/q':>10}"
    print(hdr)
    print("-" * len(hdr))

    for cfg in order:
        s = summaries.get(cfg, {})
        row = f"{cfg:<18}"
        for k in metric_keys:
            v = s.get("metrics", {}).get(k, {}).get("mean")
            row += f"{v:>9.3f}" if v is not None else f"{'-':>9}"
        row += f"{s.get('critic_pass_rate', 0)*100:>7.1f}%"
        row += f"{s.get('avg_latency_s', 0):>8.2f}"
        cost = s.get("avg_cost_usd_per_query")
        row += f"{cost:>10.5f}" if cost is not None else f"{'-':>10}"
        print(row)
    print("=" * 110)

    # ── Table 2: paired t-tests, full vs everything else ───────────────────
    if "full" in per_query:
        full_q = per_query["full"]
        print("\nPaired t-tests: full vs each other config (n = number of paired queries)")
        print("-" * 110)
        print(f"{'Comparison':<28}{'Metric':<14}{'Mean A':>9}{'Mean B':>9}{'Δ':>9}{'p':>10}{'n':>5}{'sig':>6}")
        print("-" * 110)
        for other in order:
            if other == "full" or other not in per_query:
                continue
            other_q = per_query[other]
            shared = sorted(set(full_q.keys()) & set(other_q.keys()))
            for m in ["composite", "faithfulness", "answer_relevance", "mrr", "recall@12"]:
                a_vals = [full_q[i].get(m) for i in shared if full_q[i].get(m) is not None]
                b_vals = [other_q[i].get(m) for i in shared if other_q[i].get(m) is not None]
                if len(a_vals) < 5:
                    continue
                ma, mb, d, p, n = paired_test(a_vals, b_vals)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                print(f"{'full vs ' + other:<28}{m:<14}{ma:>9.3f}{mb:>9.3f}{d:>+9.3f}{p:>10.4f}{n:>5}{sig:>6}")
        print("-" * 110)
        print("significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # ── Save machine-readable comparison ──────────────────────────────────
    out = {
        "summaries": summaries,
        "metric_keys": metric_keys,
    }
    with open(results_dir / "ablation_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSummary saved to {results_dir / 'ablation_v2_summary.json'}")


if __name__ == "__main__":
    main()
