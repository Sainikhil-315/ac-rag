"""
evaluation/ablation.py
Automated ablation study runner.

Ablation configurations tested (see ABLATION_CONFIGS below for the source of truth):
  1. baseline_rag    : naive RAG — planner/validator/refiner/critic all OFF (the "before")
  2. full            : all components enabled — the actual AC-RAG pipeline (the "after")
  3. no_planner      : USE_RETRIEVAL_PLANNER=False  → rule-based k only
  4. no_validator    : USE_VALIDATOR=False           → all passages pass unfiltered
  5. no_refiner      : USE_CONTEXT_REFINER=False     → raw docs sent to generator
  6. no_critic       : USE_CRITIC=False              → no self-reflection loop
  7. low_k           : fixed k=4 regardless of query complexity
  8. high_k          : fixed k=12 regardless of query complexity

`baseline_rag` vs `full` is the headline research comparison — the report's
"AC-RAG reduces hallucination vs. naive RAG" claim is only as good as this delta,
which is auto-computed and printed by _build_comparison_table() below.

Each configuration runs the full evaluation suite and saves its own result files.
A comparison report is printed and saved at the end.

Usage:
    python -m evaluation.ablation --test-set evaluation/test_set.json
"""

import importlib
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Windows consoles default to cp1252, which can't encode the arrow characters used in
# the comparison table below (real crash observed 2026-08-20: all 8 configs finished
# and saved successfully, but the final _print_comparison_table() call crashed the
# whole script before the comparison JSON/table files got written — same root cause
# as the fix already applied to main.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger
from config import settings as _settings_module
from evaluation.runner import load_test_set, run_evaluation, save_results, RESULTS_DIR
from vectorstore.store import VectorStoreManager

import pipeline.router as _router_mod
import pipeline.nodes.validator as _validator_mod
import pipeline.nodes.critic as _critic_mod
import pipeline.nodes.context_refiner as _context_refiner_mod
import pipeline.nodes.retrieval_planner as _retrieval_planner_mod
import pipeline.nodes.retriever as _retriever_mod

logger = setup_logger("ac_rag.ablation")

# CRITICAL: every ablation-relevant flag (USE_VALIDATOR, USE_CRITIC, RETRIEVAL_K_MAX,
# etc.) is imported into its consuming module with `from config.settings import X` —
# this binds a COPY of the value into that module's namespace at import time. Patching
# config.settings.X afterwards (setattr on the settings module) does NOT change the
# already-bound name in validator.py/critic.py/router.py/etc. — Python's import
# semantics don't create a live link back to the source module's attribute.
#
# This was a real, previously undetected bug: every ablation config silently ran with
# whatever flags were live at each node module's import time, regardless of the
# config's intended overrides (found 2026-08-20 while a full 8-config ablation run was
# in progress — stopped it immediately, since the in-flight results were meaningless).
#
# Fix: patch every dependent module's already-bound name directly, in addition to
# config.settings itself (kept in sync for any code that reads it fresh/dotted).
_DEPENDENT_MODULES = [
    _router_mod, _validator_mod, _critic_mod,
    _context_refiner_mod, _retrieval_planner_mod, _retriever_mod,
]


# ── Ablation configuration registry ──────────────────────────────────────────

# Each entry: {name, description, overrides: {settings_attr: value}}
ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "baseline_rag",
        "description": "Simple RAG — no agents, no planner, no validator, no refiner, no critic",
        "overrides": {
            "USE_RETRIEVAL_PLANNER": False,
            "USE_VALIDATOR": False,
            "USE_CONTEXT_REFINER": False,
            "USE_CRITIC": False,
            "RETRIEVAL_K_DEFAULT": 6,
        },
    },
    {
        "name": "full",
        "description": "Full AC-RAG pipeline — all components enabled",
        "overrides": {
            "USE_RETRIEVAL_PLANNER": True,
            "USE_VALIDATOR": True,
            "USE_CONTEXT_REFINER": True,
            "USE_CRITIC": True,
        },
    },
    {
        "name": "no_planner",
        "description": "Retrieval planner disabled — rule-based k only",
        "overrides": {
            "USE_RETRIEVAL_PLANNER": False,
            "USE_VALIDATOR": True,
            "USE_CONTEXT_REFINER": True,
            "USE_CRITIC": True,
        },
    },
    {
        "name": "no_validator",
        "description": "Evidence validator disabled — all passages pass unfiltered",
        "overrides": {
            "USE_RETRIEVAL_PLANNER": True,
            "USE_VALIDATOR": False,
            "USE_CONTEXT_REFINER": True,
            "USE_CRITIC": True,
        },
    },
    {
        "name": "no_refiner",
        "description": "Context refiner disabled — raw validated docs go to generator",
        "overrides": {
            "USE_RETRIEVAL_PLANNER": True,
            "USE_VALIDATOR": True,
            "USE_CONTEXT_REFINER": False,
            "USE_CRITIC": True,
        },
    },
    {
        "name": "no_critic",
        "description": "Critic disabled — no self-reflection or retry loop",
        "overrides": {
            "USE_RETRIEVAL_PLANNER": True,
            "USE_VALIDATOR": True,
            "USE_CONTEXT_REFINER": True,
            "USE_CRITIC": False,
        },
    },
    {
        "name": "low_k",
        "description": "Retrieval k fixed at minimum (4) regardless of complexity",
        "overrides": {
            "RETRIEVAL_K_DEFAULT": 4,
            "RETRIEVAL_K_MAX": 4,
            "USE_RETRIEVAL_PLANNER": False,
            "USE_VALIDATOR": True,
            "USE_CONTEXT_REFINER": True,
            "USE_CRITIC": True,
        },
    },
    {
        "name": "high_k",
        "description": "Retrieval k fixed at maximum (12) regardless of complexity",
        "overrides": {
            "RETRIEVAL_K_DEFAULT": 12,
            "RETRIEVAL_K_MIN": 12,
            "USE_RETRIEVAL_PLANNER": False,
            "USE_VALIDATOR": True,
            "USE_CONTEXT_REFINER": True,
            "USE_CRITIC": True,
        },
    },
]


# ── Settings patcher ──────────────────────────────────────────────────────────

class _SettingsPatch:
    """
    Context manager that temporarily overrides settings values for an ablation run.
    Restores originals on exit — safe for sequential ablation runs.

    Patches BOTH config.settings itself AND every dependent node/router module's
    already-imported copy of each attribute (see _DEPENDENT_MODULES above) — patching
    config.settings alone silently does nothing, since those modules imported their
    own bound copies at module-load time.

    Usage:
        with _SettingsPatch({"USE_CRITIC": False}):
            run_evaluation(...)
    """

    def __init__(self, overrides: Dict[str, Any]):
        self.overrides = overrides
        # {(module, attr): original_value}
        self._originals: Dict[tuple, Any] = {}

    def __enter__(self):
        for attr, value in self.overrides.items():
            targets = [_settings_module] + [m for m in _DEPENDENT_MODULES if hasattr(m, attr)]
            if len(targets) == 1:
                # Only config.settings has it — no consuming module imported this attr
                # by name, so nothing else needs patching (or the attr name is wrong).
                logger.warning(
                    "[Ablation] settings.%s has no consuming module bound to it — "
                    "verify this override actually affects pipeline behavior.", attr,
                )
            for mod in targets:
                if not hasattr(mod, attr):
                    continue
                self._originals[(mod, attr)] = getattr(mod, attr)
                setattr(mod, attr, value)
            logger.debug(
                "[Ablation] %s = %r patched on: %s",
                attr, value, [m.__name__ for m in targets if hasattr(m, attr)],
            )
        return self

    def __exit__(self, *args):
        for (mod, attr), original in self._originals.items():
            setattr(mod, attr, original)


# ── Comparison report ─────────────────────────────────────────────────────────

def _build_comparison_table(summaries: List[Dict[str, Any]]) -> str:
    """Build a formatted comparison table string."""
    if not summaries:
        return ""

    # Collect all metric keys across all summaries (some configs may have fewer)
    all_metric_keys: set = set()
    for s in summaries:
        all_metric_keys.update(s.get("metrics", {}).keys())
    metric_keys = sorted(all_metric_keys)

    col_w   = 16
    met_w   = 13

    # ── column headers ────────────────────────────────────────────────────────
    header_labels = ["faithfulness", "answer_rel", "completeness", "ctx_util", "rouge_l", "composite"]
    # Only keep metric keys that exist, in our preferred display order
    display_keys = [k for k in header_labels if k in metric_keys]
    # Add any remaining keys not in our preferred list
    for k in metric_keys:
        if k not in display_keys:
            display_keys.append(k)

    short = {
        "faithfulness":      "Faith",
        "answer_relevance":  "Ans.Rel",
        "completeness":      "Complete",
        "context_utilisation": "Ctx.Util",
        "rouge_l":           "ROUGE-L",
        "composite":         "Composite",
    }

    sep_len = col_w + met_w * len(display_keys) + 12 + 11
    sep = "=" * sep_len
    thin = "-" * sep_len

    lines = []
    lines.append(sep)
    lines.append("  AC-RAG EVALUATION — COMPARISON TABLE")
    lines.append(f"  Configs: {len(summaries)}   Questions: {summaries[0].get('total_queries', '?')}")
    lines.append(sep)

    hdr = f"{'Config':<{col_w}}"
    for k in display_keys:
        label = short.get(k, k[:8])
        hdr += f"{label:>{met_w}}"
    hdr += f"{'Pass%':>10}  {'Avg Lat':>8}"
    lines.append(hdr)
    lines.append(thin)

    for s in summaries:
        name = s["config"]
        tag  = " ← AC-RAG" if name == "full" else (" ← baseline" if name == "baseline_rag" else "")
        row  = f"{name:<{col_w}}"
        for k in display_keys:
            val = s.get("metrics", {}).get(k, {}).get("mean", None)
            row += f"{val:>{met_w}.4f}" if val is not None else f"{'—':>{met_w}}"
        row += f"{s.get('critic_pass_rate', 0)*100:>9.1f}%"
        row += f"  {s.get('avg_latency_s', 0):>6.1f}s"
        row += tag
        lines.append(row)

    lines.append(sep)

    # ── delta section: full vs baseline_rag ──────────────────────────────────
    full_s     = next((s for s in summaries if s["config"] == "full"), None)
    baseline_s = next((s for s in summaries if s["config"] == "baseline_rag"), None)
    if full_s and baseline_s:
        lines.append("")
        lines.append("  AC-RAG vs Baseline RAG (delta, positive = improvement)")
        lines.append(thin)
        delta_row = f"{'Δ (full − baseline)':<{col_w}}"
        for k in display_keys:
            ac  = full_s.get("metrics",     {}).get(k, {}).get("mean", None)
            bl  = baseline_s.get("metrics", {}).get(k, {}).get("mean", None)
            if ac is not None and bl is not None:
                d = ac - bl
                delta_row += f"{d:>+{met_w}.4f}"
            else:
                delta_row += f"{'—':>{met_w}}"
        delta_pass = (full_s.get("critic_pass_rate", 0) - baseline_s.get("critic_pass_rate", 0)) * 100
        delta_lat  = full_s.get("avg_latency_s", 0) - baseline_s.get("avg_latency_s", 0)
        delta_row += f"{delta_pass:>+9.1f}%"
        delta_row += f"  {delta_lat:>+6.1f}s"
        lines.append(delta_row)
        lines.append(sep)

    return "\n".join(lines)


def _print_comparison_table(summaries: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to the console."""
    print("\n" + _build_comparison_table(summaries))


# ── Main ablation runner ──────────────────────────────────────────────────────

def run_ablation_study(
    test_set_path: str,
    configs: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full ablation study: evaluate each configuration on the test set.

    Args:
        test_set_path : path to the JSON test set
        configs       : list of ablation configs to run (default: all ABLATION_CONFIGS)

    Returns:
        Dict with summaries per config and a comparison table.
    """
    configs = configs or ABLATION_CONFIGS
    test_cases = load_test_set(test_set_path)

    # Load vector store once — shared across all ablation runs
    vsm = VectorStoreManager()
    vsm.load()

    all_summaries: List[Dict[str, Any]] = []
    all_paths: Dict[str, Dict] = {}

    for cfg in configs:
        name = cfg["name"]
        desc = cfg["description"]
        overrides = cfg.get("overrides", {})

        logger.info("\n" + "=" * 60)
        logger.info("[Ablation] Running config: %s", name)
        logger.info("[Ablation] Description: %s", desc)
        if overrides:
            logger.info("[Ablation] Overrides: %s", overrides)

        with _SettingsPatch(overrides):
            results = run_evaluation(test_cases, vsm, config_name=name)
            paths = save_results(results, config_name=name)

        all_paths[name] = {k: str(v) for k, v in paths.items()}

        # Load the summary that was just saved
        summary_path = paths["summary"]
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        all_summaries.append(summary)

        logger.info("[Ablation] Config '%s' done: composite=%.4f",
                    name, summary.get("metrics", {}).get("composite", {}).get("mean", 0))

    # Print and save comparison
    _print_comparison_table(all_summaries)

    comparison = {
        "configs_run": [c["name"] for c in configs],
        "summaries": all_summaries,
        "file_paths": all_paths,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comp_path = RESULTS_DIR / "ablation_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("[Ablation] Comparison JSON saved → %s", comp_path)

    table_path = RESULTS_DIR / "ablation_comparison_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(_build_comparison_table(all_summaries))
        f.write("\n")
    logger.info("[Ablation] Comparison table saved → %s", table_path)

    return comparison


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AC-RAG Ablation Study Runner")
    parser.add_argument("--test-set", required=True, help="Path to test set JSON")
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Specific config names to run (default: all). E.g.: --configs full no_critic",
    )
    args = parser.parse_args()

    selected = ABLATION_CONFIGS
    if args.configs:
        selected = [c for c in ABLATION_CONFIGS if c["name"] in args.configs]
        if not selected:
            print(f"No matching configs found. Available: {[c['name'] for c in ABLATION_CONFIGS]}")
            sys.exit(1)

    run_ablation_study(args.test_set, configs=selected)
