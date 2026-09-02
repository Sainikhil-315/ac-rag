"""
evaluation/run_cross_provider.py
Runs the same eval subset across multiple LLM providers, to compare
faithfulness/cost/latency — the architecture already supports 4 providers via
utils/llm_factory.py, but this comparison had never been run (see
GAPS_AND_IMPROVEMENTS.md item B.10).

OpenAI, Groq, and Google are compared (ANTHROPIC_API_KEY in .env is still a
placeholder value, so that provider isn't runnable here — documented rather
than silently only testing three). GOOGLE_API_KEY was added 2026-08-20.

IMPORTANT bug this script works around (same root cause as the ablation.py fix
in the "Fix ablation study never actually applying its config overrides"
commit): utils/llm_factory.py does `from config.settings import LLM_PROVIDER,
LLM_MODEL, ...` — a bound copy at import time. Patching config.settings alone
does nothing; llm_factory.py's own bound names must be patched directly.
Additionally, get_llm() is @lru_cache'd on its arguments — since pipeline
nodes call get_llm() with NO arguments, switching the underlying LLM_PROVIDER
between runs requires clearing that cache, or every "provider switch" would
silently keep returning the first provider's cached client.

Usage:
    python -m evaluation.run_cross_provider
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger
from config.settings import BASE_DIR, OPENAI_LLM_MODEL, GROQ_LLM_MODEL, GOOGLE_LLM_MODEL

logger = setup_logger("ac_rag.cross_provider")

TEST_SET = BASE_DIR / "evaluation" / "test_set_ablation_subset.json"

# Only providers with real API keys in this environment (see module docstring).
PROVIDERS = [
    {"name": "openai", "model": OPENAI_LLM_MODEL},
    {"name": "groq", "model": GROQ_LLM_MODEL},
    {"name": "google", "model": GOOGLE_LLM_MODEL},
]


def run() -> None:
    import utils.llm_factory as llm_factory
    from evaluation.runner import load_test_set, run_evaluation, save_results
    from vectorstore.store import VectorStoreManager

    original_provider = llm_factory.LLM_PROVIDER
    original_model = llm_factory.LLM_MODEL

    test_cases = load_test_set(str(TEST_SET))
    vsm = VectorStoreManager()
    vsm.load()

    summaries = []
    try:
        for p in PROVIDERS:
            logger.info("=" * 60)
            logger.info("[CrossProvider] Running provider=%s model=%s", p["name"], p["model"])

            llm_factory.LLM_PROVIDER = p["name"]
            llm_factory.LLM_MODEL = p["model"]
            llm_factory.get_llm.cache_clear()  # see module docstring — required, not optional

            results = run_evaluation(test_cases, vsm, config_name=f"provider_{p['name']}")
            paths = save_results(results, config_name=f"provider_{p['name']}")

            import json
            with open(paths["summary"], encoding="utf-8") as f:
                summary = json.load(f)
            summaries.append(summary)
    finally:
        llm_factory.LLM_PROVIDER = original_provider
        llm_factory.LLM_MODEL = original_model
        llm_factory.get_llm.cache_clear()

    print("\n" + "=" * 70)
    print("  CROSS-PROVIDER COMPARISON")
    print("=" * 70)
    print(f"{'Provider':<12}{'Faithfulness':>14}{'Composite':>12}{'Avg Lat (s)':>14}{'Errors':>10}")
    for s in summaries:
        faith = s.get("metrics", {}).get("faithfulness", {}).get("mean", "-")
        comp = s.get("metrics", {}).get("composite", {}).get("mean", "-")
        print(f"{s['config'].replace('provider_', ''):<12}{faith:>14}{comp:>12}{s.get('avg_latency_s', '-'):>14}{s.get('errors', '-'):>10}")
    print("=" * 70)
    print("\nNote: Groq/Google's cost tables aren't in langchain_community's OpenAI-only")
    print("pricing callback (see pipeline/graph.py's run_pipeline token_usage docstring) —")
    print("their token_usage will be None. Groq is free-tier anyway; for Google, use the")
    print("published per-token pricing for the model in GOOGLE_LLM_MODEL if cost matters.")


if __name__ == "__main__":
    run()
