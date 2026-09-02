"""
evaluation/run_public_benchmark.py
Runs the AC-RAG pipeline against a real public benchmark subset instead of the
project's own document, to test generalization beyond a document the system was
explicitly evaluated against.

Gap fix (GAPS_AND_IMPROVEMENTS.md item B.7): everything else in this repo's
evaluation is measured on the AC-RAG paper itself — that can't demonstrate
generalization. This uses a real 25-question subset of SQuAD v2.0
(CC BY-SA 4.0, https://rajpurkar.github.io/SQuAD-explorer/), topic
"Computational_complexity_theory" — downloaded directly from the official
dataset (not reconstructed from memory, to avoid transcription errors):
  - 20 answerable factual questions with real reference answers
  - 5 UNANSWERABLE questions (SQuAD v2.0's adversarial design: plausible-sounding
    but not actually answerable from the given text) — these directly test the
    Entry Router / generator's "say I don't know rather than hallucinate" claim
    against content the system was never tuned on.

Source document: data/benchmark_raw/squad_computational_complexity.txt
Test set: evaluation/test_set_public_benchmark.json (has an "answerable": bool
field per question — score answerable/unanswerable questions separately, since
a good answer for one type looks like a failure for the other).

IMPORTANT — index safety: this script temporarily REPLACES vectorstore/index/
(the project's real demo index, built from the AC-RAG paper) with an index
built from the SQuAD benchmark document, then restores the original index
afterward automatically (even on failure, via try/finally). Do not interrupt
this script mid-run (Ctrl+C between build and restore) — if that happens,
re-run `python -m vectorstore.build` afterward to rebuild the real demo index
from data/raw/.

Usage:
    python -m evaluation.run_public_benchmark
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger
from config.settings import VECTORSTORE_DIR, BASE_DIR

logger = setup_logger("ac_rag.public_benchmark")

BENCHMARK_DOC_DIR = BASE_DIR / "data" / "benchmark_raw"
BENCHMARK_TEST_SET = BASE_DIR / "evaluation" / "test_set_public_benchmark.json"
INDEX_BACKUP_DIR = BASE_DIR / "vectorstore" / "_index_backup_pre_benchmark"


def _backup_index() -> bool:
    index_path = Path(VECTORSTORE_DIR)
    if not index_path.exists():
        logger.warning("No existing index at %s to back up — proceeding without backup.", index_path)
        return False
    if INDEX_BACKUP_DIR.exists():
        shutil.rmtree(INDEX_BACKUP_DIR)
    shutil.copytree(index_path, INDEX_BACKUP_DIR)
    logger.info("Backed up real demo index -> %s", INDEX_BACKUP_DIR)
    return True


def _restore_index(had_backup: bool) -> None:
    index_path = Path(VECTORSTORE_DIR)
    if not had_backup:
        logger.warning(
            "No backup was taken (there was no prior index) — leaving the "
            "benchmark index in place. Run `python -m vectorstore.build` to "
            "rebuild the real demo index from data/raw/ if needed."
        )
        return
    if index_path.exists():
        shutil.rmtree(index_path)
    shutil.copytree(INDEX_BACKUP_DIR, index_path)
    shutil.rmtree(INDEX_BACKUP_DIR)
    logger.info("Restored real demo index from backup -> %s", index_path)


def run() -> None:
    from vectorstore.build import build_index
    from evaluation.runner import load_test_set, run_evaluation, save_results

    had_backup = _backup_index()

    try:
        logger.info("Building benchmark index from %s ...", BENCHMARK_DOC_DIR)
        vsm = build_index(BENCHMARK_DOC_DIR)

        test_cases = load_test_set(str(BENCHMARK_TEST_SET))
        logger.info("Running evaluation on %d public benchmark questions...", len(test_cases))
        results = run_evaluation(test_cases, vsm, config_name="public_benchmark")
        paths = save_results(results, config_name="public_benchmark")

        # Separate answerable vs unanswerable scoring — a good "insufficient
        # information" answer on an unanswerable question is a WIN, not a failure.
        answerable_ids = {c["id"] for c in test_cases if c.get("answerable", True)}
        unanswerable_ids = {c["id"] for c in test_cases if not c.get("answerable", True)}

        correctly_abstained = 0
        for r in results:
            if r["id"] in unanswerable_ids:
                ans = (r.get("generated_answer") or "").lower()
                # Bug fixed 2026-08-20: the original pattern list only had "insufficient
                # information", which never matches the generator's actual phrasing
                # ("does not contain SUFFICIENT information" — no "in-" prefix). This
                # made a real run look like a 0/5 total failure when it was actually 3/5.
                if any(p in ans for p in (
                    "do not contain sufficient information", "does not contain sufficient information",
                    "insufficient information", "cannot determine", "don't know", "do not know",
                    "not contain enough information", "no information",
                )):
                    correctly_abstained += 1

        print("\n" + "=" * 60)
        print("  PUBLIC BENCHMARK RESULTS (SQuAD v2.0 subset)")
        print("=" * 60)
        print(f"  Answerable questions   : {len(answerable_ids)}")
        print(f"  Unanswerable questions : {len(unanswerable_ids)}")
        print(f"  Correctly abstained on unanswerable: {correctly_abstained}/{len(unanswerable_ids)}")
        print(f"  Full results -> {paths['json']}")
        print("=" * 60)

    finally:
        _restore_index(had_backup)


if __name__ == "__main__":
    run()
