"""
main.py
Entry point for the AC-RAG pipeline.

Commands:
  --build             : ingest documents from data/raw/ and build the FAISS index
  --pipeline QUERY    : run full AC-RAG pipeline for a query
  --query QUERY       : raw MMR retrieval only (quick test, no LLM)
  --ingest-test       : ingestion smoke-test (no embedding)
  --evaluate          : batch evaluation over a test set
  --ablation          : full ablation study over all pipeline configurations
"""

import argparse
import sys

# Windows consoles default to cp1252, which cannot encode ✓/✗/⚠ used in CLI output below.
# Reconfigure stdout/stderr to UTF-8 so the CLI doesn't crash mid-run on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from utils.logger import setup_logger
from config.settings import DATA_RAW_DIR

logger = setup_logger("ac_rag")


def cmd_build():
    from vectorstore.build import build_index
    vsm = build_index(DATA_RAW_DIR)
    logger.info("Index ready. Run: python main.py --pipeline 'your question'")


def cmd_pipeline(question: str):
    from pipeline.graph import run_pipeline
    logger.info("=== AC-RAG Pipeline ===")
    result = run_pipeline(question)

    print("\n" + "=" * 60)
    print(f"QUERY   : {result['query']}")
    print(f"INTENT  : {result.get('intent', 'N/A')}")
    print(f"COMPLEX : {result.get('complexity_score', 'N/A')}")
    print(f"RETRIES : {result.get('retry_count', 0)}")
    print("-" * 60)
    # NOTE: `or` not a .get() default — initial_state() sets answer=None explicitly,
    # so the key always exists and the .get() default would never trigger (same class
    # of bug fixed in generator.py's context read).
    print(f"ANSWER  :\n{result.get('answer') or 'No answer generated.'}")
    print("-" * 60)

    if result.get("critic_scores"):
        scores = result["critic_scores"]
        print("CRITIC SCORES:")
        for dim in ("faithfulness", "completeness", "table_accuracy", "figure_accuracy", "conciseness"):
            print(f"  {dim:<20}: {scores.get(dim, 'N/A')}/5")
        print(f"  {'overall':<20}: {scores.get('overall', 'N/A'):.1f}/5")
        print(f"  {'feedback':<20}: {scores.get('feedback', '')[:200]}")

    att = result.get("answer_with_attribution") or []
    if att:
        print("\nEVIDENCE ATTRIBUTION:")
        for entry in att[:3]:   # show first 3 sentences
            srcs = [s.get("file", "?") + " p." + str(s.get("page", "?")) for s in entry.get("sources", [])]
            print(f"  [{', '.join(srcs) or 'uncited'}] {entry['sentence'][:120]}")

    if result.get("error"):
        print(f"\nERROR: {result['error']}")

    usage = result.get("token_usage")
    if usage:
        print(
            f"\nTOKEN USAGE: {usage['total_tokens']} total "
            f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion) "
            f"| ${usage['total_cost_usd']:.6f} | {usage['llm_calls']} LLM calls"
        )

    print("\nSTAGE LOG:")
    for entry in result.get("stage_logs", []):
        icon = "✓" if entry["status"] == "completed" else ("⚠" if entry["status"] == "skipped" else "✗")
        print(f"  {icon} {entry['stage']:<22} {entry['status']}")
    print("=" * 60)


def cmd_query(question: str):
    from vectorstore.store import VectorStoreManager
    vsm = VectorStoreManager()
    vsm.load()
    results = vsm.mmr_search(question, k=6)
    print(f"\nTop {len(results)} passages (MMR):\n")
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        print(f"[{i}] {meta.get('file_name','?')} | page={meta.get('page','?')} | "
              f"section={meta.get('section_heading','?')} | modality={meta.get('modality','?')}")
        print(f"    {doc.page_content[:300]!r}\n")


def cmd_ingest_test():
    from ingestion import ingest_documents
    logger.info("=== Ingestion smoke-test ===")
    chunks = ingest_documents(DATA_RAW_DIR)
    logger.info("Total chunks: %d", len(chunks))
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        for key in ("chunk_id", "source", "page", "section_heading", "modality", "word_count"):
            print(f"  {key:<20}: {chunk.metadata.get(key, 'N/A')}")
        print(f"  {'text[:200]':<20}: {chunk.page_content[:200]!r}")


def cmd_evaluate(test_set_path: str):
    from vectorstore.store import VectorStoreManager
    from evaluation.runner import load_test_set, run_evaluation, save_results

    vsm = VectorStoreManager()
    vsm.load()

    test_cases = load_test_set(test_set_path)
    results = run_evaluation(test_cases, vsm)
    paths = save_results(results)

    print("\nEvaluation complete. Output files:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


def cmd_ablation(test_set_path: str, configs=None):
    from evaluation.ablation import run_ablation_study, ABLATION_CONFIGS

    selected = None
    if configs:
        from evaluation.ablation import ABLATION_CONFIGS as ALL
        selected = [c for c in ALL if c["name"] in configs]

    run_ablation_study(test_set_path, configs=selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AC-RAG Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build",        action="store_true",  help="Build vector index")
    group.add_argument("--pipeline",     metavar="QUESTION",   help="Run full pipeline")
    group.add_argument("--query",        metavar="QUESTION",   help="Raw MMR retrieval only")
    group.add_argument("--ingest-test",  action="store_true",  help="Ingestion smoke-test")
    group.add_argument("--evaluate",     metavar="TEST_SET",   help="Batch evaluation (path to JSON test set)")
    group.add_argument("--ablation",     metavar="TEST_SET",   help="Full ablation study (path to JSON test set)")

    parser.add_argument(
        "--configs", nargs="+",
        help="For --ablation: specific configs to run. E.g.: --configs full no_critic"
    )

    args = parser.parse_args()

    if args.build:
        cmd_build()
    elif args.pipeline:
        cmd_pipeline(args.pipeline)
    elif args.query:
        cmd_query(args.query)
    elif args.ingest_test:
        cmd_ingest_test()
    elif args.evaluate:
        cmd_evaluate(args.evaluate)
    elif args.ablation:
        cmd_ablation(args.ablation, configs=args.configs)
