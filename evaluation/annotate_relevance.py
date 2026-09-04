"""
evaluation/annotate_relevance.py
Auto-annotate the test set with relevant_chunk_ids by matching the reference
answer text against the indexed corpus.

Method (transparent and reproducible):
  1. Build/load the FAISS index exactly as the pipeline does.
  2. For each test case with a reference_answer:
       - Tokenize the reference answer, drop stopwords + tokens shorter than 4 chars.
       - For every chunk in the corpus, count what fraction of those content
         tokens appear in the chunk text.
       - Any chunk above RELEVANCE_TOKEN_OVERLAP (default 0.4 = 40%) is
         considered relevant.
  3. Save the augmented test set to test_set_sample_annotated.json.

This is deliberately a heuristic, not a human judgment. A reviewer can re-run
with --overlap 0.3 to see how the relevance threshold affects retrieval metrics
— and the script writes which overlap was used into the output file's header
comment so the eval result is reproducible.

Why not just trust the LLM?
  Using the same LLM to evaluate itself is the same circular-dependency
  RAGAS warns against. A lexical-overlap annotator is dumb but defensible:
  it's a transparent, deterministic function of the indexed text and the
  reference answer. The adaptive-retrieval claim then rests on the *delta*
  between full and no_planner at fixed ground truth — exactly what a research
  comparison needs.

Usage:
    python -m evaluation.annotate_relevance \\
        --test-set evaluation/test_set_sample.json \\
        --output   evaluation/test_set_annotated.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger
from config.settings import BASE_DIR
from vectorstore.store import VectorStoreManager

logger = setup_logger("ac_rag.annotate")

RELEVANCE_TOKEN_OVERLAP = 0.3   # chunk considered relevant if >= 30% of reference content tokens appear in it
MIN_CONTENT_TOKENS = 2          # skip reference answers with <2 content tokens (single-name answers still annotate)
MIN_TOKENS_FOR_FRACTION = 1     # if reference has fewer content tokens than MIN_CONTENT_TOKENS, fall back to substring check
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "if", "then", "than", "so",
    "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "has", "have", "had",
})


def _content_tokens(text: str) -> list[str]:
    """Lowercase alphanumerics, drop stopwords + short tokens. Returns content-bearing tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


def _is_relevant(reference: str, chunk_text: str, overlap: float) -> bool:
    """
    Two-tier relevance check:
      1. If reference answer has >= MIN_CONTENT_TOKENS content tokens, use
         token-overlap fraction (>= overlap).
      2. Otherwise (short answer like "MMR." or "GPT-4o."), use substring match
         of the literal reference text — this catches single-name/short answers
         that the token-fraction metric would treat as noise. The substring
         check is normalised (lowercased, trailing punctuation stripped) so
         "MMR." still matches a chunk containing "(MMR)".
    """
    ref_tokens = _content_tokens(reference)
    if len(ref_tokens) >= MIN_CONTENT_TOKENS:
        return _overlap_fraction(ref_tokens, chunk_text) >= overlap
    # Short-answer fallback: direct substring of the literal reference,
    # normalised so trailing/leading punctuation doesn't kill the match.
    ref_clean = reference.strip().lower().strip(".,;:!?\"'()[]{}")
    if not ref_clean:
        return False
    chunk_lower = chunk_text.lower()
    return ref_clean in chunk_lower


def _overlap_fraction(reference_tokens: list[str], chunk_text: str) -> float:
    """Fraction of reference tokens that appear in the chunk text."""
    if not reference_tokens:
        return 0.0
    chunk_lower = chunk_text.lower()
    hits = sum(1 for tok in reference_tokens if tok in chunk_lower)
    return hits / len(reference_tokens)


def annotate_test_set(
    test_set: list[dict],
    vsm: VectorStoreManager,
    overlap: float = RELEVANCE_TOKEN_OVERLAP,
) -> tuple[list[dict], dict]:
    """
    Returns (annotated_test_set, stats) where stats reports how many queries
    got at least one relevant chunk, and the average number of relevant chunks
    per query.
    """
    # Pull every chunk's text + chunk_id out of the FAISS docstore
    # (we need every chunk, not just top-k, because relevance is per-chunk ground truth)
    store = vsm._store
    docstore = store.docstore
    # FAISS docstore stores Documents in _dict keyed by uuid
    all_chunks = []
    for doc_id, doc in docstore._dict.items():
        all_chunks.append({
            "chunk_id": doc.metadata.get("chunk_id", str(doc_id)),
            "content":  doc.page_content,
        })
    logger.info("Loaded %d total chunks from docstore for annotation", len(all_chunks))

    annotated = []
    n_with_relevant = 0
    n_relevant_per_q = []

    for case in test_set:
        # Accept both field names — runner.py normalises them, but the
        # annotator may be pointed at a raw test set that still has the
        # original field name.
        ref = case.get("reference_answer") or case.get("answer") or ""
        if not ref:
            annotated.append({**case, "relevant_chunk_ids": []})
            continue

        relevant_ids = []
        for ch in all_chunks:
            if _is_relevant(ref, ch["content"], overlap):
                relevant_ids.append(ch["chunk_id"])

        n_relevant_per_q.append(len(relevant_ids))
        if relevant_ids:
            n_with_relevant += 1

        annotated.append({**case, "relevant_chunk_ids": relevant_ids})
        logger.debug(
            "id=%s | ref=%r | %d relevant chunks (overlap >= %.2f)",
            case.get("id"), ref[:60], len(relevant_ids), overlap,
        )

    stats = {
        "overlap_threshold": overlap,
        "total_queries":     len(test_set),
        "queries_with_at_least_one_relevant": n_with_relevant,
        "avg_relevant_per_query":  round(float(np.mean(n_relevant_per_q)) if n_relevant_per_q else 0.0, 2),
        "max_relevant_per_query":  int(np.max(n_relevant_per_q)) if n_relevant_per_q else 0,
        "min_relevant_per_query":  int(np.min(n_relevant_per_q)) if n_relevant_per_q else 0,
    }
    return annotated, stats


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate test set with relevant_chunk_ids")
    parser.add_argument("--test-set", required=True, help="Path to test set JSON")
    parser.add_argument("--output",    required=True, help="Where to write annotated test set")
    parser.add_argument("--overlap",   type=float, default=RELEVANCE_TOKEN_OVERLAP,
                        help=f"Token-overlap threshold for relevance (default {RELEVANCE_TOKEN_OVERLAP})")
    args = parser.parse_args()

    in_path  = Path(args.test_set)
    out_path = Path(args.output)

    with open(in_path, encoding="utf-8") as f:
        raw = json.load(f)
    # Accept both flat-list format and wrapped {"test_cases": [...]} format
    if isinstance(raw, dict) and "test_cases" in raw:
        test_set = raw["test_cases"]
    else:
        test_set = raw
    logger.info("Loaded %d test cases from %s", len(test_set), in_path.name)

    vsm = VectorStoreManager()
    vsm.load()
    logger.info("Vector store loaded.")

    annotated, stats = annotate_test_set(test_set, vsm, overlap=args.overlap)
    logger.info("Annotation stats: %s", stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "annotator":        "evaluation/annotate_relevance.py",
            "overlap_threshold": stats["overlap_threshold"],
            "annotation_stats":   stats,
            "note": "relevant_chunk_ids is a deterministic, transparent ground-truth "
                    "derived from token overlap with the reference answer — not human judgment. "
                    "Adaptive-retrieval claims rest on the *delta* between full and ablation "
                    "configs at fixed ground truth, not on the absolute metric value.",
        },
        "test_cases": annotated,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Annotated test set written to %s", out_path)

    print("\nAnnotation summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
