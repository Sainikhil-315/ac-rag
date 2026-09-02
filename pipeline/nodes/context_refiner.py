"""
pipeline/nodes/context_refiner.py
Context Refiner Node — CAG (Context-Aware Generation) stage (full implementation).

Algorithm:
  1. Re-rank passages by their validation score (highest first)
  2. Semantic deduplication: remove passages with pairwise cosine similarity
     above DEDUP_THRESHOLD (0.85) — keep the higher-scored one
  3. Context compression: if total context exceeds MAX_CONTEXT_CHARS,
     use LLM to compress verbose passages while preserving key facts
  4. Build structured context with [SOURCE N] attribution tags
     — each tag carries: file name, page, section, chunk_id, score
     — used by Generator for evidence attribution

Design:
  Deduplication happens BEFORE compression so we don't waste LLM tokens
  compressing passages we'll discard anyway.

  The LLM compression step is optional (COMPRESS_CONTEXT flag in settings).
  For ablation studies you can disable it without changing this file.

Ablation:
  USE_CONTEXT_REFINER=False in settings → stub pass-through (handled in graph router)
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from langchain_core.prompts import ChatPromptTemplate

from config.settings import USE_CONTEXT_REFINER
from utils.llm_factory import get_llm
from pipeline.state import ACRagState
from utils.scoring import pairwise_cosine_matrix

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEDUP_THRESHOLD = 0.85       # passages with similarity > this are considered redundant
MAX_CONTEXT_CHARS = 12_000   # ~3000 tokens; trigger compression above this
COMPRESS_CONTEXT = True      # set False to skip LLM compression (faster, less clean)


# ── Step 1: Re-rank by score ──────────────────────────────────────────────────

def _rerank(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort passages by validation score descending."""
    return sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)


# ── Step 2: Semantic deduplication ────────────────────────────────────────────

def _deduplicate(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove semantically redundant passages.
    Greedily keep each passage unless it is too similar to an already-kept one.
    The higher-scored passage (first after re-rank) is always kept.
    """
    if len(docs) <= 1:
        return docs

    texts = [doc["content"] for doc in docs]

    try:
        sim_matrix = pairwise_cosine_matrix(texts)
    except Exception as e:
        logger.warning("[ContextRefiner] Dedup similarity computation failed: %s — skipping dedup", e)
        return docs

    kept_indices = []
    for i in range(len(docs)):
        redundant = False
        for j in kept_indices:
            if sim_matrix[i, j] > DEDUP_THRESHOLD:
                logger.debug(
                    "[ContextRefiner] Dropping chunk %s (sim=%.3f with chunk %s)",
                    docs[i].get("chunk_id", i), sim_matrix[i, j], docs[j].get("chunk_id", j)
                )
                redundant = True
                break
        if not redundant:
            kept_indices.append(i)

    kept = [docs[i] for i in kept_indices]
    removed = len(docs) - len(kept)
    if removed:
        logger.info("[ContextRefiner] Dedup removed %d redundant passage(s)", removed)

    return kept


# ── Step 3: LLM compression ───────────────────────────────────────────────────

_COMPRESS_SYSTEM = """You are a context compression assistant for a document QA system.
You will receive a passage from a document and a question.
Your task: compress the passage to retain ONLY information relevant to the question.
Rules:
  - Keep all facts, numbers, names, and definitions relevant to the question
  - Remove sentences with no bearing on the question
  - Do NOT add any information not present in the original passage
  - Do NOT answer the question — only compress the source material
  - Return only the compressed passage text, no preamble"""

_COMPRESS_HUMAN = "Question: {query}\n\nPassage:\n{passage}"

_compress_prompt = ChatPromptTemplate.from_messages([
    ("system", _COMPRESS_SYSTEM),
    ("human", _COMPRESS_HUMAN),
])


def _compress_passage(query: str, passage: str) -> str:
    """Use LLM to compress a single passage to question-relevant content."""
    try:
        chain = _compress_prompt | get_llm()
        result = chain.invoke({"query": query, "passage": passage})
        compressed = result.content.strip()
        ratio = len(compressed) / max(len(passage), 1)
        logger.debug("[ContextRefiner] Compressed passage: %.0f%% of original", ratio * 100)
        return compressed
    except Exception as e:
        logger.warning("[ContextRefiner] Compression failed for passage: %s — using original", e)
        return passage


def _compress_if_needed(
    query: str, docs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Compress individual passages if total context exceeds MAX_CONTEXT_CHARS.
    Only compresses passages above a per-passage length threshold.
    """
    total_chars = sum(len(d["content"]) for d in docs)
    if not COMPRESS_CONTEXT or total_chars <= MAX_CONTEXT_CHARS:
        return docs

    logger.info(
        "[ContextRefiner] Context too large (%d chars > %d). Compressing...",
        total_chars, MAX_CONTEXT_CHARS
    )

    # Compress passages longer than average (these are the bloat sources)
    avg_len = total_chars / len(docs)
    compressed_docs = []
    for doc in docs:
        if len(doc["content"]) > avg_len * 1.2:
            compressed_content = _compress_passage(query, doc["content"])
            compressed_docs.append({**doc, "content": compressed_content, "compressed": True})
        else:
            compressed_docs.append({**doc, "compressed": False})

    new_total = sum(len(d["content"]) for d in compressed_docs)
    logger.info(
        "[ContextRefiner] After compression: %d chars (%.0f%% of original)",
        new_total, 100 * new_total / max(total_chars, 1)
    )
    return compressed_docs


# ── Step 4: Build structured context string ───────────────────────────────────

def _build_context_string(docs: List[Dict[str, Any]]) -> str:
    """
    Assemble the final context string with structured [SOURCE N] attribution tags.
    Each tag carries: file, page, section, chunk_id, score.
    This format allows the Generator to map answer sentences back to sources.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        tag_parts = [
            f"SOURCE {i}",
            f"file={meta.get('file_name', doc.get('source', '?'))}",
            f"page={meta.get('page', '?')}",
            f"section={doc.get('section_heading', 'Unknown')}",
            f"chunk={doc.get('chunk_id', '?')}",
            f"score={doc.get('score', 0):.3f}",
            f"modality={doc.get('modality', 'text')}",
        ]
        tag = "[" + " | ".join(tag_parts) + "]"
        parts.append(f"{tag}\n{doc['content']}")

    return "\n\n---\n\n".join(parts)


# ── Node function ─────────────────────────────────────────────────────────────

def context_refiner_node(state: ACRagState) -> ACRagState:
    """
    LangGraph node: Context Refiner (full implementation).
    Reads:  state["query"], state["scored_docs"]
    Writes: state["refined_context"]
    """
    query = state["query"]
    docs = state.get("scored_docs") or []

    log_entry: Dict[str, Any] = {
        "stage": "context_refiner",
        "status": "started",
        "details": {"docs_in": len(docs)},
    }

    if not docs:
        logger.warning("[ContextRefiner] No scored docs to refine.")
        log_entry["status"] = "completed"
        log_entry["details"]["context_chars"] = 0
        return {
            **state,
            "refined_context": "",
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    # ── Ablation bypass ───────────────────────────────────────────────────────
    # Bug fixed 2026-08-20: this node used to be skipped ENTIRELY via graph routing
    # when USE_CONTEXT_REFINER=False, which left refined_context at its initial
    # value of None — the generator then received no context at all (not just
    # "unrefined" context), and separately crashed on len(None). A fair "naive RAG"
    # baseline still needs to pass raw retrieved passages to the generator; it just
    # skips re-ranking/dedup/compression. Handling the bypass inside the node (like
    # validator.py and critic.py already do for their own ablation flags) keeps this
    # consistent and means the node is always reachable in the graph.
    if not USE_CONTEXT_REFINER:
        logger.info("[ContextRefiner] Ablation: refiner disabled — passing %d raw docs through unranked/unfiltered.", len(docs))
        context = _build_context_string(docs)
        log_entry["status"] = "completed"
        log_entry["details"].update({"ablation_skip": True, "context_chars": len(context)})
        return {
            **state,
            "refined_context": context,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    try:
        # Step 1: Re-rank
        ranked = _rerank(docs)
        logger.info("[ContextRefiner] Re-ranked %d passages by score", len(ranked))

        # Step 2: Semantic dedup
        deduped = _deduplicate(ranked)
        logger.info("[ContextRefiner] After dedup: %d passages", len(deduped))

        # Step 3: Compress if needed
        final_docs = _compress_if_needed(query, deduped)

        # Step 4: Build structured context
        context = _build_context_string(final_docs)

        logger.info(
            "[ContextRefiner] Final context: %d passages, %d chars",
            len(final_docs), len(context)
        )

        log_entry["status"] = "completed"
        log_entry["details"].update({
            "docs_after_dedup": len(deduped),
            "docs_final": len(final_docs),
            "context_chars": len(context),
            "compressed": any(d.get("compressed", False) for d in final_docs),
        })

        return {
            **state,
            "refined_context": context,
            "stage_logs": state["stage_logs"] + [log_entry],
        }

    except Exception as e:
        logger.error("[ContextRefiner] Failed: %s", e)
        log_entry["status"] = "failed"
        log_entry["details"]["error"] = str(e)

        # Fallback: use raw concatenation so the pipeline doesn't die
        fallback = "\n\n".join(d["content"] for d in docs)
        return {
            **state,
            "refined_context": fallback,
            "error": f"ContextRefiner failed: {e}",
            "stage_logs": state["stage_logs"] + [log_entry],
        }
