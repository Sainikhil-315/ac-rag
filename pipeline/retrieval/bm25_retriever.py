"""
pipeline/retrieval/bm25_retriever.py
Independent BM25 / Lexical Retriever for AC-RAG.

Indexes canonical document chunks independently of FAISS vector store.
Preserves exact token matching for configuration parameters, numbers,
identifiers, abbreviations, code/API names, and section titles.
"""

import math
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from config.settings import BM25_K1, BM25_B

logger = logging.getLogger(__name__)


def default_tokenize(text: str) -> List[str]:
    """
    Standard lowercase word tokenizer preserving numbers and alphanumeric tokens.
    """
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())


class BM25Retriever:
    """
    Okapi BM25 Lexical Retriever.
    Decoupled from vector embeddings — indexes raw doc content and metadata.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0

    def index_documents(self, documents: List[Any]) -> None:
        """
        Index a list of documents or chunk dicts.
        Accepts objects with .page_content & .metadata, or dicts with content/page_content & metadata.
        """
        self.documents = []
        self.doc_tokens = []
        self.doc_lens = []
        self.doc_freqs = []
        self.idf = {}

        total_len = 0
        for doc in documents:
            if hasattr(doc, "page_content"):
                content = doc.page_content
                metadata = getattr(doc, "metadata", {})
            elif isinstance(doc, dict):
                content = doc.get("content") or doc.get("page_content", "")
                metadata = doc.get("metadata", {})
            else:
                content = str(doc)
                metadata = {}

            doc_dict = {
                "content": content,
                "metadata": metadata,
            }
            self.documents.append(doc_dict)

            # Include relevant metadata in BM25 indexed text so field/metadata queries match
            meta_parts = []
            if isinstance(metadata, dict):
                for key in ("source", "section", "section_heading", "modality", "chunk_id", "file_name", "title"):
                    val = metadata.get(key)
                    if val:
                        meta_parts.append(str(val))
                if "page" in metadata and metadata["page"] is not None:
                    meta_parts.append(f"page {metadata['page']}")

            indexed_text = f"{content} {' '.join(meta_parts)}" if meta_parts else content
            tokens = default_tokenize(indexed_text)
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)

        self.corpus_size = len(self.documents)
        if self.corpus_size > 0:
            self.avgdl = total_len / self.corpus_size
            self._calc_idf()

        logger.info("[BM25] Indexed %d documents (avg len: %.1f tokens)", self.corpus_size, self.avgdl)

    def _calc_idf(self) -> None:
        """
        Calculate Inverse Document Frequency (IDF) for all terms in corpus using standard BM25 formula.
        """
        df: Dict[str, int] = {}
        for freqs in self.doc_freqs:
            for term in freqs.keys():
                df[term] = df.get(term, 0) + 1

        for term, freq in df.items():
            # Standard Okapi BM25 IDF formula with smoothing
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(0.0, idf_val)

    def search(
        self,
        query: str,
        top_k: int = 6,
        modality_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents matching query based on BM25 score.

        Args:
            query: user search string
            top_k: number of results to return
            modality_filter: "text" | "table" | "figure" | "all"

        Returns:
            List of doc dicts with added 'score' field normalized between 0 and 1.
        """
        if self.corpus_size == 0 or not query.strip():
            return []

        query_tokens = default_tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []

        for idx, (tokens, freqs, doc_len) in enumerate(zip(self.doc_tokens, self.doc_freqs, self.doc_lens)):
            doc = self.documents[idx]
            meta = doc.get("metadata", {})

            # Filter by modality if requested
            if modality_filter and modality_filter != "all":
                doc_modality = meta.get("modality", "text")
                if doc_modality != modality_filter:
                    continue

            score = 0.0
            len_norm = 1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0))
            for q_term in query_tokens:
                if q_term not in freqs:
                    continue
                tf = freqs[q_term]
                idf_val = self.idf.get(q_term, 0.0)
                term_score = idf_val * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
                score += term_score

            if score > 0.0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        max_score = scores[0][1] if scores else 1.0

        for idx, raw_score in scores[:top_k]:
            doc = self.documents[idx]
            # Normalize score relative to max score or map smoothly
            norm_score = raw_score / (max_score if max_score > 0 else 1.0)
            result_doc = {
                "content": doc["content"],
                "metadata": dict(doc["metadata"]),
                "score": float(norm_score),
                "bm25_raw_score": float(raw_score),
                "retrieval_method": "bm25",
            }
            results.append(result_doc)

        return results
