"""
utils/attribution.py
Evidence attribution utilities.

Parses the [SOURCE N] tags embedded in the context and maps
inline citations in the generated answer back to chunk metadata.

Context format (built by context_refiner.py):
  [SOURCE 1 | file=paper.pdf | page=3 | section=Introduction | chunk=abc123 | score=0.91 | modality=text]
  <passage text>

Answer format (expected from generator):
  Sentence text [1][3]. Another sentence [2].

This module:
  1. Extracts the source registry from the context string
  2. Parses inline citations from each answer sentence
  3. Builds the answer_with_attribution structure consumed by the research log
"""

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Matches: [SOURCE 1 | file=x | page=y | section=z | chunk=abc | score=0.91 | modality=text]
_SOURCE_TAG_RE = re.compile(
    r"\[SOURCE\s+(\d+)"           # group 1: source number
    r"[^\]]*file=([^\|]+)"        # group 2: file name
    r"[^\]]*page=([^\|]+)"        # group 3: page
    r"[^\]]*section=([^\|]+)"     # group 4: section
    r"[^\]]*chunk=([^\|]+)"       # group 5: chunk_id
    r"[^\]]*score=([^\|]+)"       # group 6: score
    r"[^\]]*modality=([^\]]+)\]", # group 7: modality
    re.IGNORECASE,
)

# Matches inline citations like [1], [2], [1][3], [1, 3]
_INLINE_CITE_RE = re.compile(r"\[(\d+(?:[,\s]+\d+)*)\]|\[(\d+)\]")


def build_source_registry(context: str) -> Dict[int, Dict[str, str]]:
    """
    Extract a mapping of source_number → metadata from the context string.

    Returns:
        {1: {"file": "paper.pdf", "page": "3", "section": "...", "chunk_id": "abc", ...}, ...}
    """
    registry: Dict[int, Dict[str, str]] = {}
    for match in _SOURCE_TAG_RE.finditer(context):
        num = int(match.group(1))
        registry[num] = {
            "source_num": num,
            "file": match.group(2).strip(),
            "page": match.group(3).strip(),
            "section": match.group(4).strip(),
            "chunk_id": match.group(5).strip(),
            "score": match.group(6).strip(),
            "modality": match.group(7).strip(),
        }
    return registry


def _parse_citation_numbers(cite_str: str) -> List[int]:
    """Parse '1', '1, 3', '1 3' → [1, 3]."""
    return [int(n) for n in re.findall(r"\d+", cite_str)]


def build_attribution(
    answer: str,
    context: str,
) -> List[Dict[str, Any]]:
    """
    Build per-sentence attribution by parsing inline citations in the answer.

    Args:
        answer  : generated answer text, may contain inline citations like [1][2]
        context : the refined context string with [SOURCE N | ...] tags

    Returns:
        List of dicts, one per sentence:
        [
          {
            "sentence": "The model achieved 94% accuracy [1].",
            "citations": [1],
            "sources": [{"file": "paper.pdf", "page": "3", "chunk_id": "abc", ...}]
          },
          ...
        ]
    """
    registry = build_source_registry(context)
    if not registry:
        logger.warning("[Attribution] No SOURCE tags found in context.")

    # Split answer into sentences (simple rule-based split)
    sentences = _split_sentences(answer)

    attribution = []
    for sentence in sentences:
        cite_nums: List[int] = []
        for match in _INLINE_CITE_RE.finditer(sentence):
            raw = match.group(1) or match.group(2)
            cite_nums.extend(_parse_citation_numbers(raw))

        cite_nums = sorted(set(cite_nums))
        sources = [registry[n] for n in cite_nums if n in registry]

        attribution.append({
            "sentence": sentence.strip(),
            "citations": cite_nums,
            "sources": sources,
        })

    covered = sum(1 for a in attribution if a["citations"])
    logger.debug(
        "[Attribution] %d/%d sentences have citations",
        covered, len(attribution)
    )
    return attribution


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    Uses a simple regex that handles common abbreviations and citation brackets.
    """
    # Don't split on period inside citations like [1.2] or abbreviations
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    # Fallback: if single block, return as one sentence
    return parts if parts else [text]


def clean_citations_from_answer(answer: str) -> str:
    """
    Remove inline citation markers from the answer for clean display.
    [1][2] and [1, 2] patterns are stripped.
    """
    cleaned = re.sub(r"\[\d+(?:[,\s]+\d+)*\]", "", answer)
    return re.sub(r"\s{2,}", " ", cleaned).strip()
