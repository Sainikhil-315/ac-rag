"""
config/settings.py
Central configuration for AC-RAG pipeline.
All thresholds, flags, and hyperparameters live here.
Ablation flags allow disabling pipeline stages for research comparison.
"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root before anything else is read.
# override=False means existing shell environment variables take priority.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "index"
LOG_FILE = BASE_DIR / "logs" / "pipeline.log"

# ── Ingestion ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens (approximate via character proxy: ~4 chars/token)
CHUNK_OVERLAP = 64
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".html", ".md"]

# ── Embeddings ────────────────────────────────────────────────────────────────
# OpenAI embeddings are used by default (best quality, consistent with retrieval space).
# Swap to "sentence-transformers/all-MiniLM-L6-v2" for fully local, no-cost embeddings.
EMBEDDING_MODEL = "text-embedding-3-large"
VECTORSTORE_BACKEND = "faiss"    # "faiss" | "chroma"

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Controls which LLM is used across ALL pipeline nodes (planner, generator, critic, etc.)
# Options:
#   "openai"   → OpenAI GPT models   (requires OPENAI_API_KEY)
#   "google"   → Google Gemini models (requires GOOGLE_API_KEY)
#   "anthropic"→ Anthropic Claude     (requires ANTHROPIC_API_KEY)
LLM_PROVIDER = "openai"          # ← change this one line to switch the entire pipeline

# ── LLM Models (per provider) ─────────────────────────────────────────────────
OPENAI_LLM_MODEL    = "gpt-4o"
GOOGLE_LLM_MODEL    = "gemini-3.6-flash"   # fast, low-cost, strong reasoning
# NOTE: same external-drift issue as GROQ_LLM_MODEL above — "gemini-2.0-flash" (the
# original value here) returned a 404 model_not_found as of 2026-08-20; Google's own
# error message pointed to gemini-3.6-flash as the replacement, verified working
# (including through .with_structured_output(), which every pipeline node except
# context_refiner's optional LLM-compression step uses). One compatibility note:
# Gemini 3.6's raw (non-structured) .content is a list of parts with a "signature"
# field, not a plain string — code doing `.content.strip()` directly (currently only
# context_refiner.py's _compress_passage, only triggered above MAX_CONTEXT_CHARS, and
# evaluation/ragas_eval.py's _llm_judge) would need a small compatibility shim if
# actually deploying on Google long-term. Not hit by the cross-provider eval run.
ANTHROPIC_LLM_MODEL = "claude-sonnet-4-6"
GROQ_LLM_MODEL      = "openai/gpt-oss-120b"   # free tier, fast
# NOTE: Groq deprecates/renames hosted models over time — "llama-3.3-70b-versatile"
# (the value here as of the project's original write-up) returned a 404 model_not_found
# as of 2026-08-20; verified against Groq's live /v1/models endpoint and swapped to a
# currently-available model. Re-check https://console.groq.com/docs/models if this
# breaks again — this is an external dependency outside this project's control.

# Active model (resolved by utils/llm_factory.py — do not set manually)
LLM_MODEL = {
    "openai":    OPENAI_LLM_MODEL,
    "google":    GOOGLE_LLM_MODEL,
    "anthropic": ANTHROPIC_LLM_MODEL,
    "groq":      GROQ_LLM_MODEL,
}[LLM_PROVIDER]

LLM_TEMPERATURE = 0.0     # deterministic for faithfulness
LLM_MAX_TOKENS  = 2048

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K_MIN     = 4
RETRIEVAL_K_MAX     = 12
RETRIEVAL_K_DEFAULT = 6
MMR_FETCH_K_MULTIPLIER = 3    # fetch_k = k * multiplier for MMR diversity pass
MMR_LAMBDA_MULT        = 0.5  # 0 = max diversity, 1 = max relevance

# ── Validation ────────────────────────────────────────────────────────────────
# IMPORTANT: EVIDENCE_SCORE_THRESHOLD is on the RAW COSINE SIMILARITY scale
# (utils/scoring.py:score_passages_against_query), which is NOT the same scale as
# ROUTER_SIMILARITY_THRESHOLD below (FAISS L2-distance-based: 1/(1+distance)).
# These two previously shared one constant despite being different formulas — bug
# found via real evaluation on 2026-08-19 (see GAPS_AND_IMPROVEMENTS.md item A.6):
# for identical query/passage pairs, the FAISS-based score reads ~0.13-0.15 HIGHER
# than raw cosine (measured: 0.41 vs 0.29, 0.40 vs 0.25, 0.40 vs 0.24). Using 0.30
# on both meant the validator was silently rejecting genuinely relevant passages
# that the router would have happily accepted.
#
# Re-calibrated on the actual corpus + 15-question eval subset (raw cosine, k=8,
# n=120 passages): p25=0.22, p50=0.27, p75=0.37. Precise numeric-value lookups
# ("what chunk overlap is used") cluster around 0.20-0.28 even for the CORRECT
# passage, because the query phrasing and a bare config value in the source text
# share little lexical/semantic surface — this is a structural limit of cosine
# similarity for this query type, not something a threshold alone fully fixes.
# 0.20 is chosen to sit below that relevant-but-lexically-distant cluster while
# staying above tangential/irrelevant content (~0.02-0.15, see item A.4).
EVIDENCE_SCORE_THRESHOLD = 0.20
MIN_VALID_PASSAGES       = 2      # retry retrieval if fewer passages pass

# Summarization queries ("what is this document about?") get a relaxed bar.
# Found via a real bug report (2026-08-20): a 3KB single-chunk document (a passport
# office acknowledgement slip) scored 0.07 against "what is the document about?" —
# nowhere near 0.20 — because a broad overview query has low cosine similarity to
# any ONE specific passage almost by construction; that's not a relevance failure,
# it's a mismatch between what the query is asking for and what per-passage
# similarity measures. Compounding bug: MIN_VALID_PASSAGES=2 is structurally
# impossible to satisfy for any document (however relevant) that chunks into only
# 1 piece — which every short document does. See pipeline/nodes/validator.py.
SUMMARIZATION_SCORE_THRESHOLD  = 0.05
SUMMARIZATION_MIN_VALID_PASSAGES = 1

# ── Critic / Self-Reflection ──────────────────────────────────────────────────
CRITIC_MIN_SCORE = 4    # all dimensions must be >= 4 (scale 1–5)
MAX_RETRIES      = 3

# ── Faithfulness constraint (formal objective) ────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.80    # τ in: Faith(a, R(q,D)) ≥ τ

# ── Entry Router ─────────────────────────────────────────────────────────────
# Queries with similarity >= threshold are routed to the RAG pipeline.
# Queries below the threshold receive an "I don't know" response.
# NOTE: this is on the FAISS L2-distance-based scale (1/(1+distance)), NOT the same
# scale as EVIDENCE_SCORE_THRESHOLD (raw cosine, see comment above) — measured
# ~0.13-0.15 higher than raw cosine for identical query/passage pairs. Kept as an
# independent constant on purpose; do not re-couple these two.
ROUTER_SIMILARITY_THRESHOLD = 0.30

# ── Ablation flags ────────────────────────────────────────────────────────────
USE_RETRIEVAL_PLANNER = True
USE_VALIDATOR         = True      # re-enabled after threshold calibration check (see comment above)
USE_CONTEXT_REFINER   = True
USE_CRITIC            = True

# ── Evidence-Driven Adaptive RAG Settings ──────────────────────────────────────
# BM25 / Lexical Retrieval
BM25_ENABLED = True
BM25_K1 = 1.5
BM25_B = 0.75

# Hybrid Fusion
HYBRID_FUSION_METHOD = "rrf"      # "rrf" | "linear"
HYBRID_DENSE_WEIGHT  = 0.5
HYBRID_BM25_WEIGHT   = 0.5
RRF_K                = 60

# Evidence Requirement Planning & Coverage
REQUIREMENT_MAX_COUNT        = 6
COVERAGE_THRESHOLD           = 0.60
CRITICAL_COVERAGE_THRESHOLD  = 0.80
MIN_COVERAGE_IMPROVEMENT     = 0.10

# Claim Verification & Contradiction Detection
CLAIM_SUPPORT_THRESHOLD  = 0.70
CONTRADICTION_THRESHOLD  = 0.50

# Control & Budget Limits
MAX_RETRIEVAL_ROUNDS     = 3
MAX_GENERATION_REPAIRS   = 2
MAX_TOTAL_CONTROL_STEPS  = 8

# Abstention & Confidence
ABSTENTION_ENABLED       = True

# Confidence weights: w1 * coverage + w2 * claim_support + w3 * evidence_quality - w4 * contradiction
CONFIDENCE_W_COVERAGE               = 0.40
CONFIDENCE_W_CLAIM_SUPPORT          = 0.40
CONFIDENCE_W_EVIDENCE_QUALITY       = 0.20
CONFIDENCE_W_CONTRADICTION_PENALTY  = 0.30

# New System Ablation Flags
USE_EVIDENCE_PLANNER         = True
USE_BM25                     = True
USE_HYBRID_RETRIEVAL         = True
USE_EVIDENCE_COVERAGE        = True
USE_TARGETED_RETRIEVAL       = True
USE_CLAIM_VERIFICATION       = True
USE_CONTRADICTION_DETECTION  = True
USE_ABSTENTION               = True
USE_EVIDENCE_CONFIDENCE      = True

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"    # "DEBUG" for verbose stage traces

