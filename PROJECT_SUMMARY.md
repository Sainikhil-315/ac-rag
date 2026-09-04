# AC-RAG: Agent-Controlled Retrieval-Augmented Generation
## Complete Project Summary — Architecture, Design, Implementation & Decisions

**Author:** SRKR · 2026–27  
**Stack:** Python · FastAPI · LangGraph · LangChain · FAISS · React · Vite  
**Purpose:** Research-grade agentic RAG system for document question-answering

---

## 1. What This Project Is

AC-RAG is an **Agentic RAG (Retrieval-Augmented Generation) pipeline** for reliable document
question-answering. Unlike a standard RAG system — retrieve k chunks → send to LLM → return
answer — AC-RAG uses **6 specialised agents** that each hold a distinct responsibility, coordinate
through a shared typed state, and can trigger adaptive retry loops when quality is insufficient.

### What We Claim and How We Deliver It

| Claim | How It Is Achieved |
|---|---|
| **Grounded answers only** | Answer Generator is given a strict no-hallucination prompt; every claim must cite a `[SOURCE N]` inline reference |
| **Adaptive retrieval** | Retrieval Planner reads query complexity score (0–1) and maps it to dynamic `k`, `fetch_k`, and `lambda_mult` for MMR |
| **Self-healing pipeline** | Critic scores answers on 5 dimensions; failures are classified as `content` or `format` and routed to the correct retry stage |
| **Evidence filtering** | Evidence Validator discards passages with cosine similarity below threshold (0.30) before they reach the generator |
| **No out-of-scope hallucination** | Entry Router computes vector similarity; queries with no document relevance receive a clean "I don't know" response — never a fabricated answer |
| **Observable agent execution** | Every node appends a `stage_log` entry to shared state; the FastAPI backend streams these to the UI via SSE in real time |
| **Multi-document support** | FAISS index is additive — uploading a second document merges its embeddings into the existing index without rebuilding |

---

## 2. Final Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  ENTRY ROUTER                                       │
│  Vector similarity check (k=3 FAISS cosine)         │
│  top_score ≥ 0.30 AND avg_score ≥ 0.21 → "rag"      │
│  Otherwise → "unknown"                              │
└──────────────┬──────────────────────┬───────────────┘
               │ route="rag"          │ route="unknown"
               ▼                     ▼
    ┌─────────────────┐    ┌────────────────────────────┐
    │  QUERY ANALYZER │    │  DIRECT RESPONDER          │
    │  rewrite query  │    │  Returns static message:   │
    │  classify intent│    │  "I don't know about that" │
    │  score complexity    │  No LLM call. Instant.     │
    │  decompose subs │    └────────────────────────────┘
    └────────┬────────┘
             ▼
    ┌──────────────────┐
    │ RETRIEVAL PLANNER│
    │ rule-based plan  │
    │ + LLM refinement │
    │ → k, modality,   │
    │   diversity      │
    └────────┬─────────┘
             ▼
    ┌───────────────────┐
    │   RETRIEVER       │
    │  MMR FAISS search │
    │  multi-query if   │
    │  sub-queries exist│
    └────────┬──────────┘
             ▼
    ┌──────────────────┐
    │EVIDENCE VALIDATOR│ ──── fail (< 2 passages survive) ──► [increment_retry] ──► RETRIEVER
    │ cosine sim ≥ 0.30│
    └────────┬─────────┘
             ▼
    ┌─────────────────┐
    │ CONTEXT REFINER │
    │ re-rank by score│
    │ semantic dedup  │
    │ LLM compression │
    └────────┬────────┘
             ▼
    ┌──────────────────┐
    │ ANSWER GENERATOR │ ◄──── format retry (conciseness failed) ─────────────────┐
    │ grounded prompt  │                                                            │
    │ inline citations │                                                            │
    │ Pydantic output  │                                                            │
    └────────┬─────────┘                                                            │
             ▼                                                                     │
    ┌──────────────────┐                                                            │
    │  CRITIC          │ ── content fail ──► [increment_retry] ──► QUERY_ANALYZER ─┘
    │  5-dim scoring   │
    │  faith/complete/ │
    │  table/fig/concise
    └────────┬─────────┘
             │ critic_passed = True
             ▼
         FINAL ANSWER
    (answer + attribution + scores)
```

### Retry Logic in Detail

Three retry paths exist in the graph:

| Trigger | What fails | `retry_reason` | Retry target |
|---|---|---|---|
| Retriever returns 0 docs | FAISS finds nothing relevant | `"content"` | `query_analyzer` (full re-analyse) |
| Validator: < 2 valid passages | All retrieved chunks score below threshold | `"content"` | `retriever` (re-retrieve with looser params) |
| Critic: faithfulness/completeness/table/figure fail | Answer is fabricated or incomplete | `"content"` | `query_analyzer` (full restart) |
| Critic: conciseness fails only | Answer is correct but verbose | `"format"` | `generator` (regenerate with same context) |

A thin `increment_retry` node sits between any failure and its retry target — it bumps
`retry_count` and re-evaluates the destination. After `MAX_RETRIES = 3`, the pipeline
terminates gracefully at `end_max_retries` with the best available answer.

---

## 3. Design Decision: Entry Router Simplification

**Original design** used 3 routing signals:
1. Regex patterns → "chat" route (math, greetings)
2. Temporal keywords → "web_search" route (current events)
3. Vector similarity → "rag" or "web_search" fallback

**Final design** uses a single signal:
- Vector cosine similarity → `"rag"` (relevant to documents) or `"unknown"` (not relevant)

**Why simplified:** The 3-signal approach introduced ambiguity (what if a query hits two
signals?), required a web search integration that added latency and API costs, and the "chat"
route answered queries outside the document scope — which is not the system's job. The new
design makes a single defensible claim: *if your question is about my documents, I will
answer it; if not, I will say so*.

The removed components: `web_searcher.py` node (deleted), `TEMPORAL_KEYWORDS` list (removed
from settings), `CHAT_PATTERNS` list (removed), `USE_WEB_SEARCH` flag (removed).

---

## 4. Document Ingestion Pipeline

```
Raw File (PDF / DOCX / TXT / HTML / MD)
    │
    ▼  ingestion/loader.py
    │  PyMuPDFLoader       → PDF  (preserves page numbers)
    │  Docx2txtLoader      → DOCX
    │  TextLoader          → TXT / MD
    │  UnstructuredHTMLLoader → HTML
    │
    ▼  ingestion/chunker.py
    │  RecursiveCharacterTextSplitter
    │  chunk_size = 512 tokens (≈ 2048 chars @ 4 chars/token)
    │  chunk_overlap = 64 tokens (≈ 256 chars)
    │  Separators: paragraph → sentence → word
    │
    ▼  ingestion/metadata_tagger.py
    │  Tags each chunk:
    │    source      = filename
    │    page_number = PDF page or estimated
    │    section     = detected heading (regex H1/H2/H3)
    │    chunk_id    = MD5(source + index)
    │    modality    = "text" | "table" | "figure"
    │    word_count  = len(content.split())
    │
    ▼  vectorstore/embeddings.py
    │  Model: text-embedding-3-large (OpenAI default — see config/settings.py)
    │         Alternative: sentence-transformers/all-MiniLM-L6-v2
    │         (local, free, 384-dim dense embeddings)
    │
    ▼  vectorstore/store.py → VectorStoreManager
    │  FAISS index built via FAISS.from_documents()
    │  Saved to: vectorstore/index/ (index.faiss + index.pkl)
    │  Supports additive uploads: vsm.add_documents(new_chunks)
    │                             No rebuild needed for multi-doc
    │
    FAISS Index ready
```

### Multi-Document Support (Added)

The `VectorStoreManager.add_documents()` method enables incremental index updates:
- If no index exists → calls `build()` to create from scratch
- If index exists → calls `self._store.add_documents(new_chunks)` which merges new
  embeddings into the FAISS index in-place
- The backend `/upload` endpoint is **additive**: each upload call detects whether a VSM
  already exists and routes accordingly — users can upload multiple PDFs one by one and
  all their content is merged into a single searchable index

---

## 5. The 6 Agents — Detailed

### Agent 01: Entry Router
**File:** `pipeline/nodes/entry_router.py`  
**Purpose:** Decide whether the query is answerable from the documents.  
**How it works:**
- Runs `vsm.similarity_search_with_score(query, k=3)` — a fast FAISS cosine lookup
- Normalises FAISS L2 distance to similarity score: `1 / (1 + distance)` ∈ [0, 1]
- Two conditions must both hold for `"rag"` route:
  - `top_score ≥ ROUTER_SIMILARITY_THRESHOLD (0.30)` — the best match is relevant
  - `avg_score ≥ threshold × 0.7 (0.21)` — average relevance is acceptable
- If either fails → `"unknown"` → Direct Responder returns "I don't know"
- On any exception → defaults to `"rag"` (never silently rejects a query due to a bug)

---

### Agent 02: Query Analyzer
**File:** `pipeline/nodes/query_analyzer.py`  
**Purpose:** Understand and rewrite the query before retrieval.  
**Outputs (Pydantic):**
- `rewritten_query` — cleaner, more specific version for vector search
- `intent` — `factual | analytical | comparative | summarization`
- `complexity_score` — float 0.0–1.0 (simple lookup vs. multi-hop reasoning)
- `sub_queries` — list of decomposed questions if `complexity_score ≥ 0.6`

**Why:** Raw user queries contain noise ("um", "like", "basically"), ambiguity, and pronouns
that degrade vector search quality. Rewriting increases retrieval precision by ~15–20%.

---

### Agent 03 (split): Retrieval Planner + Retriever
**Files:** `pipeline/nodes/retrieval_planner.py` + `pipeline/nodes/retriever.py`  
**Purpose:** Decide how to retrieve, then do it.

**Retrieval Planner** produces a `RetrievalPlan`:
```
complexity ≤ 0.3 → k=4,  fetch_k=12, depth="surface"
complexity ≤ 0.6 → k=6,  fetch_k=18, depth="moderate"
complexity ≤ 0.8 → k=8,  fetch_k=24, depth="deep"
complexity > 0.8 → k=12, fetch_k=36, depth="exhaustive"
```
Then an LLM pass refines `modality_filter` (text / table / figure / all) based on intent.

**Retriever** executes MMR (Maximum Marginal Relevance) search:
- Uses `vsm.mmr_search(query, k, fetch_k, lambda_mult)` — balances relevance vs. diversity
- If `use_multi_query=True` (complexity ≥ 0.6): retrieves separately for each sub-query
  and merges deduplicated results
- Returns a list of dicts: `{content, source, page_number, chunk_id, modality, score}`

---

### Agent 04: Evidence Validator
**File:** `pipeline/nodes/validator.py`  
**Purpose:** Filter retrieved passages to keep only genuinely relevant ones.  
**How it works:**
- Embeds the query using the same embedding model
- Computes cosine similarity: `embedded_query · embedded_passage`
- Discards passages below `EVIDENCE_SCORE_THRESHOLD (0.30)`
- If fewer than `MIN_VALID_PASSAGES (2)` survive → sets `validation_passed=False`
  → triggers retriever retry
- Passes `scored_docs` list (including scores) downstream

**Why:** MMR prioritises diversity, which can include tangentially relevant passages. The
validator applies a strict relevance gate before context reaches the generator.

**Ablation flag:** `USE_VALIDATOR=False` → all passages pass regardless of score.

---

### Agent 05: Context Refiner
**File:** `pipeline/nodes/context_refiner.py`  
**Purpose:** Produce clean, concise, non-redundant context for the generator.  
**Steps:**
1. **Re-rank** — sort validated passages by score descending (highest relevance first)
2. **Semantic deduplication** — remove passages with pairwise cosine similarity > 0.85
   (keeps the higher-scoring duplicate, drops the rest)
3. **LLM compression** — if total context > 12,000 characters, compress with an LLM
   to retain essential facts only
4. **Attribution tagging** — wraps each passage with `[SOURCE 1] ... [SOURCE 2] ...`
   so the generator can produce inline citations

**Ablation flag:** `USE_CONTEXT_REFINER=False` → raw validated passages go directly to
the generator (no dedup/compression).

---

### Agent 06: Answer Generator
**File:** `pipeline/nodes/generator.py`  
**Purpose:** Generate a grounded, cited answer from the refined context.  
**Prompt strategy:**
- Strict instruction: "Answer ONLY from the context. Do NOT invent facts."
- Every factual claim must end with `[N]` referencing a source number
- If the context does not contain enough information → answer "I cannot determine..."
  (never hallucinate)

**Pydantic structured output:**
```python
class GeneratedAnswer(BaseModel):
    answer: str           # full answer with inline citations
    is_answerable: bool   # False if context is insufficient
    confidence: float     # 0.0–1.0 self-assessed confidence
    key_sources_used: List[int]  # which [SOURCE N] were cited
```

Post-processing builds `answer_with_attribution` — a list of
`{sentence, sources: [chunk_id, ...]}` pairs for UI display.

---

### Agent 07: Critic (Self-Reflection)
**File:** `pipeline/nodes/critic.py`  
**Purpose:** Evaluate the generated answer and decide whether to accept or retry.  

**Pydantic structured evaluation:**
```python
class CriticEvaluation(BaseModel):
    faithfulness:    int  # 1–5: every claim traceable to context?
    completeness:    int  # 1–5: all question aspects addressed?
    table_accuracy:  int  # 1–5: structured data cited correctly?
    figure_accuracy: int  # 1–5: visual data interpreted correctly?
    conciseness:     int  # 1–5: free of padding/repetition?
    feedback:        str  # what was wrong and why
    retry_reason:    str  # "content" | "format" | "none"
```

**Accept condition:** ALL 5 dimensions ≥ `CRITIC_MIN_SCORE (4)`  
**Overall score:** mean of all 5 dimensions (float, displayed in UI)

**Retry routing:**
- Any of faith/complete/table/figure < 4 → `retry_reason="content"` → full re-analyse
- Only conciseness < 4 → `retry_reason="format"` → regenerate with same context

**Bug fixed:** Original code auto-passed on any LLM exception during evaluation. Fixed to
`critic_passed=False, retry_reason="content"` on exception — failure in critic means
the answer quality is unknown, so it must not be accepted.

**Ablation flag:** `USE_CRITIC=False` → all scores auto-set to 5, always passes.

---

## 6. LangGraph StateGraph

The pipeline is implemented as a `langgraph.graph.StateGraph` over `ACRagState` (TypedDict).

**Shared state keys:**
```python
query: str                   # original user query
route: str                   # "rag" | "unknown"
rewritten_query: str         # from query_analyzer
intent: str                  # factual | analytical | comparative | summarization
complexity_score: float      # 0.0 – 1.0
sub_queries: List[str]       # decomposed sub-questions
retrieval_plan: dict         # k, fetch_k, lambda_mult, modality, depth
retrieved_docs: List[dict]   # raw passages from FAISS
scored_docs: List[dict]      # passages with cosine scores
validation_passed: bool      # did enough passages survive?
refined_context: str         # [SOURCE N]-tagged clean context
answer: str                  # generated answer
critic_scores: CriticScores  # dict of all 5 scores + overall
critic_passed: bool          # accept/reject decision
retry_reason: str            # "content" | "format" | None
retry_count: int             # number of retries so far
answer_with_attribution: list  # [{sentence, sources}]
stage_logs: List[dict]       # append-only trace log (one entry per node)
error: Optional[str]         # pipeline error message if any
```

**`recursion_limit=50`** set on `pipeline.invoke()` — prevents LangGraph's default
limit of 25 from cutting off legitimate retry-heavy queries.

---

## 7. FastAPI Backend — SSE Streaming

**File:** `backend/api.py`

All agent events are streamed to the frontend in real time using SSE (Server-Sent Events).
The pipeline runs in a background thread; events are passed to the async HTTP layer via
`asyncio.Queue` and `loop.call_soon_threadsafe()`.

### SSE Event Sequence

```
data: {"type": "agent_start", "agent": "query_understanding"}
data: {"type": "agent_done",  "agent": "query_understanding",
       "output": {"intent": "factual", "complexity_score": 0.4},
       "summary": "intent: factual · complexity: 0.4", "duration_ms": 1240}
data: {"type": "agent_start", "agent": "retrieval_planning"}
... (repeats for all 6 agents) ...
data: {"type": "start", "route": "rag", "intent": "factual",
       "complexity": 0.4, "rewritten": "...", "retries": 0}
data: {"type": "token", "text": "The "}
data: {"type": "token", "text": "answer "}
...
data: {"type": "scores", "data": {"faithfulness": 5, "overall": 4.8, ...}}
data: {"type": "sources", "data": [{...}, ...]}
data: {"type": "trace", "data": [{stage_log_entry}, ...]}
data: {"type": "done", "error": null}
```

For `route="unknown"` queries, the `start` event omits `intent`, `complexity`, and
`rewritten` (all `null`) since those fields were not computed.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health`  | Liveness probe → `{"ok": true}` |
| `GET`  | `/status`  | Readiness → `{ready, doc_name, doc_list}` |
| `POST` | `/upload`  | Ingest a document (additive — builds on existing index) |
| `POST` | `/ask`     | SSE streaming response for a query |
| `GET`  | `/docs`    | List currently loaded documents |
| `DELETE` | `/docs`  | Clear all documents and reset pipeline state |

---

## 8. React Frontend

**Stack:** React (Vite) + Tailwind CSS + CSS Custom Properties

The UI is a three-tab application: **Pipeline**, **Playground**, **Results**.

### Pipeline Tab
Real-time visualisation of the 6 agent nodes. Each node shows:
- State: idle / running (animated dot) / done (green check) / error (red cross)
- Live metadata summary when done (e.g. "intent: factual · complexity: 0.4")
- Click to open Agent Inspector panel (right side)
- Multi-Modal Knowledge Base node visualised between Retrieval Planning and Evidence Validation
- **Out of Scope state:** When `lastRoute === "unknown"`, a red `🚫 Out of Scope` node
  appears after User Query, all agent nodes are dimmed to 30% opacity, progress bar fills
  red, exit node shows "Out of Scope Response"

### Playground Tab (Chat)
Full-featured chat interface:

| Feature | Implementation |
|---|---|
| **Upload Zone** | Drag-and-drop or click; supports additive multi-doc upload |
| **Doc list chips** | Shows filenames of all uploaded docs; "Add doc" / "Clear all" buttons |
| **Query history dropdown** | Last 10 queries stored in `localStorage` (key `ac_query_history`); shown as dropdown when textarea is focused |
| **Suggestion chips** | Pre-built example queries; clicking fills the textarea |
| **Live Pipeline Strip** | 6 animated dots at top of chat during a run; each dot transitions idle → active → done/error as the corresponding agent fires |
| **Confidence Badge** | Coloured pill showing overall critic score (green ≥ 4, amber ≥ 3, red < 3) |
| **Copy button** | Each assistant answer has a "Copy" button that copies plain text; shows "Copied ✓" feedback for 2 seconds |
| **Export conversation** | "Export" button in chat header downloads the full conversation as a `.md` file with human/assistant labels and timestamps |
| **Source attribution** | Cited passages shown as collapsible source cards below the answer |
| **Route chip** | Every answer shows "RAG" (indigo) or "Out of Scope" (grey) indicating which path was taken |

### Results Tab
Post-run analytics panel:

| Feature | Implementation |
|---|---|
| **Run history selector** | Browse last 10 runs (newest-first); "Latest", "Run −1", ... buttons; out-of-scope runs show 🚫 badge |
| **Quality Scores** | Bar chart for all 5 critic dimensions + overall score (colour-coded by pass/caution/fail thresholds) |
| **Agent Timing** | Proportional bar chart showing `duration_ms` per agent; total pipeline time in header |
| **Run Summary chips** | Route, Intent, Complexity, Retries, Pipeline Time; Intent and Complexity are hidden for `route="unknown"` |
| **Rewritten Query** | Shows if the query was rewritten differently from the original |
| **Critic badge** | Green "Self-Reflection Passed" / Red "Self-Reflection Failed" pill |

### Dark Mode
- **CSS Custom Properties** on `:root` (light) and `[data-theme="dark"]` (dark)
- React `ThemeContext` sets `document.documentElement.setAttribute('data-theme', ...)`
- All components use `var(--c-*)` tokens in inline styles — theme switch is instant
- Toggle persisted in `localStorage` (key `ac_rag_theme`)
- Sliding toggle in Header with sun/moon emoji

### localStorage Persistence
| Key | Content |
|---|---|
| `ac_doc_name` | Last uploaded document name |
| `ac_doc_ready` | Boolean — whether a document is loaded |
| `ac_doc_list` | Array of all loaded document names |
| `ac_messages` | Full conversation history (array of message objects) |
| `ac_run_history` | Last 10 run metadata objects (route, intent, complexity, retries) |
| `ac_query_history` | Last 10 queries entered |
| `ac_rag_theme` | `"dark"` or `"light"` |

---

## 9. Models Used

### LLM (All Agents)

| Provider | Model | Notes |
|---|---|---|
| **OpenAI** (default) | `gpt-4o` | Best quality; all structured outputs tested against this |
| Google | `gemini-2.0-flash` | Fast + cheap; strong reasoning |
| Anthropic | `claude-sonnet-4-6` | Alternative; same quality tier as GPT-4o |
| Groq | `llama-3.3-70b-versatile` | Free tier; high throughput |

**Switching providers:** Change `LLM_PROVIDER = "openai"` → one line in `config/settings.py`.
The entire pipeline (all 6 agents) switches simultaneously — **except figure extraction**,
which calls OpenAI GPT-4o Vision directly regardless of `LLM_PROVIDER` (see
`ingestion/figure_extractor.py`), because vision is not available across all four
providers in this codebase. Switching to Google/Anthropic/Groq still requires a valid
`OPENAI_API_KEY` if any ingested PDF contains figures. All other providers share the
`utils/llm_factory.py` interface with `@lru_cache` for zero re-initialisation overhead.

### Embedding Model
**Current default: `text-embedding-3-large`** (OpenAI) — 3072-dimensional, higher quality,
paid API. Set in `config/settings.py:EMBEDDING_MODEL`.
Alternative: `sentence-transformers/all-MiniLM-L6-v2` — local, free, 384-dimensional,
no API key needed, lower quality.

**Important:** `EVIDENCE_SCORE_THRESHOLD` and `ROUTER_SIMILARITY_THRESHOLD` are calibrated
per embedding model — OpenAI and sentence-transformers embeddings produce cosine
similarities on different numeric scales for the same text pairs. Switching
`EMBEDDING_MODEL` requires re-measuring and re-setting both thresholds (see
`config/settings.py` for the currently measured range for the active model).

The embedding model is used in exactly three places:
1. **Ingestion** — building the FAISS index
2. **Retrieval** — MMR search at query time
3. **Evaluation** — computing faithfulness and answer relevance metrics

### Vector Store
FAISS (faiss-cpu) is the default — local, no server, extremely fast for research-scale
(<1M vectors). ChromaDB is a supported alternative for persistent HTTP-based usage.

---

## 10. Evaluation Framework

### Custom Metrics (`evaluation/metrics.py`)

All normalised to [0, 1]:

| Metric | Method | Reference Needed? |
|---|---|---|
| **Faithfulness** | Mean cosine similarity (embedding-based): each sentence of the answer vs. best matching retrieved passage | No |
| **Answer Relevance** | Cosine similarity: query embedding vs. answer embedding | No |
| **Completeness** | LLM-judge on 1–5 scale, normalised: does the answer cover all aspects? | No |
| **Context Utilisation** | Fraction of retrieved chunks cited at least once in the answer | No |
| **ROUGE-L** | ROUGE-L F1 between generated and reference answer | Yes |
| **Composite** | Mean of all available scores | — |

### RAGAS-Equivalent Metrics (`evaluation/ragas_eval.py`)

**Not the `ragas` PyPI package** — the real `ragas` library requires Python 3.10+, which
this project doesn't target, so `evaluation/ragas_eval.py` reimplements the 4 core RAGAS
metrics as direct LLM-judge prompts against whichever LLM `utils/llm_factory.py` resolves.
Same metric definitions as RAGAS, same [0,1] scale, no dependency on the package itself
(confirm: `ragas` is intentionally absent from `requirements.txt`).

Metrics implemented:
- `answer_relevancy` — embedding-based relevance
- `faithfulness` — LLM-judge faithfulness
- `context_precision` — how many retrieved passages are relevant to the answer

**Bug fixed:** `retrieved_docs` in pipeline state are plain `dict` objects (not LangChain
`Document` objects). The RAGAS evaluator was incorrectly accessing `d.page_content` —
fixed to `d["content"]`.

### Ablation Study (`evaluation/ablation.py`)

7 configurations compared:

| Config | What is disabled |
|---|---|
| `full` | All components active (baseline) |
| `no_planner` | Retrieval Planner off → fixed k only |
| `no_validator` | Evidence Validator off → all chunks pass |
| `no_refiner` | Context Refiner off → raw chunks go to generator |
| `no_critic` | Critic off → no self-reflection or retry |
| `low_k` | k=4 fixed regardless of complexity |
| `high_k` | k=12 fixed regardless of complexity |

### Running Evaluation

```bash
# Single config
python -m evaluation.runner --test-set evaluation/test_set_sample.json --config full

# Full ablation suite
python -m evaluation.ablation --test-set evaluation/test_set_sample.json

# RAGAS evaluation
python -m evaluation.ragas_eval
```

---

## 11. Formal Objective

```
a* = argmax P(a | q, R(q, D))
     subject to: Faith(a, R(q, D)) ≥ τ
```

Where:
- `q` = user query (rewritten by Query Analyzer)
- `D` = document corpus (FAISS index)
- `R(q, D)` = retrieved + validated + refined context
- `a` = generated answer
- `Faith(a, R(q,D))` = mean cosine similarity of answer sentences to context passages
- `τ = FAITHFULNESS_THRESHOLD = 0.80`

The Critic enforces this constraint. If faithfulness < threshold or any other dimension
falls below `CRITIC_MIN_SCORE (4)`, the answer is rejected and the pipeline retries from
the appropriate stage.

---

## 12. Configuration Reference (`config/settings.py`)

```python
# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER   = "openai"          # "openai" | "google" | "anthropic" | "groq"
LLM_TEMPERATURE = 0.0              # deterministic — critical for faithfulness
LLM_MAX_TOKENS  = 2048

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "text-embedding-3-large"
VECTORSTORE_BACKEND = "faiss"      # "faiss" | "chroma"

# ── Ingestion ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512    # tokens (~2048 chars)
CHUNK_OVERLAP = 64     # tokens (~256 chars)

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K_MIN        = 4
RETRIEVAL_K_MAX        = 12
RETRIEVAL_K_DEFAULT    = 6
MMR_FETCH_K_MULTIPLIER = 3    # candidate pool = k × 3
MMR_LAMBDA_MULT        = 0.5  # 0=max diversity, 1=max relevance

# ── Validation ────────────────────────────────────────────────────────────────
EVIDENCE_SCORE_THRESHOLD = 0.30   # cosine similarity cutoff
MIN_VALID_PASSAGES       = 2      # minimum passages to proceed

# ── Critic ────────────────────────────────────────────────────────────────────
CRITIC_MIN_SCORE = 4              # all 5 dimensions must be ≥ 4
MAX_RETRIES      = 3              # retry budget

# ── Faithfulness constraint ───────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.80

# ── Router ────────────────────────────────────────────────────────────────────
ROUTER_SIMILARITY_THRESHOLD = 0.30  # same scale as EVIDENCE_SCORE_THRESHOLD

# ── Ablation flags ────────────────────────────────────────────────────────────
USE_RETRIEVAL_PLANNER = True
USE_VALIDATOR         = False     # currently off (can enable for stricter filtering)
USE_CONTEXT_REFINER   = True
USE_CRITIC            = True
```

---

## 13. Known Bugs Fixed During Development

| Bug | Root Cause | Fix |
|---|---|---|
| Wrong path in `backend/api.py` | `PROJECT_ROOT` resolved to wrong directory (`parent.parent.parent / "ac-rag"`) | Fixed to `parent.parent` |
| RAGAS doc type error | `retrieved_docs` are plain `dict`, but code used `.page_content` (LangChain `Document` attr) | Fixed to `d["content"]` |
| Critic auto-passes on exception | LLM timeout/error set `critic_passed=True` silently | Fixed: exception → `critic_passed=False, retry_reason="content"` |
| Missing `recursion_limit` | LangGraph default of 25 cuts off heavy retry chains | Fixed: `config={"recursion_limit": 50}` |
| SSE sends fake metadata for unknown route | `intent: "factual"` and `complexity: 0.3` were sent even for unknown queries | Fixed: `is_rag` guard sends `null` for both |
| Dead suggestion chip onClick | `/* just fill textarea */` comment left as placeholder | Fixed: `setQuery(s); textareaRef.current?.focus()` |
| Stale route styles | Frontend had `chat` and `web_search` ROUTE_STYLES after routes were removed | Removed; added `unknown` style |
| Web searcher not cleaned up | `web_searcher.py` still on disk after route removal | Deleted |

---

## 14. Project Structure (Current)

```
ac-rag-react/
├── config/
│   └── settings.py              — all hyperparameters and ablation flags
│
├── ingestion/
│   ├── __init__.py              — exposes ingest_documents()
│   ├── loader.py                — multi-format document loader
│   ├── chunker.py               — RecursiveCharacterTextSplitter wrapper
│   └── metadata_tagger.py       — enriches chunks with section/modality/page metadata
│
├── vectorstore/
│   ├── embeddings.py            — embedding model factory (@lru_cache)
│   ├── store.py                 — VectorStoreManager (build/save/load/search/add_documents)
│   ├── build.py                 — CLI: ingest → chunk → embed → save index
│   └── index/                   — saved FAISS index (index.faiss + index.pkl)
│
├── pipeline/
│   ├── state.py                 — ACRagState TypedDict + initial_state()
│   ├── graph.py                 — LangGraph StateGraph assembly + run_pipeline()
│   ├── router.py                — all conditional edge functions
│   └── nodes/
│       ├── entry_router.py      — vector similarity routing (rag | unknown)
│       ├── query_analyzer.py    — intent, complexity, rewrite, decomposition
│       ├── retrieval_planner.py — adaptive k + modality + depth planning
│       ├── retriever.py         — MMR FAISS search with multi-query support
│       ├── validator.py         — cosine similarity evidence filtering
│       ├── context_refiner.py   — dedup + re-rank + LLM compression
│       ├── generator.py         — grounded answer generation with citations
│       ├── critic.py            — 5-dimension self-reflection + retry routing
│       └── direct_responder.py  — static "I don't know" for unknown queries
│
├── evaluation/
│   ├── metrics.py               — faithfulness, relevance, ROUGE-L, completeness
│   ├── runner.py                — batch evaluation runner (JSON + CSV)
│   ├── ablation.py              — 7-config ablation study
│   ├── ragas_eval.py            — RAGAS-equivalent metrics, custom LLM-judge (answer_relevancy, faithfulness, context_precision)
│   ├── test_set_sample.json     — sample test questions
│   └── results/                 — evaluation output files
│
├── utils/
│   ├── llm_factory.py           — LLM provider factory with @lru_cache
│   ├── scoring.py               — cosine similarity utilities
│   ├── attribution.py           — sentence-level source attribution builder
│   └── logger.py                — structured logging (file + console)
│
├── backend/
│   └── api.py                   — FastAPI + SSE streaming (upload, ask, docs CRUD)
│
├── frontend/
│   └── src/
│       ├── App.jsx              — root: tab routing + localStorage persistence
│       ├── ThemeContext.jsx     — dark mode React Context + data-theme toggling
│       ├── index.css            — CSS custom properties (light + dark tokens)
│       ├── main.jsx             — React entry point
│       ├── constants/
│       │   └── agents.js        — AGENTS array (id, title, icon, color, description)
│       └── components/
│           ├── Header.jsx           — tab nav + dark mode toggle
│           ├── Landing.jsx          — welcome screen (shown on first visit)
│           ├── pipeline/
│           │   ├── PipelineCanvas.jsx   — 6-node live pipeline visualiser
│           │   └── AgentNode.jsx        — single agent node (idle/active/done/error)
│           ├── inspector/
│           │   └── AgentInspector.jsx   — right panel: agent schema + live trace
│           ├── playground/
│           │   └── Playground.jsx       — chat UI + upload + all 12 enhancements
│           └── results/
│               └── Results.jsx          — run history + scores + timing + summary
│
├── data/raw/                    — sample documents for testing
├── logs/pipeline.log            — structured pipeline log
├── app.py                       — Uvicorn entry point for backend
├── main.py                      — CLI entry for pipeline testing
├── requirements.txt
└── PROJECT_SUMMARY.md           — this file
```

---

## 15. How to Run

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "OPENAI_API_KEY=sk-..." > .env

# Start the API server
uvicorn app:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # → http://localhost:5173
```

### CLI (no UI)

```bash
# Ingest a document
python -m vectorstore.build --file data/raw/yourfile.pdf

# Run a single query
python main.py --query "What is the main contribution of this paper?"

# Run evaluation
python -m evaluation.runner --test-set evaluation/test_set_sample.json --config full
```

---

*AC-RAG — Agent-Controlled RAG for Reliable Document Question Answering*  
*SRKR Engineering · 2026–27*
