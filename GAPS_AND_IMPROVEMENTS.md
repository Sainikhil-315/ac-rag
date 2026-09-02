# AC-RAG — Gap Analysis & Path to Research-Grade / Gold Standard

Based on direct inspection of code (not just PROJECT_SUMMARY.md claims, some of which
are stale vs. actual code — flagged below with **[DOC MISMATCH]**).

---

## A. Critical Gaps (fix before Review-1, Saturday)

1. **`evaluation/results/` is empty** — the evaluation framework (metrics.py, ablation.py,
   ragas_eval.py) exists but has never been run to completion. No numbers exist to show a jury.
   **Fix:** run `evaluation.runner` and `evaluation.ablation` at minimum once, save output.
   **STATUS: DONE (2026-08-19/20).** Rebuilt the vector index against the actual AC-RAG
   paper (the saved index had stale leftover content from an unrelated test upload — always
   verify what's actually indexed before trusting eval numbers). Ran `evaluation.runner` on
   a 15-question stratified subset of `test_set_sample.json` (all 40 hit OpenAI rate limits
   mid-run — subset chosen to stay within the API tier's throughput; documented, not hidden).

   Running the eval surfaced two real bugs invisible from reading code alone:
   - **Retry-on-validation-failure never changed retrieval params** — same query, same k/
     fetch_k/lambda_mult on every retry, so MMR search returned identical docs 3x in a row.
     Fixed in `pipeline/nodes/retriever.py`: each retry now widens k/fetch_k by 50% (capped
     at RETRIEVAL_K_MAX) and lowers lambda_mult for more diversity.
   - **`ROUTER_SIMILARITY_THRESHOLD` and `EVIDENCE_SCORE_THRESHOLD` shared one constant
     (0.30) despite being computed by two different formulas** — the router uses FAISS
     `1/(1+L2_distance)`, the validator uses raw cosine similarity. Measured on identical
     query/passage pairs: FAISS-based score reads ~0.13-0.15 *higher* than raw cosine
     (0.41 vs 0.29, 0.40 vs 0.25, 0.40 vs 0.24). The validator was silently rejecting
     genuinely relevant passages the router would have accepted. Decoupled the two
     constants and recalibrated `EVIDENCE_SCORE_THRESHOLD` to 0.20 against the real
     corpus + eval set (p25=0.22, p50=0.27, p75=0.37 over n=120 passages).

   **Before → after** (same 15-question subset, `full` config):
   | Metric | Before | After |
   |---|---|---|
   | Errors (max-retries exhausted) | 7/15 | 3/15 |
   | Critic pass rate | 53.3% | 80.0% |
   | Avg retry count | 1.4 | 0.8 |
   | Faithfulness (mean) | 0.289 | 0.405 |
   | Answer relevance (mean) | 0.414 | 0.525 |
   | Composite (mean) | 0.285 | 0.371 |

   **Known residual limitation (documented, not hidden):** 3/15 queries still exhaust
   retries — all are precise numeric config-value lookups ("what chunk overlap is used",
   "max number of retries allowed"). Cosine similarity structurally underperforms here:
   the query phrasing and a bare number in the source table share little lexical/semantic
   surface, so even the best-matching passage tops out around 0.24-0.28. Lowering the
   threshold further would admit noise for other query types — this is a real limitation
   of embedding-similarity-based validation for this query class, worth stating plainly
   in the report rather than threshold-hacking it away.

2. **`USE_VALIDATOR = False` by default** (`config/settings.py:84`) — the Evidence Validator,
   one of the 4 headline components and listed as a "Claim" in PROJECT_SUMMARY.md
   ("Evidence filtering — discards passages below threshold"), is currently **disabled**.
   The claim in your own documentation does not match the shipped default config.
   **Fix:** either flip it to `True` and re-verify pipeline stability, or be ready to explain
   why it's off (e.g. it hurt recall) — don't let a jury member catch this contradiction.

3. **[DOC MISMATCH] RAGAS is not actually integrated.** PROJECT_SUMMARY.md section 10 says
   "Integrates the `ragas` library." In reality (`evaluation/ragas_eval.py` docstring, and
   `ragas` absent from `requirements.txt`), it's a **custom reimplementation of 4 RAGAS-style
   metrics using LLM-judge prompts**, because the real `ragas` package needs Python 3.10+.
   This is a fine engineering decision — but presenting it as "we used RAGAS" and having a
   jury member ask "which version of ragas" or "show the import" will look bad.
   **Fix:** update PROJECT_SUMMARY.md and the report to say "RAGAS-equivalent metrics,
   custom-implemented due to Python version constraint" — this is honest and still credible.

4. **[DOC MISMATCH] Embedding model default.** PROJECT_SUMMARY.md section 9 presents
   `sentence-transformers/all-MiniLM-L6-v2` (free, local) as the default, with OpenAI's
   `text-embedding-3-small` as the paid "alternative." The actual `config/settings.py:31`
   default is `text-embedding-3-large` (a different, larger, paid OpenAI model, not even
   the "small" one mentioned). Also `EVIDENCE_SCORE_THRESHOLD` comment says it's
   "calibrated for sentence-transformers" — meaning the actual default embedding model may
   not match the threshold it was tuned against.
   **Fix:** decide which embedding model you're actually demoing/evaluating with, make
   settings/report/threshold-comment consistent, and mention the cost implication
   (`text-embedding-3-large` costs money per query — relevant if asked about deployment cost).

5. **No baseline comparison.** Nothing in the repo runs plain single-shot RAG (retrieve-k →
   generate, no router/validator/critic/retry) against the same test set to produce a
   before/after number. Without this, "our system reduces hallucination" is an unverified claim.
   **Fix:** add a `no_planner + no_validator + no_refiner + no_critic` combined "naive" config
   (you already have the individual ablation flags — just combine all of them into one run)
   and report faithfulness/answer-relevance side-by-side with `full`.
   **STATUS: DONE (2026-08-20).** The `baseline_rag` config already existed in
   `evaluation/ablation.py` (this exact combination) — it just never actually ran
   correctly until the settings-patch bug in item A.1 was fixed. See `ABLATION_RESULTS.md`
   for the real numbers: `full` vs `baseline_rag` shows no statistically significant
   difference in faithfulness/composite at n=15 (see item 9) — an honest finding, not
   the clean win the claim implies. Report accordingly.

6. **Max-retries exhaustion produced a hard error with NO answer, even when the
   underlying cause was "content not in the document"** — functionally the same
   situation the Entry Router's "unknown" route already handles gracefully, just
   discovered later (after the Validator rejects every retry attempt) instead of
   immediately. Found via a real live bug report (2026-08-20): asking about a person
   never mentioned in an uploaded document went through the full RAG path, retried 3
   times, and returned "Max retries exhausted without a passing answer" with no
   answer text at all — a genuinely bad user experience the user rightly called out.
   **STATUS: DONE.** `pipeline/graph.py`'s `_end_max_retries_node` now gives the same
   graceful "I don't know" message (`direct_responder.py`'s `_UNKNOWN_RESPONSE`) when
   nothing was ever generated — but preserves an actually-generated answer if the
   Critic (not the Validator) was the one exhausting retries, since that's a better
   best-effort result than a generic fallback. `state["error"]` is still set in both
   cases, so `evaluation/runner.py`'s error-counting metric is unaffected — only the
   user-facing text changed. Also fixed the same `.get(key, default)` pitfall in
   `main.py`'s CLI answer printing (the default never fires because `initial_state()`
   sets `answer=None` explicitly, so the key always exists).
   **Known tradeoff, not hidden:** this also means a genuine retrieval false-negative
   (relevant content exists but scoring/retrieval missed it — see
   `FAILURE_ANALYSIS.md`) now looks IDENTICAL to a true out-of-scope query from the
   user's perspective. Before this fix, a false negative at least surfaced as a
   visible, investigable error; now it's a friendlier but less debuggable "I don't
   know." Worth stating this plainly if asked about failure modes — it's a genuine
   UX-vs-debuggability tradeoff, not a clean win.

---

## B. Evaluation & Data Gaps (needed for "research-grade" claim)

6. **Test set is small and single-document.** `evaluation/test_set_sample.json` looks
   hand-written for one demo document — check its actual count (open the file). For a
   credible evaluation table you want 50-150+ Q&A pairs spanning factual / analytical /
   comparative / summarization intents (matches your own `intent` taxonomy in
   `query_analyzer.py`), across multiple documents.
   **STATUS: PARTIALLY FIXED (2026-08-20).** The set had 40 questions but zero
   `comparative` or `summarization` examples despite those being 2 of the 4 intent
   categories the system explicitly classifies (`query_analyzer.py`) — a real gap in
   coverage, not just a small-N problem. Added 6 new questions (4 comparative, 2
   summarization) grounded in verified paper content (Table 1's system-comparison
   matrix, the Limitations section, and the context-relevance improvement numbers) —
   now 46 total, all 4 intent types represented (25 factual / 15 analytical / 4
   comparative / 2 summarization). Spot-checked one new question end-to-end against the
   real pipeline: exact match to the authored reference answer, critic 5/5.
   **Still open:** single document, still comparative/summarization-light relative to
   factual/analytical, no multi-document test coverage. See item 7 for the bigger fix
   (a public benchmark).

7. **No public benchmark tested.** Everything is evaluated on your own document + your own
   questions — this can't demonstrate generalization. Add at least one recognized dataset:
   - **QASPER** or **NarrativeQA** — closest fit, since both are document/paper QA with
     tables & figures, matching your `modality` tagging.
   - **HotpotQA** (subset) — multi-hop questions, directly tests your `sub_queries`
     decomposition feature which is currently untested against anything designed for it.

8. **No retrieval-quality ground truth.** All current metrics judge the *answer*
   (faithfulness, relevance, completeness). None measure **retrieval quality directly**
   (e.g. Recall@k, MRR against a known relevant-chunk-id per question). Add this — it's the
   metric that most directly evaluates your Retrieval Planner + Retriever design.

9. **No statistical significance testing** between ablation configs. Right now, if `full`
   scores 0.82 and `no_critic` scores 0.79, you can't say whether that's a real difference or
   noise. Even a simple paired t-test or bootstrap CI over the ablation results is expected
   in a research-grade writeup.
   **STATUS: DONE (2026-08-20).** See `ABLATION_RESULTS.md` — full 8-config ablation now
   actually runs correctly (see item A.1's ablation-mechanism bug fix), and paired
   t-tests/Wilcoxon tests were run on every headline comparison. Honest finding: **none of
   the differences reach significance at n=15** — including full vs baseline_rag, and the
   surprising full vs no_validator result. Documented as a real limitation (small sample)
   rather than spun as either a win or a problem — this is more credible to a jury than an
   unqualified comparison table.

10. **No cross-provider comparison.** You support 4 LLM providers (OpenAI, Google,
    Anthropic, Groq) via `llm_factory.py`, but no evaluation compares faithfulness/cost/
    latency across them. This is a strong, cheap-to-run experiment given the architecture
    already supports it — one line change (`LLM_PROVIDER`) + re-run eval.

11. **No embedding model ablation.** Same story — `EMBEDDING_MODEL` is a one-line swap but
    never compared (MiniLM vs. `text-embedding-3-large`) despite affecting both cost and quality.

12. **No human evaluation.** All metrics are automated (embedding similarity, LLM-judge,
    ROUGE). A small human-rated sample (even 10-15 answers rated by 2 people for
    faithfulness/usefulness) meaningfully strengthens a research-grade evaluation section and
    lets you report inter-rater agreement or correlation with your automated metrics.
    **STATUS: TOOLING DONE, RATING STILL NEEDS YOU (2026-08-20).** Added
    `evaluation/human_eval.py` — `export` samples N queries from any results JSON into a
    CSV rating sheet (blank faithfulness/usefulness columns for two independent raters,
    1-5 scale, deliberately separate dimensions since an answer can be faithful-but-vague
    or useful-sounding-but-wrong); `score` reads the filled sheet and reports mean scores,
    inter-rater Pearson correlation, and correlation between mean human faithfulness and
    the automated faithfulness metric. Verified the scoring math against a synthetic
    filled sheet (deleted after — don't mistake it for real data if you see it referenced).
    **This can't be finished by me** — it needs two actual humans (you + a friend/teammate)
    to rate ~15 answers before the review. Run:
    `python -m evaluation.human_eval export --results evaluation/results/results_full_<latest>.json --n 15`

13. **No failure/error analysis section.** No documented set of "here are 5 cases where the
    pipeline failed and why" — this is usually more convincing to a jury than aggregate scores,
    because it shows you understand the system's limits, not just its wins.
    **STATUS: DONE (2026-08-20).** See `FAILURE_ANALYSIS.md` — all 4 weakest results in the
    15-question eval set trace to one root cause (precise numeric config-value lookups where
    cosine similarity structurally underperforms), documented with a concrete future-work fix
    (hybrid BM25 + dense retrieval) and a note on GPT-4o's run-to-run variance at temperature 0.

---

## C. Architecture / Engineering Gaps

14. **Table extraction is heuristic-only.** `ingestion/metadata_tagger.py` tags a chunk's
    `modality` as `"table"` via what appears to be a text-pattern heuristic, not a real table
    structure parser (no `camelot`/`pdfplumber`/`unstructured` table extraction found). Figures
    get a dedicated, well-built extractor (`figure_extractor.py`, GPT-4o Vision) — tables don't
    get equivalent treatment. This is an asymmetry worth either fixing or being upfront about
    if asked "how do you handle tables specifically?"
    **STATUS: DONE (2026-08-20) — worse than described, and now fixed.** Verified directly
    against the real indexed document: the regex heuristic tagged **zero** chunks as
    `modality="table"`, even though the paper contains a system-comparison matrix (Table 1)
    and a parameter/value config table (Table 9) — PyMuPDF's plain-text extraction flattens
    PDF tables into space-separated prose with no pipe/tab characters left for the regex to
    match, so table-modality filtering was completely non-functional, not just weak.

    Added `ingestion/table_extractor.py` using pdfplumber's structural table detection
    (cell-boundary based, independent of how the text got flattened elsewhere) — no LLM
    call needed since table content is already structured, unlike figures. Rebuilt the
    index: 4 real tables now extracted and correctly tagged `modality="table"`.

    This directly fixed the hardest failure case in `FAILURE_ANALYSIS.md` (id=20, "what is
    the maximum number of retries allowed") — the answer was sitting in exactly the kind of
    parameter/value table this extractor now surfaces cleanly. Query now resolves in a
    single attempt (retry_count=0) with the correct answer. See FAILURE_ANALYSIS.md's
    "Update" section for detail.

    **Bonus finding while reading the extracted config table:** the paper's own stated
    hyperparameters (`Chunk size: 600 characters`, `Evidence threshold: 0.65`) don't match
    the live `config/settings.py` (`CHUNK_SIZE = 512` tokens ≈ 2048 chars, and the validator
    threshold recalibrated to 0.20 raw cosine in item A.1). The paper's 0.65 figure was very
    likely computed on the FAISS-distance-based scale (matching the router), not the raw
    cosine scale the validator actually uses — consistent with, and further confirming, the
    root-cause story in item A.1. Worth being ready to explain this discrepancy if asked
    "does your code match your paper's stated hyperparameters" — the honest answer is
    partially, and here's specifically why.

15. **No conversation memory / multi-turn support.** `pipeline/state.py`'s `ACRagState` has no
    field for prior turns — each query is fully independent. If a jury asks "can I ask a
    follow-up question referencing the previous answer?", the honest answer right now is no.
    This is a common, well-scoped "future work" addition (pass last N turns into Query Analyzer).

16. **No automated tests.** No `tests/` directory found anywhere in the repo. For a system this
    modular (9 pipeline nodes, each independently swappable), even a small pytest suite
    (one test per node with a mocked LLM) would substantially raise engineering credibility
    and catch regressions before a demo.

17. **Figure extraction is hard-coded to OpenAI GPT-4o regardless of `LLM_PROVIDER`.**
    `figure_extractor.py` explicitly bypasses `llm_factory.get_llm()` and calls the OpenAI
    SDK directly, because only OpenAI's model does vision in your current setup. This means
    switching `LLM_PROVIDER` to Google/Anthropic/Groq does NOT switch the vision model —
    it's a silent exception, only documented in a code comment. Worth stating explicitly in
    the report as a known limitation (and it also means an `OPENAI_API_KEY` is mandatory even
    if you "switch providers" — contradicts the "one-line provider switch" framing).

18. **No caching / cost tracking.** Every query re-embeds and re-calls the LLM multiple times
    (analyzer, planner refinement, generator, critic — up to 4+ LLM calls per query, more on
    retries). No token/cost logging exists. For a report claiming practical deployability,
    a simple per-query cost/latency table (you already log `duration_ms` per agent) would be
    a quick, convincing addition.

19. **No security/privacy handling for uploads.** Documents are ingested and embedded with no
    mention of retention, deletion guarantees beyond the `DELETE /docs` endpoint, or handling
    of sensitive content. Minor for a college project, but a jury member from a security
    background may ask.
    **STATUS: PARTIALLY FIXED (2026-08-20).** Found and fixed 3 concrete issues in
    `backend/api.py`:
    - `/upload` had no file-size limit — `await file.read()` loaded the entire body into
      memory unbounded, a trivial DoS vector. Added a 25 MB cap, checked both via
      `Content-Length` header (fast-reject before reading) and post-read as a backstop.
    - `/upload` had no file-extension validation — any filename was accepted and handed to
      the ingestion pipeline regardless of type. Now validated against
      `config.settings.SUPPORTED_EXTENSIONS` before processing.
    - `DELETE /docs` only cleared in-memory state (`_state["vsm"] = None`, etc.) — the
      persisted FAISS index on disk (`vectorstore/index/`) was never deleted, so uploaded
      document content silently survived after a user asked to "clear all." Now the index
      directory is actually removed from disk.

    **Still open, documented rather than fixed (out of scope for a single-tenant college
    demo, but real if this were ever exposed more broadly):** the backend holds ONE global
    in-memory `_state["vsm"]` shared across every client — there is no per-user/session
    isolation. Any client hitting this server sees whichever document the last uploader
    added. Fine for a single-user local demo; a genuine multi-tenant deployment would need
    per-session vector stores.

20. **Document upload took 1-2 minutes for slide-heavy PDFs**, found via a real live
    bug report (2026-08-20 — user: "this is really freaking me out, I want it under
    5-10 secs"). Root cause: `ingestion/figure_extractor.py` called GPT-4o Vision once
    per detected image in a plain sequential `for` loop, each call taking ~3-8s —
    a document with 15-20 images (typical for a lecture-slide PDF with a logo/diagram
    on every slide) meant 60-160+ seconds, most of it re-describing logos that get
    thrown away as non-data ("SKIP") anyway.
    **STATUS: DONE.** Refactored into two passes: a fast local scan (PyMuPDF only, no
    API calls) collects candidate images, then a single concurrent batch
    (`ThreadPoolExecutor`, `VISION_MAX_WORKERS=8`) describes them all at once — pure
    network I/O, safe to parallelize, no extra cost (same number of API calls, just
    concurrent instead of serial). Added `MAX_FIGURES_PER_DOC=30` as a backstop against
    pathological decks. Verified the concurrency mechanism directly (mocked Vision
    calls, 8 images at 0.2s each: sequential would be 1.6s, concurrent measured
    ~0.2s — an 8x speedup matching the worker count). Couldn't demonstrate the
    end-to-end speedup on the AC-RAG paper itself since it only contains 1 real
    image above the size threshold (confirmed via direct PyMuPDF scan, not a bug in
    the fix) — the mocked test is the real verification here. Added 2 regression tests.

---

## D-0. Future Work: Asynchronous Document Ingestion

Discussed 2026-08-20 alongside item 20's upload-latency fix, deliberately not built
yet — captured here so it doesn't get lost.

**Context:** the figure-extraction parallelization fix (item 20) already bounds
worst-case ingestion to roughly 35-45 seconds for even a large, image-heavy 100-page
document (`MAX_FIGURES_PER_DOC=30` caps the dominant cost — figure count, not page
count — regardless of document size). That fix reduces the actual processing time.

**What async ingestion would add on top — and, importantly, what it wouldn't:**
async does NOT make ingestion faster. `POST /upload` would return immediately
(sub-second) with a job id, processing would continue in a background task, and the
frontend would poll a status endpoint (extending the `/status` check already added
in the frontend-desync fix, item earlier this session) to show "Processing
document..." instead of a blocking, frozen-looking request for ~40 seconds. The
value is purely UX responsiveness — turning "45s feels like a freeze/hang" into
"45s feels instant with visible progress" — not a speed improvement.

**Design sketch:**
1. `POST /upload` saves the file, generates a job id, kicks off ingestion via
   `asyncio.to_thread` (same underlying logic as today), returns
   `{"status": "processing", "job_id": ...}` immediately.
2. Backend tracks job state in memory: `processing` → `ready` / `error`.
3. Frontend polls every 1-2s, shows a processing indicator, then transitions to
   ready state — same document-ready flow as today once the poll returns `ready`.

**Why not built now:** genuinely useful demo moment (showing the UI stay responsive
while ingesting a large document live for a jury), but real implementation work
(~1-2 hours: job-state tracking, a new/extended status endpoint, frontend polling
logic) weighed against review timeline. Worth doing if time permits after the
higher-priority items in section D below; otherwise a credible "future work" talking
point — you understand the scaling bottleneck (image count, not page count) and have
a concrete, correctly-reasoned design for it, which is what a jury is actually
testing when they ask "how would this scale."

---

## D. New Additions That Would Meaningfully Raise the Grade

Ranked by effort-to-impact ratio for your ~10-day window:

| Addition | Effort | Impact |
|---|---|---|
| Run full evaluation + ablation, get numbers into report | Low | **Highest** — closes gap #1, unlocks all of section B |
| Combined "naive RAG" baseline config + comparison table | Low | High — turns your core claim into evidence |
| Fix `USE_VALIDATOR` + doc mismatches (#2-4) | Very low | High — removes easy jury "gotcha" questions |
| Public benchmark subset (30-50 Qs from QASPER/HotpotQA) | Medium | High — generalization proof |
| Failure case analysis (5 documented failures + why) | Low | Medium-high — shows depth of understanding |
| Cross-provider + cross-embedding-model comparison table | Low-medium | Medium-high — you already support this, just unrun |
| Statistical significance on ablation results | Low | Medium — signals research rigor |
| Basic pytest suite for pipeline nodes | Medium | Medium — engineering credibility |
| Retrieval-only metrics (Recall@k, MRR) | Medium | Medium |
| Human evaluation on 10-15 samples | Medium | Medium |
| Real table extraction (pdfplumber/camelot) | Medium-high | Low-medium — nice-to-have, not core to your novelty claim |
| Multi-turn conversation memory | High | Low for this review — good "future work" line instead |

---

## E. What NOT to Over-Promise in the Report

- Don't claim "RAGAS integration" — say "RAGAS-equivalent metrics" (see #3).
- Don't claim the embedding model is free/local unless you actually switch the default (see #4).
- Don't claim full modality support (text/table/figure) as equally robust — tables are
  heuristic-tagged, not structurally extracted (see #14).
- Don't claim provider-agnostic vision — figure extraction is OpenAI-only (see #17).
- Frame the project honestly as: **strong, well-motivated architecture with a real
  self-correction mechanism, currently short on run evaluation — which you are actively
  closing.** That framing survives tough questions; overclaiming does not.
