# AC-RAG — Documented Failure Cases

Gap fix (GAPS_AND_IMPROVEMENTS.md item B.13 — "no failure/error analysis section").
Real failures from the corrected evaluation run (`evaluation/results/results_full_20260820_001122.json`,
15-question stratified subset, `full` config, post threshold/retry-escalation fix — see
GAPS_AND_IMPROVEMENTS.md item A.1). This is more convincing to a jury than aggregate
scores alone: it shows the system's actual failure boundary, not just its wins.

## The failure class: precise numeric/procedural config-value lookups

All 4 weakest results in the 15-question set are the same underlying failure mode:

| ID | Query | Outcome | Composite |
|---|---|---|---|
| 20 | "What is the maximum number of retries allowed?" | Max retries exhausted, no answer | 0.000 |
| 7  | "What is the retrieval range (k)?" | "Insufficient information" (honest, not wrong) | 0.106 |
| 13 | "What happens if insufficient valid passages remain after validation?" | "Insufficient information" | 0.160 |
| 19 | "What happens when formatting issues are detected?" | "Insufficient information" | 0.167 |

### Root cause

These questions ask for a specific config value or procedural detail that exists in
the source document as a bare number in a settings table or a short factual sentence
buried in prose (e.g. `MAX_RETRIES = 3` in a config listing, or "a retrieval retry
counter restricts the number of attempts"). The question's natural-language phrasing
and the source text's phrasing share very little lexical or semantic surface — cosine
similarity between the two tops out around 0.20-0.28 even for the single genuinely
correct passage (measured directly, see GAPS_AND_IMPROVEMENTS.md item A.1). This is a
structural property of embedding-similarity search for this query type, not a bug:
"what is the maximum number of retries" and "MAX_RETRIES = 3" are semantically the
same fact but embed quite differently.

### Why id=20 fails harder than 7/13/19

Query 20's actual correct passage (the `MAX_RETRIES = 3` line as it appears in a dense
config table) scored so low that even after 3 retries with progressively widened k
(retry-escalation fix, item A.1) it never cleared `MIN_VALID_PASSAGES = 2`. The other
three found *some* borderline-relevant passages, enough for the generator to run and
honestly report "insufficient information" rather than hallucinate a wrong number —
which is the entry router / generator's core design goal working as intended, just at
a low-confidence extreme.

### Update (2026-08-20): id=20 fixed by real table extraction, not threshold tuning

The root cause turned out to be more specific than "cosine similarity struggles with
numeric lookups" — the actual `MAX_RETRIES = 3` value lives in a genuine PDF table
(a "Parameter | Value" config table on page 9), which PyMuPDF's plain-text extraction
flattens into unstructured prose with no table markers. `metadata_tagger.py`'s
table-detection heuristic (regex for pipe/tab characters) then found nothing to tag,
so this content never got a "table" modality tag distinguishing it from noise — see
GAPS_AND_IMPROVEMENTS.md item C.14. Adding real structural table extraction
(`ingestion/table_extractor.py`, pdfplumber) pulled this exact table out cleanly as
markdown, and the query now resolves in a single attempt (retry_count=0) with the
correct answer: "The maximum number of retries allowed is three for both retrieval
retries and reflection retries."

This is a better fix than the "hybrid BM25" suggestion below for this specific case —
worth keeping both points for the report: the config-table content wasn't just hard to
embed, it was **structurally invisible** to the modality system entirely before this fix.

### What this means for the report

- Lowering `EVIDENCE_SCORE_THRESHOLD` further to force weak matches through would let
  noisier, less relevant content pass validation for every other query too — a net
  quality loss. The real fix for this failure class was giving the retriever a clean,
  correctly-tagged structural representation of the content, not tuning a similarity
  threshold against a flattened, noisy one.
- A remaining, more general future-work item for content that resists BOTH table
  extraction and embedding similarity: a **hybrid retrieval** approach (BM25/keyword
  search alongside dense embedding search) would catch lexical-overlap cases like
  "MAX_RETRIES" vs "maximum number of retries" even outside of tables.
- The system's behavior on failure is itself evidence for the "no hallucination" claim:
  the other weak cases (ids 7, 13, 19) produced an honest "insufficient information"
  answer rather than a fabricated one, even under low-confidence retrieval.

## Minor run-to-run variance note

Comparing this run to an earlier verification run on the same 15 questions (both at
`LLM_TEMPERATURE=0.0`), the exact set of erroring IDs shifted slightly (3 errors
including id=19 in one run, 2 errors excluding id=19 in another). GPT-4o at
temperature 0 is *nearly* but not perfectly deterministic in practice — worth
mentioning if a jury member asks about reproducibility, and worth running evaluation
more than once before reporting a single number as ground truth.
