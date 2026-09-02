# AC-RAG — Public Benchmark Results (SQuAD v2.0)

Gap fix (GAPS_AND_IMPROVEMENTS.md item B.7). First out-of-domain evaluation — every
other evaluation in this repo tests against the AC-RAG paper itself, the one document
the system was developed and tuned against. This tests generalization to content the
system has never seen: a real SQuAD v2.0 subset (CC BY-SA 4.0, topic
"Computational_complexity_theory"), 20 answerable + 5 unanswerable questions.

## Overall results (25 questions)

| Metric | Value |
|---|---|
| Faithfulness | 0.495 |
| Composite | 0.485 |
| Answer relevance | 0.702 |
| Completeness | 0.704 |
| Critic pass rate | 84% |
| Avg retries | 0.6 |
| Errors (max retries exhausted) | 4/25 |
| Avg cost/query | $0.0196 |
| Total cost | $0.489 |

**Generalization signal: these numbers are comparable to, or slightly better than, the
in-domain evaluation on the AC-RAG paper itself** (faithfulness 0.463, composite 0.466
in `ABLATION_RESULTS.md`'s `full` row). The system is not obviously overfit to the
one document it was developed against — a genuinely useful, non-obvious result to
report, since it wasn't guaranteed going in.

## Unanswerable-question handling: 3/5 correctly abstained, 2 real failures

SQuAD v2.0's unanswerable questions are adversarially designed to sound plausible
without actually being answerable from the given text — a direct, out-of-domain test
of the "say I don't know rather than hallucinate" claim.

| # | Question | Outcome |
|---|---|---|
| 1 | "What is one example of an instance that the qualitative answer to the traveling salesman fails to answer?" | **Hallucinated** — fabricated specific, plausible-sounding details ("a round trip through all sites in Milan whose total length is at most 10 km") that appear nowhere in the source text. |
| 2 | "What is the term for the set of all unconnected graphs related to this decision problem?" | Correctly abstained: "does not contain sufficient information." |
| 3 | "What process classifies problems that can and cannot be solved with approximately unlimited resources?" | Correctly abstained (same phrasing) — though see below, this one also triggered a pipeline "error." |
| 4 | "What can be viewed as a limited collection of instances together with a solution for every instance?" | Correctly abstained (same phrasing) — also triggered a pipeline "error." |
| 5 | "What are two fields of theoretical computer science that closely mirror computational **simplicity** theory?" | **Wrong** — answered as if the question said "**complexity** theory" (the real, adjacent question in the set), missing that the one-word substitution makes it a different, unanswerable question. |

**Two distinct, honest failure modes, not one:**
- **Confident fabrication** (#1) — the generator invented specific numbers/place names
  with no basis in the context. This is the failure mode the whole "no hallucination"
  architecture is meant to prevent, and it still happened once in 5 tries on genuinely
  adversarial input. Worth stating plainly, not minimizing.
- **Topic-similarity confusion** (#5) — dense retrieval matched the semantically similar
  *real* question/answer pair despite the query's one-word substitution changing its
  meaning entirely. This is a known, general limitation of embedding-based retrieval
  (it can't reliably detect that "simplicity" vs "complexity" invalidates the match) —
  a lexical/keyword signal (BM25-style) would likely catch this specific case, same
  future-work direction already noted in `FAILURE_ANALYSIS.md`.

## Bonus finding: the Critic sometimes rejects genuinely-correct abstentions

Questions #3 and #4 above both got the right final answer text ("does not contain
sufficient information") but were ALSO flagged as pipeline errors ("max retries
exhausted"). Reading the trace: the Critic's completeness scoring appears to penalize
an honest "I don't know" as an *incomplete* answer worth retrying, rather than
recognizing it as the correct outcome for a genuinely unanswerable question — wasting
the full retry budget on an answer that was already right.

This is a real, previously undiscovered nuance in the Critic's design: it has no
special case for "the query is legitimately unanswerable from context," so it keeps
pushing for a fuller answer that doesn't exist. **Not fixed here** (found via this
specific out-of-domain test, and fixing it well requires more thought about how the
Critic should distinguish "vague because the generator gave up too early" from
"correctly abstained because the content genuinely isn't there") — documented as a
concrete, scoped future-work item.

## Reproducing this

```bash
python -m evaluation.run_public_benchmark
```
Automatically backs up the real demo index, builds a temporary index from
`data/benchmark_raw/squad_computational_complexity.txt`, runs the 25-question eval,
then restores the original index — safe to re-run without disturbing the demo.

Raw per-query results: `evaluation/results/results_public_benchmark_<timestamp>.json`
(gitignored — regenerate via the command above).
