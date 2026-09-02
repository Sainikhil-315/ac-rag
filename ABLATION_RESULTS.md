# AC-RAG — Ablation Study Results

Gap fix (GAPS_AND_IMPROVEMENTS.md items A.5 and B.9). First time this comparison has
actually run correctly — see the "Fix ablation study never actually applying its config
overrides" commit (2026-08-20): every prior attempt silently ran all 8 configs with
identical real pipeline behavior due to a Python import-binding bug in the settings
patcher. This data is from the corrected mechanism, on the 15-question stratified
subset (`evaluation/test_set_ablation_subset.json`), against the AC-RAG paper index
with table extraction enabled.

## Full comparison table

| Config | Faithfulness | Completeness | ROUGE-L | Composite | Ans.Relevance | Ctx.Util | Critic Pass% | Avg Latency |
|---|---|---|---|---|---|---|---|---|
| baseline_rag (naive) | 0.469 | 0.653 | 0.264 | 0.451 | 0.668 | 0.200 | 0% * | 4.7s |
| **full (AC-RAG)** | 0.463 | 0.707 | 0.246 | 0.466 | 0.663 | 0.253 | 93.3% | 13.6s |
| no_planner | 0.469 | 0.653 | 0.246 | 0.453 | 0.664 | 0.233 | 93.3% | 11.0s |
| no_validator | 0.486 | 0.680 | 0.287 | 0.473 | 0.701 | 0.214 | 100% | 10.7s |
| no_refiner | 0.475 | 0.653 | 0.267 | 0.456 | 0.670 | 0.217 | 93.3% | 12.5s |
| no_critic | 0.477 | 0.693 | 0.246 | 0.463 | 0.666 | 0.233 | 0% * | 9.1s |
| low_k (fixed k=4) | 0.451 | 0.653 | 0.262 | 0.467 | 0.671 | 0.300 | 93.3% | 13.8s |
| high_k (fixed k=12) | 0.506 | 0.613 | 0.254 | 0.428 | 0.667 | 0.100 | 93.3% | 14.0s |

\* `no_critic` and `baseline_rag` both disable the critic entirely — `critic_passed` is
never set (stays `None`), so 0% here means "never evaluated," not "always failed." Don't
read this column as a quality signal for those two rows.

Raw data: `evaluation/results/summary_<config>_<timestamp>.json` (per-config aggregates)
and `results_<config>_<timestamp>.json` (per-query detail, gitignored — regenerate with
`python -m evaluation.ablation --test-set evaluation/test_set_ablation_subset.json`).

## Statistical significance (paired tests, n=15, scipy 1.18)

Ran paired t-tests (and Wilcoxon signed-rank as a non-parametric cross-check) on the
per-query metric values between configs, since all configs share the same 15 questions.

| Comparison | Metric | Mean A | Mean B | Paired t p-value | Wilcoxon p-value |
|---|---|---|---|---|---|
| full vs baseline_rag | faithfulness | 0.463 | 0.469 | 0.434 | 0.776 |
| full vs baseline_rag | composite | 0.466 | 0.451 | 0.130 | 0.133 |
| full vs baseline_rag | answer_relevance | 0.663 | 0.668 | 0.468 | 0.799 |
| full vs no_validator | faithfulness | 0.463 | 0.486 | 0.267 | — |
| full vs no_validator | composite | 0.466 | 0.473 | 0.778 | — |

**None of these reach significance at p < 0.05.** This is itself an honest, reportable
finding — not a null result to hide.

## Interpretation — say this plainly in the report, don't spin it

1. **On these automated point-estimate metrics, `full` does not show a statistically
   significant improvement over `baseline_rag` at n=15.** The faithfulness/composite/
   answer-relevance deltas are all within noise. A jury member comparing the numbers
   directly will see this — better to state it first than have it caught.

2. **What DOES clearly differ is pipeline *behavior*, not the automated quality score:**
   critic pass rate (93.3% for full vs n/a for baseline since it has no critic), retry
   counts, and the qualitative self-correction mechanism (validator/critic catching and
   retrying on genuinely bad retrievals — see FAILURE_ANALYSIS.md for concrete cases).
   The honest framing: **the benefit is reliability and process, not necessarily raw
   automated faithfulness score at this sample size.**

3. **`no_validator` scoring as well as or better than `full`** on faithfulness (0.486
   vs 0.463) and reaching 100% critic pass rate is a genuine, slightly uncomfortable
   finding — but it is NOT statistically significant either (p=0.267). Report it as
   "no significant difference detected between full and no_validator at n=15" rather
   than "the validator hurts" or "the validator helps." Both framings would overclaim
   from this sample size.

4. **`high_k`'s low context-utilisation score (0.100) is a metric-definition artifact,
   not necessarily worse retrieval quality** — `context_utilisation` is
   `cited_sources / total_retrieved`, so retrieving more passages (k=12) mechanically
   lowers the ratio even if the same number of useful passages get cited. Worth a
   one-line caveat if this column comes up.

5. **What this data honestly supports for the report:** the architecture is sound and
   the self-correction mechanism visibly changes behavior (retries, pass rates,
   documented failure recovery in FAILURE_ANALYSIS.md), but the current n=15
   single-document sample is too small to make a statistically defensible claim that
   automated faithfulness/composite scores improve. **The credible claim is about
   reliability and process, not a point-estimate score win** — and stating the
   significance test result directly is more convincing to a technical jury than a
   bare comparison table would be, because it shows methodological rigor rather than
   cherry-picked numbers.

## Recommended next step (future work, not done here)

A larger sample (the original 40-46 question test set, or ideally 100+, run once rate
limits allow) would have more power to detect real effects if they exist. This n=15
subset was chosen specifically to stay within observed OpenAI rate limits during this
session (see GAPS_AND_IMPROVEMENTS.md item A.1) — it is a deliberately small, documented
compromise, not the final word on the system's performance.
