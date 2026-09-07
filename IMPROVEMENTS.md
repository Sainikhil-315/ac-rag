> **Purpose:** This is the single source of truth for the next AC-RAG implementation pass. Give this page to the coding agent and implement the repository against it in one coordinated pass. Do not treat it as a loose TODO list.
> 

## 1. Execution Rules

1. Inspect the current repository before editing.	
2. Implement all specified changes before doing the final test cycle.
3. Do not invent a different architecture, research direction, API, or data contract when this specification already defines one.
4. Do not rewrite unrelated modules.
5. Preserve working ingestion, vector-store, API, frontend, configuration, evaluation, and CLI behavior unless explicitly changed below.
6. If an existing implementation conflicts with this specification, resolve the conflict according to the target architecture and acceptance criteria, and document the conflict in the implementation report.
7. Never fabricate test results, benchmark scores, latency, cost, significance, or successful retrieval.
8. Do not claim the new system improves metrics until the experiments actually demonstrate it.
9. After all code changes: run the complete pytest suite, fix failures, rerun it, then run integration/evaluation/ablation experiments.
10. Finish with a machine-readable and human-readable implementation/results report.

## 2. Current Project Baseline

Repository: `Sainikhil-315/ac-rag`

Current major directories:

```
backend/
config/
data/
evaluation/
frontend/
ingestion/
pipeline/
tests/
utils/
vectorstore/
```

Current graph:

```
START
  ↓
entry_router
  ├── unknown → direct_responder → END
  └── rag → query_analyzer
                ↓
          retrieval_planner
                ↓
             retriever
                ↓
             validator
            ┌───────┴────────┐
       insufficient        sufficient
            ↓                  ↓
       retrieval retry   context_refiner
                               ↓
                           generator
                               ↓
                             critic
                         ┌─────┴─────┐
                       pass         fail
                        ↓             ↓
                      END          retry
```

Current system already has query analysis, complexity estimation, decomposition, retrieval planning, dense MMR retrieval, multi-query retrieval, modality filtering, validation, context refinement, generation, critique, retry routing, document ingestion, table/figure extraction, FAISS, embeddings, tests, evaluation, and ablation infrastructure.

Current known research limitations:

- Retrieval adaptation is mainly parameter adaptation (`k`, `fetch_k`, MMR, modality), not explicit evidence planning.
- Validation is primarily passage relevance/similarity, not claim-level evidence support.
- Retry widens retrieval rather than targeting missing information.
- Dense retrieval can miss exact configuration values, numbers, identifiers, and procedural phrases.
- Generator self-confidence is not sufficient as an evidence confidence mechanism.
- Unsupported answers/abstentions need explicit control.
- Existing benchmark/ablation results do **not** justify a claim that the current system significantly beats baseline.

Existing corrected baseline observations must remain available for comparison. Do not overwrite them with new-system numbers.

## 3. Research Direction

### Working direction

**Evidence-Driven Adaptive Retrieval-Augmented Generation for Reliable Document Question Answering**

### Research question

Can a RAG system dynamically determine what evidence is required to answer a question, retrieve evidence targeted to those requirements, verify generated claims against that evidence, and abstain when sufficient support cannot be established?

### Hypothesis

Explicit evidence requirements + requirement-level coverage + targeted retrieval + claim/evidence verification + evidence-aware abstention should reduce unsupported answers and improve reliability compared with fixed-depth baseline RAG and purely parameter-adaptive retrieval.

This is a hypothesis, not a result.

### Intended contribution

The contribution is the unified control loop:

```
Question
 → evidence requirements
 → retrieval strategy
 → hybrid evidence search
 → requirement coverage
 → targeted retrieval for gaps
 → generation from verified evidence
 → claim extraction
 → claim/evidence verification
 → contradiction handling
 → repair/retrieve or abstain
 → final answer + trace
```

Do not describe basic RAG, dense retrieval, BM25, MMR, query decomposition, self-reflection, reranking, or multimodal retrieval as individually novel.

## 4. Target Architecture

```
USER QUERY
  ↓
Entry Router
  ↓
Query Understanding
  ↓
Evidence Requirement Planner
  ↓
Retrieval Controller
  ↓
Dense + BM25 + Table/Figure Retrieval
  ↓
Fusion + Reranking
  ↓
Evidence Coverage Engine
  ├── sufficient → Context Refinement → Generation
  └── insufficient → Targeted Retrieval Loop → Coverage Engine
                                      ↓
                              max attempts → Abstain

Generation
  ↓
Claim Extraction
  ↓
Claim ↔ Evidence Verification
  ├── supported → confidence → final answer
  ├── unsupported → targeted retrieval / repair
  └── contradicted → contradiction handling / abstain
```

## 5. Component Migration Matrix

| Component | Decision | Target |
| --- | --- | --- |
| Entry Router | MODIFY | Initial routing/answerability prior; never the final evidence decision |
| Query Analyzer | MODIFY | Intent, complexity, rewrite, decomposition, evidence needs |
| Retrieval Planner | REBUILD | Evidence-aware retrieval controller |
| Dense Retriever | KEEP/MODIFY | Dense candidate retrieval |
| MMR | KEEP | Diversity control |
| Multi-query | MODIFY | Requirement-specific query generation |
| Validator | REBUILD | Evidence coverage/support engine |
| Context Refiner | KEEP/MODIFY | Refine approved evidence |
| Generator | MODIFY | Generate only from verified/approved evidence |
| Critic | REBUILD | Claim/evidence verification and control |
| Retry | MODIFY | Evidence-driven retry reasons and targeted queries |
| Direct Responder | KEEP/MODIFY | Graceful unknown/abstention response |
| Citations | MODIFY | Trace requirements → evidence → claims → citations |
| Table extraction | KEEP | Structured table evidence |
| Figure extraction | KEEP/MODIFY | Structured figure evidence |
| FAISS | KEEP | Dense backend |
| BM25/lexical | NEW | Exact-term/value retrieval |
| Evidence Requirement Planner | NEW | R1...Rn information requirements |
| Evidence Coverage Engine | NEW | Requirement support states/scores |
| Targeted Retrieval Loop | NEW | Search specifically for unresolved requirements |
| Claim Extraction | NEW | Atomic answer claims |
| Claim-Evidence Verification | NEW | Support/contradiction assessment |
| Contradiction Detection | NEW | Conflicting evidence/claims |
| Evidence-derived Confidence | NEW | Confidence from coverage and verification |
| Evidence Trace | NEW | Auditable provenance graph/record |

## 6. Shared State Contract

Update `pipeline/state.py` without creating duplicate state definitions.

Required concepts:

```python
query
conversation_history
route
rewritten_query
intent
complexity_score
decomposed_queries

evidence_requirements
retrieval_plan
retrieved_docs
retrieved_evidence
evidence_coverage
missing_requirements
targeted_queries

scored_docs
validation_passed
refined_context

answer
answer_with_attribution
claims
claim_verifications
contradictions

is_answerable
evidence_confidence

retry_count
retry_reason
retrieval_attempts
reflection_attempts
stage_logs
error
token_usage
evidence_trace
```

Use structured Pydantic models for new contracts where practical.

### EvidenceRequirement

Minimum fields:

```
id
requirement
requirement_type
priority
keywords
expected_modality
status
support_score
supporting_evidence_ids
```

Requirement types may include:

- fact
- value
- definition
- procedure
- comparison
- relationship
- summary
- numeric
- table
- figure

### EvidenceItem

Minimum fields:

```
id
content
source
page
section
chunk_id
modality
retrieval_method
retrieval_score
rerank_score
```

### EvidenceCoverage

Minimum fields:

```
requirement_id
status
score
supporting_evidence_ids
missing_reason
```

Allowed status values:

```
SUPPORTED
PARTIAL
MISSING
CONTRADICTED
NOT_APPLICABLE
```

### Claim

Minimum fields:

```
id
text
claim_type
requirement_ids
citation_ids
```

### ClaimVerification

Minimum fields:

```
claim_id
status
support_score
contradiction_score
evidence_ids
reason
```

Allowed status values:

```
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
CONTRADICTED
```

### EvidenceTrace

Represent:

```
requirement → evidence → claim → citation → verification
```

The trace must be serializable for evaluation/debugging.

## 7. File-by-File Implementation Plan

### `pipeline/state.py`

- Add the new evidence/claim/verification models.
- Add new shared state fields.
- Keep old fields required by existing callers until migration is complete.
- Document field ownership.
- Do not store arbitrary untyped dictionaries where a stable contract is possible.

**Acceptance:** graph nodes can exchange requirements, evidence coverage, claims, verification, and trace without hidden globals.

### `pipeline/nodes/entry_router.py`

Keep the cheap initial router but change its semantics:

- Treat similarity as a routing prior only.
- Do not let router similarity alone decide that a query is answerable.
- Preserve unknown/direct response behavior.
- Log router score and decision separately from evidence sufficiency.

**Acceptance:** a semantically low-similarity but lexically exact query can still reach RAG; final answerability is decided by evidence coverage.

### `pipeline/nodes/query_analyzer.py`

Extend the structured analysis output with:

```
evidence_requirements: List[str]
```

Requirements must be:

- short
- independently verifiable
- information needs, not answers
- specific enough to drive retrieval
- maximum 6 by default

Examples:

```
Question: What embedding model and chunk size does the system use?
Requirements:
R1 = embedding model
R2 = chunk size
```

For comparisons:

```
R1 = characteristic of system A
R2 = corresponding characteristic of system B
R3 = comparison criterion
```

Keep intent, complexity, rewrite, decomposition, and conversation history support.

### `pipeline/nodes/retrieval_planner.py`

Rebuild as the Retrieval Controller.

Inputs:

- query
- rewritten query
- intent
- complexity
- evidence requirements
- current coverage
- missing requirements
- retrieval history

Outputs:

```
retrieval strategy
query list
k/fetch_k
lexical vs dense weights
MMR lambda
modality priorities
reranking requirement
retrieval depth
stop/continue recommendation
```

The controller must answer:

1. What should be searched?
2. Which retrieval method is appropriate?
3. Which modality is likely relevant?
4. Is another retrieval round justified?
5. What exact missing requirement should the next search target?

Do not simply increase `k` on every retry.

### `pipeline/nodes/retriever.py`

Keep dense FAISS/MMR infrastructure.

Add:

- lexical/BM25 retrieval
- query-specific retrieval
- exact-term/value matching
- requirement-specific search
- metadata-preserving result normalization
- duplicate removal across retrieval methods

Dense retrieval remains best for semantic concepts.

Lexical retrieval must help with:

- identifiers
- configuration names
- exact values
- numbers
- abbreviations
- API names
- section names
- procedural phrases

### NEW `pipeline/retrieval/bm25_retriever.py`

Implement a repository-compatible lexical retriever.

Requirements:

- deterministic
- metadata preserving
- same normalized result interface as dense retrieval
- no network dependency
- usable in tests

If a suitable existing dependency is already installed, use it. Otherwise add a minimal dependency only when necessary.

### NEW `pipeline/retrieval/hybrid.py`

Implement dense + lexical fusion.

Preferred approach:

```
Dense results
+
BM25 results
↓
normalize/rank
↓
Reciprocal Rank Fusion or equivalent deterministic fusion
↓
optional rerank
```

Do not hard-code a magic fusion weight without exposing it in configuration.

### `pipeline/nodes/validator.py`

Rebuild as the Evidence Coverage Engine.

Responsibilities:

- map retrieved evidence to requirements
- assess support strength
- detect missing requirements
- detect contradictory evidence
- assign requirement statuses
- calculate coverage

Coverage should be requirement-weighted, not merely document-count based.

A basic coverage formula may be:

```
coverage = sum(priority_i * support_i) / sum(priority_i)
```

where `support_i` is normalized to [0,1].

Expose thresholds in configuration.

### NEW `pipeline/nodes/targeted_retrieval.py`

Implement the targeted retrieval controller/loop.

For each unresolved requirement:

1. Build a targeted query.
2. Select dense, lexical, table, or figure retrieval.
3. Retrieve candidates.
4. Merge/deduplicate.
5. Recompute coverage.
6. Stop if sufficient.

Stop when:

- all critical requirements are sufficiently supported, OR
- maximum retrieval rounds are reached, OR
- repeated attempts produce no meaningful coverage improvement.

Meaningful improvement should be configurable.

### `pipeline/nodes/context_refiner.py`

Keep existing reranking/deduplication/compression capabilities.

Change input semantics:

- prioritize evidence linked to required claims
- preserve source metadata
- do not compress away evidence needed for exact values
- preserve tables/figures when they are the actual support

### `pipeline/nodes/generator.py`

Modify generation so that it receives:

- question
- evidence requirements
- approved evidence
- coverage state
- contradiction state

Rules:

- Answer only from approved evidence.
- Do not fill missing requirements from model knowledge.
- Cite claims with source IDs.
- If a required fact is missing, explicitly say it is not established by the available evidence.
- Do not treat high semantic similarity as proof.

Remove or de-emphasize self-assessed confidence as the main confidence mechanism.

### NEW `pipeline/nodes/claim_extractor.py`

Extract atomic factual claims from the draft answer.

Requirements:

- one independently verifiable proposition per claim
- preserve numbers and units
- preserve qualifiers
- preserve citations if already attached
- map claims back to evidence requirements

### NEW `pipeline/nodes/claim_verifier.py`

For every claim:

1. Find supporting evidence.
2. Determine whether evidence entails/supports the claim.
3. Detect contradiction.
4. Return structured verification.

The verifier must distinguish:

```
supported
partial
unsupported
contradicted
```

Do not use raw similarity as the sole verification criterion.

### NEW `pipeline/nodes/contradiction_detector.py`

Detect conflicts between:

- evidence passages
- claims
- multiple documents
- different versions where metadata suggests precedence

If contradictory evidence exists, preserve both sides in the trace and do not silently choose one.

Where document version/authority metadata provides a valid resolution rule, use it explicitly and record the reason.

### `pipeline/nodes/critic.py`

Rebuild from generic answer scoring into a verification/control node.

Inputs:

- claims
- claim verifications
- requirement coverage
- contradictions
- draft answer

Outputs:

```
pass
repair
retrieve_more
abstain
```

Decision logic:

- supported claims + sufficient critical coverage → pass
- missing evidence for important claims → targeted retrieval
- unsupported claim → repair or retrieve
- contradiction → resolve with evidence or abstain
- exhausted budget → abstain/partial answer

### `pipeline/router.py`

Update routing to support explicit control states.

Required conceptual routes:

```
entry → analyze
analyze → plan
plan → retrieve
retrieve → coverage
coverage → retrieve_more OR refine
refine → generate
generate → verify
verify → final OR retrieve_more OR repair OR abstain
```

Maintain graceful maximum-retry behavior.

### `pipeline/graph.py`

Rewire the graph around the new evidence lifecycle.

Do not create infinite loops.

Maintain explicit counters:

- retrieval attempts
- generation/repair attempts
- total retry count

Maximum limits must be configuration-driven.

Every terminal path must produce a valid final state.

## 8. Prompts

All prompts must be centralized in the appropriate prompt/module locations rather than duplicated across nodes.

### Evidence Requirement Planner prompt

```
You are an evidence requirement planner.
Given a user question, identify the minimum independently verifiable information needed to answer it correctly.
Return short requirements, not answers.
Each requirement must be specific enough to retrieve and verify.
For numerical/configuration questions preserve exact entities and values as retrieval targets.
For comparisons create corresponding requirements for each side.
Maximum 6 requirements.
```

### Evidence Matching prompt

```
Given an evidence requirement and candidate passages, determine whether each passage supports the requirement.
Do not judge based only on topic similarity.
Look for explicit factual support, exact values, definitions, procedures, relationships, or table/figure evidence.
Return support score, status, evidence IDs, and a short reason.
```

### Claim Extraction prompt

```
Convert the draft answer into atomic factual claims.
Preserve exact numbers, units, qualifiers, conditions, and distinctions.
Do not add facts.
Map each claim to one or more evidence requirements.
```

### Claim Verification prompt

```
Verify each claim only against the supplied evidence.
Classify each claim as SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or CONTRADICTED.
A passage being topically similar is not sufficient.
Exact values and procedural claims require explicit support.
Return evidence IDs and a concise reason.
```

### Contradiction prompt

```
Compare the candidate claims/evidence for factual conflict.
Do not call two statements contradictory merely because they discuss different conditions, versions, or scopes.
If they genuinely conflict, identify the conflicting propositions and the supporting evidence IDs.
```

### Generation prompt

```
Answer the user's question using only the approved evidence.
Every factual claim must be traceable to evidence.
Do not invent missing information.
If critical evidence is unavailable, state that the source material does not establish the answer.
Use citations tied to evidence IDs.
```

## 9. Retrieval Strategy

Use a hybrid strategy:

```
query
 ├── semantic rewrite → dense retrieval
 ├── exact terms → BM25
 ├── requirement-specific terms → targeted retrieval
 └── modality signal → table/figure retrieval
```

Use fusion before final reranking.

Preserve:

- source file
- page
- section
- chunk ID
- modality
- retrieval method
- raw score
- fused/rerank score

Do not discard provenance during fusion.

## 10. Evidence Lifecycle

Every query must conceptually move through:

```
Requirement created
    ↓
Evidence candidate retrieved
    ↓
Evidence assessed
    ↓
Requirement coverage updated
    ↓
Missing requirements identified
    ↓
Targeted retrieval if necessary
    ↓
Requirement satisfied / unresolved
    ↓
Generation
    ↓
Claims extracted
    ↓
Claims verified
    ↓
Final answer or abstention
```

## 11. Abstention Policy

The system must prefer an explicit evidence-grounded abstention over a fabricated answer.

Abstain when:

- a critical requirement remains missing after the retrieval budget is exhausted
- a key claim is unsupported
- evidence is contradictory and cannot be resolved
- retrieved evidence is insufficient to distinguish competing answers

Possible response pattern:

> “I could not establish this from the available documents.”
> 

For partially answerable questions, provide only supported portions and clearly mark unresolved portions.

Do not fabricate a confidence number just to avoid abstention.

## 12. Evidence-Derived Confidence

Confidence must be based on measurable evidence signals.

Candidate signals:

- weighted requirement coverage
- critical-requirement coverage
- claim support ratio
- contradiction penalty
- evidence quality/rerank score
- retrieval stability across rounds

A simple configurable score may be:

```
confidence =
  w1 * requirement_coverage
+ w2 * claim_support_ratio
+ w3 * evidence_quality
- w4 * contradiction_ratio
```

Weights must be configurable and documented.

Do not call this calibrated probability unless calibration is actually evaluated.

## 13. Citation and Evidence Trace

Final answers must retain source attribution.

Trace example:

```
R1: embedding model
  ↓
E12: config/settings.py, relevant section
  ↓
C1: “The system uses ...”
  ↓
verified: SUPPORTED
  ↓
answer citation [E12]
```

Expose enough metadata for debugging and paper analysis.

## 14. Ingestion and Structured Evidence

Keep existing ingestion support:

- PDF
- DOCX
- TXT
- HTML
- Markdown

Keep the current chunking approach unless a change is necessary for evidence retrieval.

Preserve structured metadata.

Table extraction:

- keep pdfplumber structural extraction
- preserve table identity
- preserve row/column context where possible
- make table evidence retrievable separately

Figure extraction:

- preserve figure identity
- preserve page/source
- preserve extracted textual description
- allow figure-specific retrieval when the requirement expects a figure

Do not break existing ingestion tests.

## 15. Configuration

Update `config/settings.py` with explicit configuration for:

```
BM25 enabled
hybrid fusion method
fusion weights
requirement max count
coverage threshold
critical coverage threshold
claim support threshold
contradiction threshold
max retrieval rounds
max repair rounds
minimum coverage improvement
abstention enabled
confidence weights
```

Keep existing configuration names where they remain meaningful.

Avoid unexplained magic numbers.

## 16. API / Frontend Compatibility

Do not break existing frontend/API response contracts unnecessarily.

If new fields are exposed, prefer additive fields such as:

```
evidence_confidence
evidence_trace
citations
verification_summary
abstained
```

Existing answer fields must continue to work.

Do not expose internal chain-of-thought. Expose structured verification metadata and concise reasons only.

## 17. Tests — RUN AFTER ALL CODE CHANGES

Do not stop the implementation after each individual file merely to run isolated tests. After the complete code migration, run the full test cycle.

### Unit tests

Add/update tests for:

- evidence requirement parsing
- evidence requirement limits
- requirement types
- state serialization
- BM25 retrieval
- hybrid fusion
- exact-term retrieval
- evidence matching
- coverage calculation
- missing-requirement detection
- targeted query generation
- claim extraction
- claim verification
- contradiction detection
- evidence confidence
- abstention decisions
- citation/evidence trace

### Regression tests

Existing tests must continue to cover:

- query analyzer
- router
- retriever
- validator/coverage
- critic
- graph retry behavior
- figure extractor
- retrieval metrics

### Integration tests

At minimum test these flows:

1. Simple answerable factual question.
2. Multi-hop/comparative question.
3. Exact configuration/value question.
4. Table-based question.
5. Figure-based question.
6. Unsupported question.
7. Partially supported question.
8. Contradictory evidence question.
9. Query requiring a targeted retrieval retry.
10. Maximum retrieval budget exhaustion.

### Graph invariants

Test that:

- every path terminates
- retry counters cannot become infinite
- abstention produces a valid response
- missing evidence cannot silently become a generated fact
- critic/verification failures route correctly
- final state remains serializable

## 18. Required Final Test Commands

Use the repository's actual environment/entry points. At minimum execute:

```bash
pytest -q
```

Then, if available:

```bash
pytest -q tests
```

Run targeted tests for any failed subsystem, fix them, then rerun the full suite.

Do not report “tests pass” unless the final full run actually passes.

## 19. Evaluation Design

Do not compare only against the old full AC-RAG system.

At minimum compare:

1. Vanilla/fixed-k RAG baseline.
2. Existing AC-RAG baseline.
3. New evidence-driven AC-RAG.

Where feasible include additional controlled variants.

### Required metrics

Answer quality:

- faithfulness
- completeness
- answer relevance
- exact match/F1/ROUGE where appropriate

Evidence quality:

- requirement coverage
- critical requirement coverage
- claim support ratio
- contradiction rate
- unsupported claim rate
- citation support rate

Reliability:

- correct abstention rate
- false abstention rate
- unsupported answer rate

Efficiency:

- latency
- retrieval rounds
- retrieved passages
- token usage
- estimated cost/query

### Important

Do not replace objective metrics with only LLM-judge scores.

If LLM evaluation is used, document:

- model
- prompt
- temperature
- aggregation
- known limitations

## 20. Evaluation Dataset

Preserve the existing evaluation infrastructure and dataset where useful, but expand the test set around known failure modes.

Required categories:

- direct factual
- analytical
- comparative
- summarization
- numeric/configuration
- table
- figure
- multi-hop
- unanswerable
- partially answerable
- contradiction/version conflict

Include enough examples per category to make failure analysis meaningful.

Do not silently alter benchmark questions between baseline and proposed-system runs.

## 21. Ablation Study

Run controlled ablations for the new contribution.

Minimum:

```
full evidence-driven system
- evidence requirement planning
- hybrid retrieval
- evidence coverage engine
- targeted retrieval loop
- claim verification
- contradiction detection
- evidence-aware abstention
- evidence-derived confidence
```

Also retain useful existing ablations such as:

- no planner
- no validator/coverage
- no refiner
- no critic/verification
- low-k
- high-k

Each ablation must disable only the intended component while keeping all else fixed.

## 22. Failure Analysis

For every important failure capture:

```
question
expected behavior
actual behavior
requirements
retrieval results
coverage state
claims
verification state
failure category
root cause
possible fix
```

Required categories:

- retrieval miss
- lexical miss
- semantic miss
- requirement-planning error
- evidence-matching error
- claim-extraction error
- verification error
- contradiction handling error
- abstention error
- generation error
- ingestion/metadata error

## 23. Experiment Reproducibility

Create or update scripts so that a fresh run can reproduce:

```
baseline
proposed system
ablations
metrics
failure analysis
latency/cost summary
```

Save results as machine-readable files such as JSON/JSONL/CSV plus a human-readable Markdown summary.

Every result file must include:

- experiment name
- configuration
- model
- dataset/version
- timestamp
- number of examples
- metrics
- failures if available

## 24. Required Output Artifacts

After implementation/evaluation, maintain:

```
IMPLEMENTATION_REPORT.md
TEST_RESULTS.md
EVALUATION_RESULTS.md
ABLATION_RESULTS.md
FAILURE_ANALYSIS.md
```

Do not overwrite historical baseline results without preserving them.

## 25. Acceptance Criteria

The implementation is complete only when all are true:

- [ ]  Evidence requirements are generated before retrieval.
- [ ]  Requirements are stored in shared state.
- [ ]  Dense retrieval still works.
- [ ]  BM25/lexical retrieval works.
- [ ]  Hybrid fusion works.
- [ ]  Exact-value queries can use lexical evidence.
- [ ]  Evidence coverage is requirement-level.
- [ ]  Missing requirements are explicitly identified.
- [ ]  Targeted retrieval searches for missing requirements.
- [ ]  Retrieval loops have hard termination limits.
- [ ]  Context refinement preserves required evidence.
- [ ]  Generation is restricted to approved evidence.
- [ ]  Claims are extracted from generated answers.
- [ ]  Claims are verified against evidence.
- [ ]  Contradictions are represented explicitly.
- [ ]  Unsupported claims trigger repair/retrieval or abstention.
- [ ]  Unanswerable questions can produce grounded abstentions.
- [ ]  Confidence is evidence-derived.
- [ ]  Citations map back to evidence.
- [ ]  Evidence trace is serializable.
- [ ]  Existing ingestion remains functional.
- [ ]  Existing API/frontend behavior remains compatible unless explicitly changed.
- [ ]  Full pytest passes.
- [ ]  Integration tests pass.
- [ ]  Baseline/proposed experiments execute successfully.
- [ ]  Ablations execute successfully.
- [ ]  Failure analysis is generated.
- [ ]  No result is fabricated.

## 26. One-Shot Coding-Agent Workflow

The coding agent must follow this exact workflow:

```
READ REPOSITORY
    ↓
READ THIS SPECIFICATION
    ↓
MAP EXISTING FILES TO SPEC
    ↓
IMPLEMENT ALL REQUIRED CODE CHANGES
    ↓
IMPLEMENT NEW MODULES
    ↓
UPDATE CONFIG
    ↓
UPDATE TESTS
    ↓
RUN FULL PYTEST
    ↓
FIX ALL FAILURES
    ↓
RUN FULL PYTEST AGAIN
    ↓
RUN INTEGRATION TESTS
    ↓
RUN BASELINE
    ↓
RUN NEW SYSTEM
    ↓
RUN ABLATIONS
    ↓
RUN FAILURE ANALYSIS
    ↓
WRITE RESULTS ARTIFACTS
    ↓
FINAL IMPLEMENTATION REPORT
```

Do not stop after the first successful code edit and ask for permission to continue.

## 27. Coding-Agent Safeguards

The agent must not:

- invent unrelated features
- redesign the frontend without need
- replace LangGraph without explicit reason
- replace FAISS without explicit reason
- remove working ingestion
- remove existing evaluation history
- hard-code benchmark outcomes
- delete tests just because they fail
- weaken assertions merely to make tests pass
- silently disable verification
- silently fall back to unsupported model knowledge
- create infinite retrieval loops
- expose private chain-of-thought

If a test exposes a real design conflict, fix the implementation rather than weakening the test unless the test itself contradicts this specification.

## 28. Research Integrity Rules

The final paper must distinguish:

- what was implemented
- what was measured
- what was hypothesized
- what was statistically tested
- what failed
- what remains a limitation

Never reuse old claims that are contradicted by corrected experiments.

Never state that the proposed system is statistically superior unless the final experiment supports that statement.

## 29. Final Verification Checklist

Before declaring completion, answer all of these from actual repository state:

1. What files were changed?
2. What new files were added?
3. What old components were replaced?
4. What state contracts changed?
5. What graph routes changed?
6. How does the system decide what evidence it needs?
7. How does it retrieve missing evidence?
8. How does it know evidence is sufficient?
9. How does it verify generated claims?
10. How does it detect contradiction?
11. When does it abstain?
12. How is confidence calculated?
13. How are citations traced?
14. What tests passed?
15. What tests failed, if any?
16. What are baseline results?
17. What are proposed-system results?
18. What do the ablations show?
19. What failure cases remain?
20. Which research claims are actually supported?