# AC-RAG — Project Report Outline (PRC Review-1)

Structure to follow for report + PPT. Fill each section as content is finalized.

## 1. Title Page
- Project title, team members, guide name, department, college, academic year

## 2. Abstract
- 200-300 words: problem, approach, key result

## 3. Problem Statement
- Why naive RAG fails: no fact-checking, fixed retrieval regardless of query difficulty,
  no "I don't know" capability, hallucination risk

## 4. Objectives
- Adaptive retrieval based on query complexity
- Hallucination reduction via self-reflection critic
- Evidence filtering before generation
- Multi-document support
- Observable / explainable pipeline execution

## 5. Literature Survey
Papers to cover (2-3 lines each: what they did, gap, how this project differs):
- Lewis et al. 2020 — original RAG
- Self-RAG (Asai et al. 2023)
- Corrective RAG / CRAG (Yan et al. 2024)
- Adaptive-RAG
- RAGAS (Es et al. 2023)

## 6. Existing System vs Proposed System
- Table: naive RAG limitation → this project's fix

## 7. System Architecture / Design
- High-level block diagram (thesis/system architecture.png)
- UML diagrams:
  - Use Case diagram
  - Sequence diagram (router → analyzer → planner → retriever → validator → refiner → generator → critic, with retry back-edges)
  - Activity diagram (route + critic pass/fail branches)
  - Class diagram (ACRagState, RetrievalPlan, GeneratedAnswer, CriticEvaluation)
  - Data Flow Diagram (Level 0 + Level 1 — ingestion flow, query flow)

## 8. Technology Stack
- Python, FastAPI, LangGraph, LangChain, FAISS, sentence-transformers, React, Vite, Tailwind, SSE
- One line each: why chosen

## 9. Algorithms / Techniques Adopted
- MMR (Maximum Marginal Relevance) retrieval
- RecursiveCharacterTextSplitter chunking
- Cosine similarity thresholding (routing + validation)
- LLM structured output (Pydantic) per agent
- Self-reflection / critic retry loop
- Complexity-adaptive retrieval scaling

## 10. Implementation
- Module-wise: ingestion, vectorstore, pipeline/nodes, backend, frontend
  (base on PROJECT_SUMMARY.md section 14)

## 11. Formal Problem Formulation
- a* = argmax P(a | q, R(q, D)), subject to Faith(a, R(q,D)) >= tau
- Explain each term (PROJECT_SUMMARY.md section 11)

## 12. Dataset Used
- Test document(s) + evaluation/test_set_sample.json
- Public benchmark plan if added (HotpotQA / QASPER / RAGBench)

## 13. Evaluation Metrics
- Faithfulness, Answer Relevance, Completeness, Context Utilisation, ROUGE-L, RAGAS metrics
  (define each, 1 line)

## 14. Results & Analysis
- Ablation study tables/graphs (7 configs)
- Critic scores, agent timing
- ** Needs real run output before submission — evaluation/results/ currently empty **

## 15. Ablation Study
- Dedicated subsection: what breaks when each agent is disabled

## 16. Screenshots
- Pipeline tab, Playground chat, Results tab

## 17. Conclusion & Future Scope
- Current state, planned improvements (enable validator by default, larger benchmark, more LLM providers tested)

## 18. References
- IEEE/APA format, papers from Section 5
