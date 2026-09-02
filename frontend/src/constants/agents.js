export const AGENTS = [
  {
    id: 'query_understanding',
    number: '01',
    icon: '🧠',
    title: 'Query Understanding',
    subtitle: 'Understands intent · Refines vague questions',
    color: '#6366F1',
    lightBg: '#EEF2FF',
    input: 'Raw user query (string)',
    output: 'rewritten_query, intent, complexity_score, decomposed_queries',
    description:
      'Understands the intent behind the user query and refines vague or poorly worded questions so the vector search gets better results. Classifies intent into factual, analytical, comparative, or summarization. Scores complexity 0–1 and splits complex queries into focused sub-queries.',
    example: {
      in:  '"wat is acrag accurcy"',
      out: 'rewritten: "What is the accuracy of AC-RAG?"\nintent: factual\ncomplexity: 0.2\nsub_queries: []',
    },
  },
  {
    id: 'retrieval_planning',
    number: '02',
    icon: '📋',
    title: 'Retrieval Planning',
    subtitle: 'Decides how much to retrieve · Selects sources · Chooses content type',
    color: '#8B5CF6',
    lightBg: '#F5F3FF',
    input: 'intent, complexity_score, rewritten_query',
    output: 'retrieval_plan { k, fetch_k, lambda_mult, modality_filter, use_multi_query, depth }, retrieved_docs',
    description:
      'Decides how much to retrieve and selects the appropriate sources and content type from the Multi-Modal Knowledge Base. Dynamically sets retrieval parameters (4–12 passages) based on complexity, modality filter (text / table / figure), and whether to use multi-query retrieval for complex questions. Then executes MMR search against the vector store.',
    example: {
      in:  'intent: analytical\ncomplexity: 0.75',
      out: 'k: 10\nmodality: all\nuse_multi_query: true\ndepth: deep\n→ 10 passages retrieved',
    },
  },
  {
    id: 'evidence_validation',
    number: '03',
    icon: '✅',
    title: 'Evidence Validation',
    subtitle: 'Checks relevance · Checks sufficiency · Removes contradictions',
    color: '#F59E0B',
    lightBg: '#FFFBEB',
    input: 'retrieved_docs, rewritten_query',
    output: 'scored_docs [ { ...doc, score: float } ], validation_passed',
    description:
      'Checks relevance and sufficiency of every retrieved passage. Passages scoring below threshold are discarded (removing irrelevant or contradictory evidence). If fewer than 2 passages survive, the pipeline retries retrieval with adjusted parameters (up to 3 times).',
    example: {
      in:  '4 passages',
      out: 'chunk_042 → 0.81 ✓\nchunk_019 → 0.74 ✓\nchunk_031 → 0.41 ✗ discarded\nchunk_007 → 0.29 ✗ discarded\nvalidation_passed: true',
    },
  },
  {
    id: 'context_refinement',
    number: '04',
    icon: '✂️',
    title: 'Context Refinement',
    subtitle: 'Remove duplicates · Order logically · Compress context',
    color: '#10B981',
    lightBg: '#ECFDF5',
    input: 'scored_docs',
    output: 'refined_context (string — ordered, deduplicated, compressed passages)',
    description:
      'Removes duplicate passages, orders the remaining evidence logically by relevance score, and compresses verbose passages to fit within the LLM context window. Assembles the final structured context window that gets passed to the LLM for answer generation.',
    example: {
      in:  '2 scored passages',
      out: '"[chunk_042, p.6] The faithfulness score...\n[chunk_019, p.3] AC-RAG achieved..."',
    },
  },
  {
    id: 'answer_generation',
    number: '05',
    icon: '💡',
    title: 'LLM',
    subtitle: 'Grounded · Attributed · Cited',
    color: '#3B82F6',
    lightBg: '#EFF6FF',
    input: 'refined_context, rewritten_query',
    output: 'answer (string), answer_with_attribution [ { sentence, sources } ]',
    description:
      'The LLM generates a grounded answer strictly from the refined context with explicit source citations. Every factual claim is attributed back to a specific source chunk and page number. If the context lacks sufficient information, the LLM clearly states this rather than hallucinating.',
    example: {
      in:  'context: 2 passages\nquery: "What is AC-RAG accuracy?"',
      out: '"AC-RAG achieves 60% answer accuracy..."\nattribution: sentence 1 → chunk_042 p.6',
    },
  },
  {
    id: 'self_reflection',
    number: '06',
    icon: '🪞',
    title: 'Self-Reflection (Critic)',
    subtitle: 'Reviews answer · Checks support by evidence · Detects missing/over-confidence',
    color: '#EF4444',
    lightBg: '#FEF2F2',
    input: 'answer, refined_context, rewritten_query',
    output: 'critic_scores { faithfulness, completeness, table_accuracy, figure_accuracy, conciseness, overall }, critic_passed',
    description:
      'A separate critic model reviews the generated answer, checks that every claim is supported by evidence, and detects missing information or over-confidence. Scores five dimensions (each 1–5). All must score ≥ 4 to pass ("Answer Strong?"). Format failures trigger regeneration; content failures trigger a full retrieval retry.',
    example: {
      in:  'answer: "AC-RAG achieves 60%..."',
      out: 'faithfulness:    5 ✓\ncompleteness:    4 ✓\ntable_accuracy:  5 ✓\nfigure_accuracy: 4 ✓\nconciseness:     4 ✓\noverall: 4.4 → PASS',
    },
  },
]
