// Research-tester metadata for the 6 mapped pipeline stages (backend/api.py's
// NODE_TO_AGENT). Icon names are lucide-react component names, resolved in
// PipelineDetailPanel.jsx — kept as strings here so this file stays data-only.
export const AGENTS = [
  {
    id: 'query_understanding',
    icon: 'Brain',
    title: 'Query Understanding',
    subtitle: 'Intent, complexity, decomposed sub-queries, evidence requirements',
  },
  {
    id: 'retrieval_planning',
    icon: 'ListTree',
    title: 'Retrieval Planning + Retrieval',
    subtitle: 'Plan (k, strategy, weights) and the passages actually retrieved',
  },
  {
    id: 'evidence_validation',
    icon: 'ShieldCheck',
    title: 'Evidence Coverage Validation',
    subtitle: 'Per-requirement coverage, missing requirements, targeted retrieval rounds',
  },
  {
    id: 'context_refinement',
    icon: 'Scissors',
    title: 'Context Refinement',
    subtitle: 'Deduplicated, reranked, compressed context passed to the generator',
  },
  {
    id: 'answer_generation',
    icon: 'Sparkles',
    title: 'Answer Generation',
    subtitle: 'Grounded answer with inline source attribution',
  },
  {
    id: 'self_reflection',
    icon: 'ScanEye',
    title: 'Self-Reflection, Claims & Contradictions',
    subtitle: 'Critic scores, claim verification, contradiction detection, confidence, control budget',
  },
]
