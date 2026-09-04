import AgentNode from './AgentNode'
import { AGENTS } from '../../constants/agents'

function Connector({ active }) {
  return (
    <div className="flex flex-col items-center py-0.5">
      <div className={`w-px h-4 transition-colors ${active ? 'bg-accent' : 'bg-border'}`} />
      <svg width="8" height="5" viewBox="0 0 8 5" fill="none">
        <path
          d="M4 5L0 0H8L4 5Z"
          fill={active ? 'var(--c-accent)' : 'var(--c-text-faint)'}
        />
      </svg>
    </div>
  )
}

function KnowledgeBaseNode({ active }) {
  return (
    <div className={`
      rounded-lg px-3 py-2.5 mb-1 transition-all duration-300
      flex flex-col gap-1.5
      ${active ? 'bg-accent-bg border border-accent-bdr' : 'bg-surface-2 border border-border'}
    `}>
      <div className="flex items-center gap-1.5">
        <span className="text-xs">🗄️</span>
        <span className={`text-[10px] font-mono font-semibold ${active ? 'text-accent-txt' : 'text-muted'}`}>
          Multi-Modal Knowledge Base
        </span>
      </div>
      <div className="flex gap-1.5 flex-wrap">
        {[{ label: 'Text', icon: '📝' }, { label: 'Table', icon: '📊' }, { label: 'Figure', icon: '🖼️' }].map(({ label, icon }) => (
          <div
            key={label}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-surface border border-border"
          >
            <span className="text-[8px]">{icon}</span>
            <span className={`text-[8px] font-mono ${active ? 'text-accent-txt' : 'text-muted'}`}>
              {label}
            </span>
            <div className={`w-1 h-1 rounded-full ${active ? 'bg-accent' : 'bg-text-faint'}`} />
          </div>
        ))}
      </div>
    </div>
  )
}

function OutOfScopeNode() {
  return (
    <div className="bg-danger-bg border border-danger rounded-lg px-3 py-2.5 flex items-center gap-2.5 opacity-90">
      <span className="text-sm">🚫</span>
      <div>
        <p className="text-[10px] font-mono font-semibold text-danger">
          Out of Scope
        </p>
        <p className="text-[9px] text-secondary mt-0.5">
          Query bypassed the RAG pipeline
        </p>
      </div>
    </div>
  )
}

const TOTAL_AGENTS = AGENTS.length

export default function PipelineCanvas({ agentStates, agentMeta, selectedAgent, onSelectAgent, lastRoute }) {
  const getState = (id) => agentStates?.[id] || 'idle'
  const getMeta  = (id) => {
    const m = agentMeta?.[id]
    if (!m) return null
    return typeof m === 'string' ? m : (m.summary || null)
  }
  const doneCount = AGENTS.filter(a => getState(a.id) === 'done').length
  const isOutOfScope = lastRoute === 'unknown'
  const anyActive = AGENTS.some(a => getState(a.id) !== 'idle')
  const retrievalState = getState('retrieval_planning')
  const kbActive = retrievalState === 'active' || retrievalState === 'done'

  return (
    <div className="w-72 flex-shrink-0 border-r border-border bg-surface-2 flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3.5 border-b border-border bg-surface flex-shrink-0">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-semibold text-muted uppercase tracking-wider">
            Pipeline
          </span>
          {isOutOfScope ? (
            <span className="text-[9px] font-semibold text-danger bg-danger-bg border border-danger rounded-full px-2 py-0.5">
              Out of Scope
            </span>
          ) : (
            <span className="text-[10px] font-mono text-muted">
              {doneCount}/{TOTAL_AGENTS} done
            </span>
          )}
        </div>
        {/* Progress bar */}
        <div className="mt-2 h-1 bg-bg rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-400"
            style={{
              width: isOutOfScope ? '100%' : `${(doneCount / TOTAL_AGENTS) * 100}%`,
              background: isOutOfScope ? 'var(--c-danger)' : doneCount === TOTAL_AGENTS ? 'var(--c-success)' : 'var(--c-accent)',
            }}
          />
        </div>
      </div>

      {/* Nodes */}
      <div className="flex-1 overflow-y-auto py-2.5 px-2.5 space-y-0.5">
        {/* Entry */}
        <div className="flex items-center gap-2.5 px-2.5 py-1.5 bg-surface border border-border rounded-lg">
          <div className="w-1.5 h-1.5 rounded-full bg-success flex-shrink-0" />
          <span className="text-[11px] text-secondary font-medium">User Query</span>
        </div>

        <Connector active={anyActive || isOutOfScope} />

        {isOutOfScope ? (
          <>
            <OutOfScopeNode />
            <div className="opacity-30 pointer-events-none">
              {AGENTS.map((agent, idx) => (
                <div key={agent.id}>
                  <AgentNode agent={agent} state="idle" selected={false} meta={null} onClick={() => {}} />
                  {agent.id === 'retrieval_planning' && (
                    <>
                      <Connector active={false} />
                      <KnowledgeBaseNode active={false} />
                    </>
                  )}
                  {idx < AGENTS.length - 1 && <Connector active={false} />}
                </div>
              ))}
            </div>
          </>
        ) : (
          AGENTS.map((agent, idx) => {
            const state    = getState(agent.id)
            const meta     = getMeta(agent.id)
            const selected = selectedAgent === agent.id
            const arrowActive = state === 'done'

            return (
              <div key={agent.id}>
                <AgentNode
                  agent={agent}
                  state={state}
                  selected={selected}
                  meta={meta}
                  onClick={() => onSelectAgent(selected ? null : agent.id)}
                />
                {agent.id === 'retrieval_planning' && (
                  <>
                    <Connector active={arrowActive} />
                    <KnowledgeBaseNode active={kbActive} />
                  </>
                )}
                {idx < AGENTS.length - 1 && <Connector active={arrowActive} />}
              </div>
            )
          })
        )}

        <Connector active={doneCount === TOTAL_AGENTS || isOutOfScope} />

        {/* Exit */}
        <div className={`
          flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg transition-all
          ${isOutOfScope
            ? 'bg-danger-bg border border-danger'
            : doneCount === TOTAL_AGENTS
            ? 'bg-success-bg border border-success'
            : 'bg-surface border border-border'}
        `}>
          <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
            isOutOfScope ? 'bg-danger' : doneCount === TOTAL_AGENTS ? 'bg-success' : 'bg-text-faint'
          }`} />
          <span className={`text-[11px] font-medium ${
            isOutOfScope ? 'text-danger' : doneCount === TOTAL_AGENTS ? 'text-success' : 'text-muted'
          }`}>
            {isOutOfScope ? 'Out of Scope Response' : 'Final Answer'}
          </span>
        </div>
      </div>

      {/* Footer hint */}
      <div className="px-4 py-2.5 border-t border-border bg-surface flex-shrink-0">
        <p className="text-[10px] text-muted text-center">
          {isOutOfScope
            ? 'Query was out of scope — pipeline bypassed'
            : 'Click any agent to inspect it'}
        </p>
      </div>
    </div>
  )
}
