import AgentNode from './AgentNode'
import { AGENTS } from '../../constants/agents'

// Connector arrow between nodes
function Connector({ active }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '2px 0' }}>
      <div style={{
        width: '1px', height: '16px',
        background: active ? 'var(--c-accent)' : 'var(--c-border)',
        transition: 'background 0.3s',
      }} />
      <svg width="8" height="5" viewBox="0 0 8 5" fill="none">
        <path d="M4 5L0 0H8L4 5Z" fill={active ? 'var(--c-accent)' : 'var(--c-text-faint)'} />
      </svg>
    </div>
  )
}

// Multi-Modal Knowledge Base node shown between Retrieval Planning and Evidence Validation
function KnowledgeBaseNode({ active }) {
  return (
    <div style={{
      background: active ? 'var(--c-accent-bg)' : 'var(--c-surface-2)',
      border: `1px solid ${active ? 'var(--c-accent-bdr)' : 'var(--c-border)'}`,
      borderRadius: '10px',
      padding: '10px 12px',
      display: 'flex',
      flexDirection: 'column',
      gap: '6px',
      transition: 'all 0.3s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span style={{ fontSize: '14px' }}>🗄️</span>
        <span style={{ fontSize: '11px', fontWeight: 700, color: active ? 'var(--c-accent-txt)' : 'var(--c-text-muted)' }}>
          Multi-Modal Knowledge Base
        </span>
      </div>
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        {[
          { label: 'Text',   icon: '📝' },
          { label: 'Table',  icon: '📊' },
          { label: 'Figure', icon: '🖼️' },
        ].map(({ label, icon }) => (
          <div key={label} style={{
            display: 'flex', alignItems: 'center', gap: '3px',
            background: 'var(--c-surface)', border: `1px solid ${active ? 'var(--c-accent-bdr)' : 'var(--c-border)'}`,
            borderRadius: '5px', padding: '2px 6px',
          }}>
            <span style={{ fontSize: '9px' }}>{icon}</span>
            <span style={{ fontSize: '9px', fontWeight: 600, color: active ? 'var(--c-accent-txt)' : 'var(--c-text-muted)', fontFamily: 'monospace' }}>
              {label}
            </span>
            <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: active ? 'var(--c-accent)' : 'var(--c-text-faint)' }} />
          </div>
        ))}
      </div>
    </div>
  )
}

// Out-of-scope bypass node shown when lastRoute === "unknown"
function OutOfScopeNode() {
  return (
    <div style={{
      background: 'var(--c-danger-bg)',
      border: '1px solid var(--c-danger)',
      borderRadius: '10px', padding: '10px 14px',
      display: 'flex', alignItems: 'center', gap: '10px',
      opacity: 0.9,
    }}>
      <span style={{ fontSize: '18px' }}>🚫</span>
      <div>
        <p style={{ margin: 0, fontSize: '11px', fontWeight: 700, color: 'var(--c-danger)' }}>
          Out of Scope
        </p>
        <p style={{ margin: 0, fontSize: '10px', color: 'var(--c-text-sec)', marginTop: '2px' }}>
          Query bypassed the RAG pipeline
        </p>
      </div>
    </div>
  )
}

const TOTAL_AGENTS = AGENTS.length

export default function PipelineCanvas({ agentStates, agentMeta, selectedAgent, onSelectAgent, lastRoute }) {
  // agentStates: { query_understanding: 'idle'|'active'|'done'|'error', ... }
  // agentMeta:   { query_understanding: 'intent: factual · 0.2', ... }

  const getState = (id) => agentStates?.[id] || 'idle'
  const getMeta  = (id) => {
    const m = agentMeta?.[id]
    if (!m) return null
    return typeof m === 'string' ? m : (m.summary || null)
  }
  const doneCount = AGENTS.filter(a => getState(a.id) === 'done').length
  const isOutOfScope = lastRoute === 'unknown'
  const anyActive = AGENTS.some(a => getState(a.id) !== 'idle')

  // KB is "active" when retrieval_planning is active or done (it's being accessed)
  const retrievalState = getState('retrieval_planning')
  const kbActive = retrievalState === 'active' || retrievalState === 'done'

  return (
    <div style={{
      width: '300px',
      flexShrink: 0,
      borderRight: '1px solid var(--c-border)',
      background: 'var(--c-surface-2)',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      overflow: 'hidden',
      transition: 'background 0.2s, border-color 0.2s',
    }}>

      {/* Canvas header */}
      <div style={{
        padding: '14px 16px 10px',
        borderBottom: '1px solid var(--c-border)',
        background: 'var(--c-surface)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--c-text-sec)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            Pipeline
          </span>
          {isOutOfScope ? (
            <span style={{
              fontSize: '10px', fontWeight: 600, color: 'var(--c-danger)',
              background: 'var(--c-danger-bg)', border: '1px solid var(--c-danger)',
              borderRadius: '999px', padding: '1px 8px',
            }}>
              Out of Scope
            </span>
          ) : (
            <span style={{ fontSize: '11px', fontFamily: 'monospace', color: 'var(--c-text-muted)' }}>
              {doneCount}/{TOTAL_AGENTS} done
            </span>
          )}
        </div>
        {/* Progress bar */}
        <div style={{ marginTop: '8px', height: '2px', background: 'var(--c-bg)', borderRadius: '999px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: isOutOfScope ? '100%' : `${(doneCount / TOTAL_AGENTS) * 100}%`,
            background: isOutOfScope ? 'var(--c-danger)' : doneCount === TOTAL_AGENTS ? 'var(--c-success)' : 'var(--c-accent)',
            borderRadius: '999px',
            transition: 'width 0.4s ease, background 0.3s',
          }} />
        </div>
      </div>

      {/* Nodes — scrollable */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '14px 12px' }}>

        {/* Entry node */}
        <div style={{
          background: 'var(--c-surface)',
          border: '1px solid var(--c-border)',
          borderRadius: '8px',
          padding: '8px 12px',
          marginBottom: '2px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--c-success)', flexShrink: 0 }} />
          <span style={{ fontSize: '12px', color: 'var(--c-text-sec)', fontWeight: 500 }}>User Query</span>
        </div>

        <Connector active={anyActive || isOutOfScope} />

        {/* Out-of-scope path: show bypass node, dim all agents */}
        {isOutOfScope ? (
          <>
            <OutOfScopeNode />
            <div style={{ opacity: 0.3, pointerEvents: 'none', marginTop: '4px' }}>
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
          /* Normal RAG path */
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

        {/* Exit node */}
        <div style={{
          background: isOutOfScope
            ? 'var(--c-danger-bg)'
            : doneCount === TOTAL_AGENTS ? 'var(--c-success-bg)' : 'var(--c-surface)',
          border: `1px solid ${isOutOfScope ? 'var(--c-danger)' : doneCount === TOTAL_AGENTS ? 'var(--c-success)' : 'var(--c-border)'}`,
          borderRadius: '8px',
          padding: '8px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          transition: 'all 0.3s',
        }}>
          <div style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: isOutOfScope ? 'var(--c-danger)' : doneCount === TOTAL_AGENTS ? 'var(--c-success)' : 'var(--c-text-faint)',
            transition: 'background 0.3s',
            flexShrink: 0,
          }} />
          <span style={{
            fontSize: '12px', fontWeight: 500,
            color: isOutOfScope ? 'var(--c-danger)' : doneCount === TOTAL_AGENTS ? 'var(--c-success)' : 'var(--c-text-muted)',
          }}>
            {isOutOfScope ? 'Out of Scope Response' : 'Final Answer'}
          </span>
        </div>

      </div>

      {/* Canvas footer hint */}
      <div style={{
        padding: '10px 14px',
        borderTop: '1px solid var(--c-border)',
        background: 'var(--c-surface)',
        flexShrink: 0,
      }}>
        <p style={{ fontSize: '11px', color: 'var(--c-text-muted)', textAlign: 'center' }}>
          {isOutOfScope ? 'Query was out of scope — pipeline bypassed' : 'Click any agent to inspect it'}
        </p>
      </div>
    </div>
  )
}
