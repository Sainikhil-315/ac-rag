import { useState, useEffect } from 'react'
import { AGENTS } from '../../constants/agents'

// ─── agentMeta shape (set by Playground during a live run) ───────────────────
// null                                  → no run yet
// { summary, input, output, duration_ms, error }

export default function AgentInspector({ selectedAgentId, agentStates, agentMeta }) {
  const [innerTab, setInnerTab] = useState('overview')

  // Reset inner tab to overview whenever a different agent is selected
  useEffect(() => { setInnerTab('overview') }, [selectedAgentId])

  const agent = AGENTS.find(a => a.id === selectedAgentId)

  // ── empty state ──────────────────────────────────────────────────────────
  if (!agent) {
    return (
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        gap: '10px', background: 'var(--c-surface-2)',
      }}>
        <div style={{
          width: '44px', height: '44px', borderRadius: '12px',
          background: 'var(--c-bg)', border: '1px solid var(--c-border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '20px',
        }}>🔬</div>
        <p style={{ fontSize: '13px', color: 'var(--c-text-sec)', fontWeight: 500 }}>
          Select an agent to inspect
        </p>
        <p style={{ fontSize: '11px', color: 'var(--c-text-faint)', maxWidth: '210px', textAlign: 'center', lineHeight: 1.5 }}>
          Click any node in the pipeline to see its schema, description, and live trace
        </p>
      </div>
    )
  }

  const state = agentStates?.[agent.id] || 'idle'
  const meta  = agentMeta?.[agent.id]  ?? null   // null | { summary, input, output, duration_ms, error }

  const BADGE = {
    idle:   { label: 'Idle',    bg: 'var(--c-bg)',          color: 'var(--c-text-sec)', dot: 'var(--c-text-faint)' },
    active: { label: 'Running', bg: 'var(--c-accent-bg)',   color: 'var(--c-accent)',   dot: 'var(--c-accent)'     },
    done:   { label: 'Done',    bg: 'var(--c-success-bg)',  color: 'var(--c-success)',  dot: 'var(--c-success)'    },
    error:  { label: 'Error',   bg: 'var(--c-danger-bg)',   color: 'var(--c-danger)',   dot: 'var(--c-danger)'     },
  }[state]

  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      background: 'var(--c-surface-2)', overflow: 'hidden',
      borderTop: `3px solid ${agent.color}`,
    }}>

      {/* ── Inspector header ───────────────────────────────────────────── */}
      <div style={{
        padding: '14px 20px 0',
        background: 'var(--c-surface)',
        borderBottom: '1px solid var(--c-border)',
        flexShrink: 0,
      }}>
        {/* Agent identity row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
          <div style={{
            width: '38px', height: '38px', borderRadius: '10px',
            background: agent.lightBg, flexShrink: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '18px',
          }}>
            {agent.icon}
          </div>

          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
              <span style={{ fontSize: '11px', fontFamily: 'monospace', color: agent.color, fontWeight: 700 }}>
                {agent.number}
              </span>
              <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--c-text)' }}>
                {agent.title}
              </span>
            </div>
            <p style={{ fontSize: '11px', color: 'var(--c-text-sec)', margin: 0, marginTop: '2px' }}>
              {agent.subtitle}
            </p>
          </div>

          {/* State badge */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            background: BADGE.bg, borderRadius: '999px',
            padding: '3px 10px', flexShrink: 0,
          }}>
            <div style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: BADGE.dot,
              animation: state === 'active' ? 'pulse-dot 1.5s ease-in-out infinite' : 'none',
            }} />
            <span style={{ fontSize: '11px', fontWeight: 600, color: BADGE.color }}>
              {BADGE.label}
            </span>
          </div>
        </div>

        {/* Inner tab bar */}
        <div style={{ display: 'flex', gap: '0' }}>
          {['overview', 'trace'].map(tab => {
            const active = innerTab === tab
            const label  = tab === 'overview' ? 'Overview' : 'Live Trace'
            return (
              <button key={tab} onClick={() => setInnerTab(tab)} style={{
                padding: '7px 16px',
                fontSize: '12px', fontWeight: active ? 600 : 500,
                color: active ? agent.color : 'var(--c-text-sec)',
                background: 'transparent',
                border: 'none',
                borderBottom: active ? `2px solid ${agent.color}` : '2px solid transparent',
                cursor: 'pointer',
                transition: 'color 0.15s, border-color 0.15s',
                fontFamily: 'Inter, sans-serif',
                marginBottom: '-1px',
              }}>
                {label}
                {tab === 'trace' && meta && (
                  <span style={{
                    marginLeft: '5px', fontSize: '10px', fontWeight: 600,
                    background: state === 'error' ? 'var(--c-danger-bg)' : 'var(--c-success-bg)',
                    color: state === 'error' ? 'var(--c-danger)' : 'var(--c-success)',
                    borderRadius: '999px', padding: '1px 6px',
                  }}>
                    {state === 'error' ? 'err' : state === 'active' ? '…' : '✓'}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Tab content ────────────────────────────────────────────────── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>

        {innerTab === 'overview' && (
          <OverviewTab agent={agent} />
        )}

        {innerTab === 'trace' && (
          <TraceTab state={state} meta={meta} agent={agent} />
        )}

      </div>
    </div>
  )
}

// ─── Overview tab ─────────────────────────────────────────────────────────────
function OverviewTab({ agent }) {
  return (
    <>
      <Section label="Description">
        <p style={{ fontSize: '13px', color: 'var(--c-text-2)', lineHeight: 1.65, margin: 0 }}>
          {agent.description}
        </p>
      </Section>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <Section label="Input schema">
          <Mono>{agent.input}</Mono>
        </Section>
        <Section label="Output schema">
          <Mono>{agent.output}</Mono>
        </Section>
      </div>

      <Section label="Example">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <IORow label="In"  value={agent.example.in}  color="#3F3F46" />
          <IORow label="Out" value={agent.example.out} color="#18181B" />
        </div>
      </Section>
    </>
  )
}

// ─── Live Trace tab ───────────────────────────────────────────────────────────
function TraceTab({ state, meta, agent }) {

  // No run yet
  if (!meta && state === 'idle') {
    return (
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        gap: '8px', paddingTop: '40px',
      }}>
        <div style={{ fontSize: '28px', opacity: 0.3 }}>{agent.icon}</div>
        <p style={{ fontSize: '13px', color: 'var(--c-text-muted)', fontWeight: 500 }}>No run data yet</p>
        <p style={{ fontSize: '11px', color: 'var(--c-text-faint)', textAlign: 'center', maxWidth: '200px', lineHeight: 1.5 }}>
          Switch to the Playground tab, upload a PDF and run a query to see live data here
        </p>
      </div>
    )
  }

  // Currently running
  if (state === 'active') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: '10px',
          background: 'var(--c-accent-bg)', border: '1px solid var(--c-accent-bdr)',
          borderRadius: '8px', padding: '12px 14px',
        }}>
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            {[0, 1, 2].map(i => (
              <span key={i} className="running-dot" style={{ background: agent.color }} />
            ))}
          </div>
          <span style={{ fontSize: '13px', color: agent.color, fontWeight: 600 }}>
            Agent is running…
          </span>
        </div>
        {/* Show previous input if available */}
        {meta?.input && (
          <Section label="Input (this run)">
            <JsonBlock data={meta.input} accentColor={agent.color} />
          </Section>
        )}
      </div>
    )
  }

  // Error state
  if (state === 'error') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{
          background: 'var(--c-danger-bg)', border: '1px solid var(--c-danger)',
          borderRadius: '8px', padding: '12px 14px',
        }}>
          <p style={{ fontSize: '12px', color: 'var(--c-danger)', fontWeight: 600, margin: '0 0 4px' }}>
            Agent failed
          </p>
          <p style={{ fontSize: '12px', color: 'var(--c-danger)', margin: 0, fontFamily: 'monospace', opacity: 0.8 }}>
            {meta?.error || 'Unknown error'}
          </p>
        </div>
        {meta?.input && (
          <Section label="Input at failure">
            <JsonBlock data={meta.input} accentColor="#EF4444" />
          </Section>
        )}
      </div>
    )
  }

  // Done — show full trace
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

      {/* Timing chip */}
      {meta?.duration_ms != null && (
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <Chip label="Duration" value={`${meta.duration_ms} ms`} color={agent.color} />
          {meta?.summary && <Chip label="Summary" value={meta.summary} color={agent.color} />}
        </div>
      )}

      {/* Summary (if no timing chip showed it) */}
      {meta?.summary && meta?.duration_ms == null && (
        <Section label="Summary">
          <Mono accent color={agent.color}>{meta.summary}</Mono>
        </Section>
      )}

      {/* Input */}
      {meta?.input != null && (
        <Section label="Input (actual)">
          <JsonBlock data={meta.input} accentColor={agent.color} />
        </Section>
      )}

      {/* Output */}
      {meta?.output != null && (
        <Section label="Output (actual)">
          <JsonBlock data={meta.output} accentColor={agent.color} />
        </Section>
      )}

      {/* Fallback: plain summary text if no structured data */}
      {!meta?.input && !meta?.output && meta?.summary && (
        <Section label="Output">
          <Mono accent color={agent.color}>{meta.summary}</Mono>
        </Section>
      )}

    </div>
  )
}

// ─── Reusable primitives ──────────────────────────────────────────────────────
function Section({ label, children }) {
  return (
    <div>
      <p style={{
        fontSize: '10px', fontWeight: 600, color: 'var(--c-text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.07em', margin: '0 0 6px 0',
      }}>
        {label}
      </p>
      <div style={{
        background: 'var(--c-surface)', border: '1px solid var(--c-border)',
        borderRadius: '8px', padding: '10px 12px',
      }}>
        {children}
      </div>
    </div>
  )
}

function IORow({ label, value, color }) {
  return (
    <div>
      <span style={{
        fontSize: '10px', fontWeight: 700, color: 'var(--c-text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '4px',
        display: 'block',
      }}>
        {label}
      </span>
      <pre style={{
        margin: 0, fontSize: '11.5px',
        fontFamily: 'JetBrains Mono, Fira Code, monospace',
        color: color || 'var(--c-text-2)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.6,
      }}>
        {value}
      </pre>
    </div>
  )
}

function Mono({ children, accent, color }) {
  return (
    <pre style={{
      margin: 0, fontSize: '11.5px',
      fontFamily: 'JetBrains Mono, Fira Code, monospace',
      color: accent ? (color || 'var(--c-accent)') : 'var(--c-text-2)',
      whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.6,
    }}>
      {children}
    </pre>
  )
}

function Chip({ label, value, color }) {
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: '5px',
      background: 'var(--c-surface)', border: '1px solid var(--c-border)',
      borderRadius: '6px', padding: '4px 10px',
    }}>
      <span style={{ fontSize: '10px', color: 'var(--c-text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </span>
      <span style={{ fontSize: '12px', color: color || 'var(--c-accent)', fontWeight: 600, fontFamily: 'monospace' }}>
        {value}
      </span>
    </div>
  )
}

// Renders an object as syntax-highlighted key: value lines
function JsonBlock({ data, accentColor }) {
  if (data === null || data === undefined) return null

  if (typeof data === 'string') {
    return (
      <pre style={{
        margin: 0, fontSize: '11.5px',
        fontFamily: 'JetBrains Mono, Fira Code, monospace',
        color: 'var(--c-text-2)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.6,
      }}>
        {data}
      </pre>
    )
  }

  const lines = formatObject(data)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
      {lines.map((line, i) => (
        <div key={i} style={{ display: 'flex', gap: '8px', fontSize: '11.5px', fontFamily: 'JetBrains Mono, Fira Code, monospace', lineHeight: 1.6 }}>
          {line.key != null && (
            <span style={{ color: accentColor, fontWeight: 600, flexShrink: 0 }}>
              {line.key}:
            </span>
          )}
          <span style={{ color: 'var(--c-text-2)', wordBreak: 'break-word' }}>
            {line.value}
          </span>
        </div>
      ))}
    </div>
  )
}

// Converts an object into a flat array of { key, value } pairs for display
function formatObject(obj, prefix = '') {
  const lines = []
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      lines.push(...formatObject(v, key))
    } else if (Array.isArray(v)) {
      lines.push({ key, value: JSON.stringify(v) })
    } else {
      lines.push({ key, value: String(v) })
    }
  }
  return lines
}
