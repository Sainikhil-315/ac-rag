import { useState, useEffect } from 'react'
import { AGENTS } from '../../constants/agents'

function Section({ label, children }) {
  return (
    <div>
      <p className="text-[10px] font-semibold text-muted uppercase tracking-wider mb-1.5">
        {label}
      </p>
      <div className="bg-surface border border-border rounded-lg p-3">
        {children}
      </div>
    </div>
  )
}

function IORow({ label, value, color }) {
  return (
    <div>
      <span className="text-[9px] font-bold text-muted uppercase tracking-wider block mb-1">
        {label}
      </span>
      <pre className="text-[11px] font-mono leading-relaxed whitespace-pre-wrap break-words" style={{ color: color || 'var(--c-text-2)' }}>
        {value}
      </pre>
    </div>
  )
}

function Mono({ children, accent, color }) {
  return (
    <pre
      className="text-[11px] font-mono leading-relaxed whitespace-pre-wrap break-words"
      style={{ color: accent ? (color || 'var(--c-accent)') : 'var(--c-text-2)' }}
    >
      {children}
    </pre>
  )
}

function Chip({ label, value, color }) {
  return (
    <div className="inline-flex items-center gap-1.5 bg-surface border border-border rounded-md px-2.5 py-1">
      <span className="text-[9px] font-semibold text-muted uppercase tracking-wider">
        {label}
      </span>
      <span className="text-[11px] font-semibold font-mono" style={{ color: color || 'var(--c-accent)' }}>
        {value}
      </span>
    </div>
  )
}

function JsonBlock({ data, accentColor }) {
  if (data === null || data === undefined) return null

  if (typeof data === 'string') {
    return (
      <pre className="text-[11px] font-mono leading-relaxed whitespace-pre-wrap break-words" style={{ color: 'var(--c-text-2)' }}>
        {data}
      </pre>
    )
  }

  const lines = formatObject(data)

  return (
    <div className="flex flex-col gap-0.5">
      {lines.map((line, i) => (
        <div
          key={i}
          className="flex gap-2 text-[11px] font-mono leading-relaxed"
        >
          {line.key != null && (
            <span className="flex-shrink-0 font-semibold" style={{ color: accentColor || 'var(--c-accent)' }}>
              {line.key}:
            </span>
          )}
          <span className="break-words" style={{ color: 'var(--c-text-2)' }}>
            {line.value}
          </span>
        </div>
      ))}
    </div>
  )
}

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

function OverviewTab({ agent }) {
  return (
    <>
      <Section label="Description">
        <p className="text-sm text-text-2 leading-relaxed">
          {agent.description}
        </p>
      </Section>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Section label="Input schema">
          <Mono>{agent.input}</Mono>
        </Section>
        <Section label="Output schema">
          <Mono>{agent.output}</Mono>
        </Section>
      </div>

      <Section label="Example">
        <div className="flex flex-col gap-2.5">
          <IORow label="In"  value={agent.example.in}  color="#3F3F46" />
          <IORow label="Out" value={agent.example.out} color="#18181B" />
        </div>
      </Section>
    </>
  )
}

function TraceTab({ state, meta, agent }) {
  if (!meta && state === 'idle') {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2 pt-10">
        <div className="text-2xl opacity-25">{agent.icon}</div>
        <p className="text-sm text-muted font-medium">No run data yet</p>
        <p className="text-xs text-text-faint text-center max-w-[220px] leading-relaxed">
          Switch to the Playground tab, upload a PDF and run a query to see live data here
        </p>
      </div>
    )
  }

  if (state === 'active') {
    return (
      <div className="flex flex-col gap-3.5">
        <div className="flex items-center gap-2.5 bg-accent-bg border border-accent-bdr rounded-lg px-3.5 py-3">
          <div className="flex gap-1 items-center">
            {[0, 1, 2].map(i => (
              <span key={i} className="block w-1 h-1 rounded-full bg-accent animate-[pulse-dot_1.5s_ease-in-out_infinite]" />
            ))}
          </div>
          <span className="text-sm font-semibold" style={{ color: agent.color }}>
            Agent is running…
          </span>
        </div>
        {meta?.input && (
          <Section label="Input (this run)">
            <JsonBlock data={meta.input} accentColor={agent.color} />
          </Section>
        )}
      </div>
    )
  }

  if (state === 'error') {
    return (
      <div className="flex flex-col gap-3.5">
        <div className="bg-danger-bg border border-danger rounded-lg px-3.5 py-3">
          <p className="text-xs font-semibold text-danger mb-1">Agent failed</p>
          <p className="text-xs font-mono opacity-80" style={{ color: 'var(--c-danger)' }}>
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

  // Done — full trace
  return (
    <div className="flex flex-col gap-3.5">
      {meta?.duration_ms != null && (
        <div className="flex gap-2 flex-wrap">
          <Chip label="Duration" value={`${meta.duration_ms} ms`} color={agent.color} />
          {meta?.summary && <Chip label="Summary" value={meta.summary} color={agent.color} />}
        </div>
      )}

      {meta?.summary && meta?.duration_ms == null && (
        <Section label="Summary">
          <Mono accent color={agent.color}>{meta.summary}</Mono>
        </Section>
      )}

      {meta?.input != null && (
        <Section label="Input (actual)">
          <JsonBlock data={meta.input} accentColor={agent.color} />
        </Section>
      )}

      {meta?.output != null && (
        <Section label="Output (actual)">
          <JsonBlock data={meta.output} accentColor={agent.color} />
        </Section>
      )}

      {!meta?.input && !meta?.output && meta?.summary && (
        <Section label="Output">
          <Mono accent color={agent.color}>{meta.summary}</Mono>
        </Section>
      )}
    </div>
  )
}

export default function AgentInspector({ selectedAgentId, agentStates, agentMeta }) {
  const [innerTab, setInnerTab] = useState('overview')

  useEffect(() => { setInnerTab('overview') }, [selectedAgentId])

  const agent = AGENTS.find(a => a.id === selectedAgentId)

  if (!agent) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2.5 bg-surface-2">
        <div className="w-11 h-11 rounded-xl bg-surface border border-border flex items-center justify-center text-2xl">
          🔬
        </div>
        <p className="text-sm text-secondary font-medium">
          Select an agent to inspect
        </p>
        <p className="text-xs text-text-faint text-center max-w-[220px] leading-relaxed">
          Click any node in the pipeline to see its schema, description, and live trace
        </p>
      </div>
    )
  }

  const state = agentStates?.[agent.id] || 'idle'
  const meta  = agentMeta?.[agent.id] ?? null

  const BADGE = {
    idle:   { label: 'Idle',    bg: 'bg-bg',          color: 'var(--c-text-sec)', dot: 'var(--c-text-faint)' },
    active: { label: 'Running', bg: 'bg-accent-bg',   color: 'var(--c-accent)',   dot: 'var(--c-accent)' },
    done:   { label: 'Done',    bg: 'bg-success-bg',  color: 'var(--c-success)',  dot: 'var(--c-success)' },
    error:  { label: 'Error',   bg: 'bg-danger-bg',   color: 'var(--c-danger)',   dot: 'var(--c-danger)' },
  }[state]

  return (
    <div className="flex-1 flex flex-col bg-surface-2 overflow-hidden" style={{ borderTop: `3px solid ${agent.color}` }}>
      {/* Inspector header */}
      <div className="px-5 py-4 border-b border-border bg-surface flex-shrink-0">
        {/* Agent identity row */}
        <div className="flex items-center gap-3 mb-3">
          <div
            className="w-10 h-10 rounded-xl flex-shrink-0 flex items-center justify-center text-lg border"
            style={{ backgroundColor: agent.lightBg, borderColor: agent.color + '33', color: agent.color }}
          >
            {agent.icon}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-mono font-semibold" style={{ color: agent.color }}>
                {agent.number}
              </span>
              <span className="text-sm font-semibold text-primary">
                {agent.title}
              </span>
            </div>
            <p className="text-xs text-secondary mt-0.5">
              {agent.subtitle}
            </p>
          </div>

          {/* State badge */}
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${BADGE.bg}`}>
            <div
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: BADGE.dot,
                animation: state === 'active' ? 'pulse-dot 1.5s ease-in-out infinite' : 'none',
              }}
            />
            <span className="text-[10px] font-semibold" style={{ color: BADGE.color }}>
              {BADGE.label}
            </span>
          </div>
        </div>

        {/* Inner tabs */}
        <div className="flex gap-0.5 border-b border-border">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'trace', label: 'Live Trace' },
          ].map(tab => {
            const active = innerTab === tab.id
            const showDot = tab.id === 'trace'
            const dotStatus = state === 'error' ? 'err' : state === 'active' ? '…' : '✓'
            const dotColor = state === 'error' ? 'text-danger bg-danger-bg' : state === 'active' ? 'text-accent bg-accent-bg' : 'text-success bg-success-bg'

            return (
              <button
                key={tab.id}
                onClick={() => setInnerTab(tab.id)}
                className={`
                  px-4 py-2 text-xs font-medium rounded-t-lg transition-all
                  ${active
                    ? 'text-primary border-b-2'
                    : 'text-secondary hover:text-primary border-b-2 border-transparent'}
                `}
                style={{
                  borderColor: active ? agent.color : 'transparent',
                  color: active ? 'var(--c-text)' : 'var(--c-text-sec)',
                }}
              >
                {tab.label}
                {showDot && (
                  <span className={`ml-1.5 text-[8px] font-bold px-1.5 py-0.25 rounded-full ${dotColor}`}>
                    {dotStatus}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3.5">
        {innerTab === 'overview' && <OverviewTab agent={agent} />}
        {innerTab === 'trace' && <TraceTab state={state} meta={meta} agent={agent} />}
      </div>
    </div>
  )
}
