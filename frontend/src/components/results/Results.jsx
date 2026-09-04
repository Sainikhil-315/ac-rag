import { useState } from 'react'
import { AGENTS } from '../../constants/agents'
import { Badge } from '../../ui/Badge'

function scoreColor(val) {
  if (val == null) return 'text-text-faint'
  if (val >= 4)   return 'text-success'
  if (val >= 3)   return 'text-warning'
  return 'text-danger'
}

function Card({ title, children, className = "" }) {
  return (
    <div className={`bg-surface border border-border rounded-xl p-5 shadow-sm ${className}`}>
      <p className="text-[10px] font-semibold text-muted uppercase tracking-wider mb-4">
        {title}
      </p>
      {children}
    </div>
  )
}

function ScoreRow({ label, value, max = 5, isOverall = false }) {
  const pct   = value != null ? Math.min((value / max) * 100, 100) : 0
  const color = scoreColor(value)
  return (
    <div className={`flex items-center gap-2.5 mb-${isOverall ? '3.5' : '2'}`}>
      <span className={`font-medium flex-shrink-0 w-28 ${isOverall ? 'text-sm text-primary' : 'text-xs text-secondary'}`}>
        {label}
      </span>
      <div className={`flex-1 h-${isOverall ? '2' : '1'} bg-bg rounded-full overflow-hidden`}>
        <div
          className="h-full rounded-full transition-all duration-600"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className={`font-mono text-right flex-shrink-0 w-8 ${isOverall ? 'text-lg' : 'text-sm'} font-bold`} style={{ color }}>
        {value != null ? value.toFixed(1) : '—'}
      </span>
    </div>
  )
}

function TimingRow({ agent, duration_ms, maxDuration }) {
  const pct = maxDuration > 0 ? Math.min((duration_ms / maxDuration) * 100, 100) : 0
  return (
    <div className="flex items-center gap-2.5 mb-2">
      <div className="flex items-center gap-1.5 w-36 flex-shrink-0">
        <span className="text-[10px] font-mono font-bold" style={{ color: agent.color }}>
          {agent.number}
        </span>
        <span className="text-xs text-text-2 font-medium truncate">
          {agent.title}
        </span>
      </div>
      <div className="flex-1 h-1 bg-bg rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-600 opacity-70"
          style={{ width: `${pct}%`, background: agent.color }}
        />
      </div>
      <span className="text-xs text-secondary font-mono w-12 text-right flex-shrink-0">
        {duration_ms != null ? `${duration_ms} ms` : '—'}
      </span>
    </div>
  )
}

function SummaryChip({ label, value, color = 'var(--c-accent)' }) {
  return (
    <div className="inline-flex items-center gap-1.5 bg-surface-2 border border-border rounded-md px-2.5 py-1">
      <span className="text-[9px] font-semibold text-muted uppercase tracking-wider">
        {label}
      </span>
      <span className="text-xs font-bold font-mono" style={{ color }}>
        {value}
      </span>
    </div>
  )
}

function EmptyResults() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-3 bg-bg">
      <div className="w-12 h-12 rounded-xl bg-surface-2 border border-border flex items-center justify-center text-2xl">
        📊
      </div>
      <p className="text-sm text-secondary font-medium">No run data yet</p>
      <p className="text-xs text-text-faint text-center max-w-[260px] leading-relaxed">
        Switch to the <strong className="text-muted">Playground</strong> tab, upload a PDF and run a query — results will appear here
      </p>
    </div>
  )
}

function RunSelector({ runs, selectedIdx, onSelect }) {
  if (!runs || runs.length === 0) return null
  return (
    <div className="flex gap-1.5 flex-wrap">
      {runs.map((run, i) => {
        const active = i === selectedIdx
        const isUnknown = run.route === 'unknown'
        return (
          <button
            key={i}
            onClick={() => onSelect(i)}
            className={`
              px-2.5 py-1 text-xs font-mono font-medium rounded-md transition-all
              ${active
                ? isUnknown
                  ? 'text-danger bg-danger-bg border border-danger'
                  : 'text-accent bg-accent-bg border border-accent-bdr'
                : 'text-secondary bg-surface hover:bg-surface-2 border border-border'}
            `}
          >
            {i === 0 ? 'Latest' : `Run −${i}`}
            {isUnknown && <span className="ml-1 opacity-70">🚫</span>}
          </button>
        )
      })}
    </div>
  )
}

export default function Results({ agentMeta, runHistory }) {
  const [selectedRunIdx, setSelectedRunIdx] = useState(0)

  const runs = runHistory || []
  const selectedRun = runs[selectedRunIdx] ?? null

  const criticMeta = agentMeta?.['self_reflection']
  const scores     = criticMeta?.output?.scores
  const passed     = criticMeta?.output?.passed

  const timings = AGENTS.map(a => ({
    agent:       a,
    duration_ms: agentMeta?.[a.id]?.duration_ms ?? null,
  }))

  const validTimings  = timings.filter(t => t.duration_ms != null)
  const maxDuration   = Math.max(...validTimings.map(t => t.duration_ms), 1)
  const totalDuration = validTimings.reduce((s, t) => s + t.duration_ms, 0)

  const hasAnyData = validTimings.length > 0 || scores || runs.length > 0

  if (!hasAnyData) return <EmptyResults />

  const isUnknownRoute = selectedRun?.route === 'unknown'

  const SCORE_ROWS = [
    { key: 'faithfulness',    label: 'Faithfulness' },
    { key: 'completeness',    label: 'Completeness' },
    { key: 'table_accuracy',  label: 'Table Accuracy' },
    { key: 'figure_accuracy', label: 'Figure Accuracy' },
    { key: 'conciseness',     label: 'Conciseness' },
  ]

  const overall = scores?.overall ?? null

  return (
    <div className="flex-1 overflow-y-auto p-6 bg-bg flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="text-[10px] font-semibold text-muted uppercase tracking-wider">
          Run Results
        </span>
        {passed != null && (
          <Badge variant={passed ? 'success' : 'danger'} size="sm">
            {passed ? '✓ Self-Reflection Passed' : '✗ Self-Reflection Failed'}
          </Badge>
        )}
      </div>

      {/* Run selector */}
      {runs.length > 1 && (
        <RunSelector runs={runs} selectedIdx={selectedRunIdx} onSelect={setSelectedRunIdx} />
      )}

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Quality Scores */}
        <Card title="Quality Scores">
          {overall != null && (
            <>
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-bold text-primary">Overall</span>
                <span className="text-3xl font-extrabold font-mono" style={{ color: scoreColor(overall) }}>
                  {overall.toFixed(1)}
                </span>
              </div>
              <ScoreRow label="Overall" value={overall} isOverall />
              <div className="border-t border-bg pt-3 mt-1" />
            </>
          )}

          {scores
            ? SCORE_ROWS.map(r => (
                <ScoreRow key={r.key} label={r.label} value={scores[r.key] ?? null} />
              ))
            : (
              <p className="text-xs text-muted text-center py-3">
                {isUnknownRoute
                  ? 'No critic scores — query was out of scope'
                  : 'No critic scores — unknown route'}
              </p>
            )
          }
        </Card>

        {/* Agent Timing */}
        <Card title="Agent Timing">
          {timings.map(({ agent, duration_ms }) => (
            <TimingRow
              key={agent.id}
              agent={agent}
              duration_ms={duration_ms}
              maxDuration={maxDuration}
            />
          ))}
          {totalDuration > 0 && (
            <div className="flex items-center justify-between border-t border-bg pt-2.5 mt-1.5">
              <span className="text-xs font-semibold text-secondary">
                Total (pipeline only)
              </span>
              <span className="text-sm font-bold text-primary font-mono">
                {totalDuration >= 1000
                  ? `${(totalDuration / 1000).toFixed(2)} s`
                  : `${totalDuration} ms`}
              </span>
            </div>
          )}
        </Card>
      </div>

      {/* Run Summary */}
      {selectedRun && (
        <Card title="Run Summary">
          <div className="flex flex-wrap gap-2">
            <SummaryChip label="Route" value={selectedRun.route} color={selectedRun.route === 'rag' ? 'var(--c-accent)' : 'var(--c-danger)'} />
            {!isUnknownRoute && selectedRun.intent != null && (
              <SummaryChip label="Intent" value={selectedRun.intent} />
            )}
            {!isUnknownRoute && selectedRun.complexity != null && (
              <SummaryChip label="Complexity" value={selectedRun.complexity.toFixed(2)} />
            )}
            <SummaryChip label="Retries" value={selectedRun.retries ?? 0} />
            {totalDuration > 0 && (
              <SummaryChip
                label="Pipeline Time"
                value={totalDuration >= 1000
                  ? `${(totalDuration / 1000).toFixed(2)} s`
                  : `${totalDuration} ms`}
              />
            )}
          </div>

          {!isUnknownRoute && selectedRun.rewritten && selectedRun.rewritten !== selectedRun.query && (
            <div className="mt-3 p-2.5 bg-surface-2 border border-border rounded-lg">
              <span className="text-[9px] font-semibold text-muted uppercase tracking-wider mr-2">
                Rewritten Query
              </span>
              <span className="text-sm text-accent font-mono">
                {selectedRun.rewritten}
              </span>
            </div>
          )}
        </Card>
      )}

      {/* Score legend */}
      <div className="flex gap-4 justify-end pt-1">
        {[
          ['≥ 4.0', 'var(--c-success)', 'Pass'],
          ['≥ 3.0', '#D97706', 'Caution'],
          ['< 3.0', 'var(--c-danger)', 'Fail'],
        ].map(([range, color, label]) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ background: color }} />
            <span className="text-[10px] text-muted">{range} — {label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
