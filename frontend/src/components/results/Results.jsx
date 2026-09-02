import { useState } from 'react'
import { AGENTS } from '../../constants/agents'

// ── Score colour helper ───────────────────────────────────────────────────────
function scoreColor(val) {
  if (val == null) return 'var(--c-text-faint)'
  if (val >= 4)   return 'var(--c-success)'
  if (val >= 3)   return '#D97706'
  return 'var(--c-danger)'
}

// ── Section wrapper ───────────────────────────────────────────────────────────
function Card({ title, children, style = {} }) {
  return (
    <div style={{
      background: 'var(--c-surface)', border: '1px solid var(--c-border)',
      borderRadius: '10px', padding: '16px 18px', ...style,
    }}>
      <p style={{
        fontSize: '10px', fontWeight: 600, color: 'var(--c-text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.07em', margin: '0 0 14px',
      }}>
        {title}
      </p>
      {children}
    </div>
  )
}

// ── Score bar row ─────────────────────────────────────────────────────────────
function ScoreRow({ label, value, max = 5, isOverall = false }) {
  const pct   = value != null ? Math.min((value / max) * 100, 100) : 0
  const color = scoreColor(value)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: isOverall ? '14px' : '8px' }}>
      <span style={{
        fontSize: isOverall ? '12px' : '11px',
        fontWeight: isOverall ? 700 : 500,
        color: 'var(--c-text-2)',
        width: '110px', flexShrink: 0,
      }}>
        {label}
      </span>
      <div style={{
        flex: 1, height: isOverall ? '8px' : '5px',
        background: 'var(--c-bg)', borderRadius: '999px', overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: color, borderRadius: '999px',
          transition: 'width 0.6s ease',
        }} />
      </div>
      <span style={{
        fontSize: isOverall ? '14px' : '12px',
        fontWeight: 700, color,
        fontFamily: 'monospace', width: '30px', textAlign: 'right', flexShrink: 0,
      }}>
        {value != null ? value.toFixed(1) : '—'}
      </span>
    </div>
  )
}

// ── Agent timing bar ──────────────────────────────────────────────────────────
function TimingRow({ agent, duration_ms, maxDuration }) {
  const pct = maxDuration > 0 ? Math.min((duration_ms / maxDuration) * 100, 100) : 0
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '7px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', width: '140px', flexShrink: 0 }}>
        <span style={{
          fontSize: '10px', fontFamily: 'monospace', fontWeight: 700,
          color: agent.color, flexShrink: 0,
        }}>
          {agent.number}
        </span>
        <span style={{
          fontSize: '11px', color: 'var(--c-text-2)', fontWeight: 500,
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}>
          {agent.title}
        </span>
      </div>
      <div style={{
        flex: 1, height: '5px',
        background: 'var(--c-bg)', borderRadius: '999px', overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: agent.color, borderRadius: '999px',
          opacity: 0.7, transition: 'width 0.6s ease',
        }} />
      </div>
      <span style={{
        fontSize: '11px', color: 'var(--c-text-sec)', fontFamily: 'monospace',
        width: '52px', textAlign: 'right', flexShrink: 0,
      }}>
        {duration_ms != null ? `${duration_ms} ms` : '—'}
      </span>
    </div>
  )
}

// ── Run summary chips ─────────────────────────────────────────────────────────
function SummaryChip({ label, value, accent }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '5px',
      background: accent ? 'var(--c-accent-bg)' : 'var(--c-surface-2)',
      border: `1px solid ${accent ? 'var(--c-accent-bdr)' : 'var(--c-border)'}`,
      borderRadius: '6px', padding: '5px 12px',
    }}>
      <span style={{
        fontSize: '10px', fontWeight: 600, color: 'var(--c-text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.05em',
      }}>
        {label}
      </span>
      <span style={{
        fontSize: '12px', fontWeight: 700,
        color: accent ? 'var(--c-accent)' : 'var(--c-text)',
        fontFamily: 'monospace',
      }}>
        {value}
      </span>
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────
function EmptyResults() {
  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      gap: '10px', background: 'var(--c-bg)',
    }}>
      <div style={{
        width: '48px', height: '48px', borderRadius: '12px',
        background: 'var(--c-surface-2)', border: '1px solid var(--c-border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '22px',
      }}>📊</div>
      <p style={{ fontSize: '13px', color: 'var(--c-text-sec)', fontWeight: 500 }}>
        No run data yet
      </p>
      <p style={{ fontSize: '11px', color: 'var(--c-text-faint)', maxWidth: '240px', textAlign: 'center', lineHeight: 1.5 }}>
        Switch to the <strong style={{ color: 'var(--c-text-muted)' }}>Playground</strong> tab, upload a PDF and run a query — results will appear here
      </p>
    </div>
  )
}

// ── Run history selector ──────────────────────────────────────────────────────
function RunSelector({ runs, selectedIdx, onSelect }) {
  if (!runs || runs.length === 0) return null
  return (
    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
      {runs.map((run, i) => {
        const active = i === selectedIdx
        const isUnknown = run.route === 'unknown'
        return (
          <button key={i} onClick={() => onSelect(i)} style={{
            padding: '4px 10px', fontSize: '11px', fontFamily: 'monospace',
            fontWeight: active ? 700 : 500,
            color: active ? (isUnknown ? 'var(--c-danger)' : 'var(--c-accent)') : 'var(--c-text-sec)',
            background: active ? (isUnknown ? 'var(--c-danger-bg)' : 'var(--c-accent-bg)') : 'var(--c-surface)',
            border: `1px solid ${active ? (isUnknown ? 'var(--c-danger)' : 'var(--c-accent-bdr)') : 'var(--c-border)'}`,
            borderRadius: '6px', cursor: 'pointer',
            transition: 'all 0.15s',
          }}>
            {i === 0 ? 'Latest' : `Run −${i}`}
            {isUnknown && <span style={{ marginLeft: '4px', opacity: 0.7 }}>🚫</span>}
          </button>
        )
      })}
    </div>
  )
}

// ── Main Results component ────────────────────────────────────────────────────
export default function Results({ agentMeta, runHistory }) {
  const [selectedRunIdx, setSelectedRunIdx] = useState(0)

  // runHistory is newest-first array; default to index 0 (latest)
  const runs = runHistory || []
  const selectedRun = runs[selectedRunIdx] ?? null

  // Scores live in self_reflection agent meta
  const criticMeta = agentMeta?.['self_reflection']
  const scores     = criticMeta?.output?.scores
  const passed     = criticMeta?.output?.passed

  // Timing: collect duration_ms per agent
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
    <div style={{
      flex: 1, overflowY: 'auto', padding: '20px',
      background: 'var(--c-bg)', display: 'flex', flexDirection: 'column', gap: '14px',
    }}>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <span style={{
          fontSize: '11px', fontWeight: 600, color: 'var(--c-text-sec)',
          textTransform: 'uppercase', letterSpacing: '0.07em',
        }}>
          Run Results
        </span>
        {passed != null && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            background: passed ? 'var(--c-success-bg)' : 'var(--c-danger-bg)',
            border: `1px solid ${passed ? 'var(--c-success)' : 'var(--c-danger)'}`,
            borderRadius: '999px', padding: '3px 12px',
          }}>
            <span style={{ fontSize: '13px' }}>{passed ? '✓' : '✗'}</span>
            <span style={{
              fontSize: '11px', fontWeight: 700,
              color: passed ? 'var(--c-success)' : 'var(--c-danger)',
            }}>
              {passed ? 'Self-Reflection Passed' : 'Self-Reflection Failed'}
            </span>
          </div>
        )}
      </div>

      {/* ── Run history selector ────────────────────────────────────────── */}
      {runs.length > 1 && (
        <RunSelector runs={runs} selectedIdx={selectedRunIdx} onSelect={setSelectedRunIdx} />
      )}

      {/* ── Two-column main grid ────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>

        {/* Quality Scores */}
        <Card title="Quality Scores">
          {overall != null && (
            <>
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                marginBottom: '6px',
              }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--c-text)' }}>Overall</span>
                <span style={{
                  fontSize: '28px', fontWeight: 800,
                  color: scoreColor(overall), fontFamily: 'monospace',
                  lineHeight: 1,
                }}>
                  {overall.toFixed(1)}
                </span>
              </div>
              <ScoreRow label="Overall" value={overall} isOverall />
              <div style={{ borderTop: '1px solid var(--c-bg)', paddingTop: '12px', marginTop: '4px' }} />
            </>
          )}

          {scores
            ? SCORE_ROWS.map(r => (
                <ScoreRow key={r.key} label={r.label} value={scores[r.key] ?? null} />
              ))
            : (
              <p style={{ fontSize: '12px', color: 'var(--c-text-muted)', textAlign: 'center', padding: '12px 0' }}>
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
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              borderTop: '1px solid var(--c-bg)', paddingTop: '10px', marginTop: '6px',
            }}>
              <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--c-text-sec)' }}>
                Total (pipeline only)
              </span>
              <span style={{
                fontSize: '13px', fontWeight: 700, color: 'var(--c-text)', fontFamily: 'monospace',
              }}>
                {totalDuration >= 1000
                  ? `${(totalDuration / 1000).toFixed(2)} s`
                  : `${totalDuration} ms`}
              </span>
            </div>
          )}
        </Card>
      </div>

      {/* ── Run Summary ─────────────────────────────────────────────────── */}
      {selectedRun && (
        <Card title="Run Summary">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            <SummaryChip label="Route"   value={selectedRun.route} accent={selectedRun.route === 'rag'} />
            {/* Only show intent and complexity for RAG route */}
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
            <div style={{
              marginTop: '12px', background: 'var(--c-surface-2)', border: '1px solid var(--c-border)',
              borderRadius: '6px', padding: '8px 10px',
            }}>
              <span style={{
                fontSize: '10px', fontWeight: 600, color: 'var(--c-text-muted)',
                textTransform: 'uppercase', letterSpacing: '0.06em', marginRight: '8px',
              }}>
                Rewritten Query
              </span>
              <span style={{ fontSize: '12px', color: 'var(--c-accent)', fontFamily: 'monospace' }}>
                {selectedRun.rewritten}
              </span>
            </div>
          )}
        </Card>
      )}

      {/* ── Score colour legend ──────────────────────────────────────────── */}
      <div style={{
        display: 'flex', gap: '16px', justifyContent: 'flex-end', paddingTop: '2px',
      }}>
        {[['≥ 4.0', 'var(--c-success)', 'Pass'], ['≥ 3.0', '#D97706', 'Caution'], ['< 3.0', 'var(--c-danger)', 'Fail']].map(([range, color, label]) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }} />
            <span style={{ fontSize: '10px', color: 'var(--c-text-muted)' }}>{range} — {label}</span>
          </div>
        ))}
      </div>

    </div>
  )
}
