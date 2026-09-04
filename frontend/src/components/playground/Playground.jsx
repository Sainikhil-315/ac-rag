import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Copy, Check, FileText, RefreshCw, Upload as UploadIcon } from 'lucide-react'
import { AGENTS } from '../../constants/agents'

let _id = 0
const nextId = () => ++_id

// ── Helpers ───────────────────────────────────────────────────────────────────
function scoreColor(v) {
  if (v == null) return '#D4D4D8'
  return v >= 4 ? '#059669' : v >= 3 ? '#D97706' : '#DC2626'
}

function lsGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v !== null ? JSON.parse(v) : fallback }
  catch { return fallback }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch {}
}

// ── User message bubble ───────────────────────────────────────────────────────
function UserBubble({ content }) {
  return (
    <div className="flex justify-end animate-slide-up">
      <div className="max-w-[72%] bg-accent-bg border border-accent-bdr rounded-2xl rounded-tr-sm px-4 py-3 text-sm text-accent-txt leading-relaxed">
        {content}
      </div>
    </div>
  )
}

// ── Expandable section ───────────────────────────────────────────────────────
function Expandable({ icon, label, badge, badgeColor = '#6366F1', defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-t border-border-in mt-3 pt-2">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 w-full text-left bg-none border-none cursor-pointer py-1.5"
      >
        <span className="text-sm">{icon}</span>
        <span className="text-[10px] font-semibold text-secondary uppercase tracking-wider">
          {label}
        </span>
        {badge != null && (
          <span
            className="text-[9px] font-bold px-1.5 py-0.25 rounded-full text-white"
            style={{ background: badgeColor }}
          >
            {badge}
          </span>
        )}
        <span className="ml-auto text-xs text-text-faint">
          {open ? '▲' : '▼'}
        </span>
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  )
}

// ── Score bar row ─────────────────────────────────────────────────────────────
function ScoreBar({ label, value }) {
  const pct   = value != null ? Math.min((value / 5) * 100, 100) : 0
  const color = scoreColor(value)
  return (
    <div className="flex items-center gap-2 mb-1.5">
      <span className="text-xs text-secondary w-24 flex-shrink-0">{label}</span>
      <div className="flex-1 h-1 bg-bg rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="text-xs font-bold font-mono w-7 text-right flex-shrink-0" style={{ color }}>
        {value != null ? parseFloat(value).toFixed(1) : '—'}
      </span>
    </div>
  )
}

// ── Agent pipeline trace ──────────────────────────────────────────────────────
function AgentTrace({ trace }) {
  if (!trace?.length) return null
  return (
    <div className="flex flex-col gap-1">
      {trace.map(({ agentId, summary, duration_ms }) => {
        const agent = AGENTS.find(a => a.id === agentId)
        if (!agent) return null
        return (
          <div
            key={agentId}
            className="flex items-center gap-2 bg-surface-2 rounded-md border border-border px-2 py-1.5"
          >
            <span className="text-xs">{agent.icon}</span>
            <span className="text-[10px] font-mono font-bold" style={{ color: agent.color }}>
              {agent.number}
            </span>
            <span className="text-xs font-semibold text-text-2 truncate">
              {agent.title}
            </span>
            {summary && (
              <span className="text-[10px] text-muted truncate">
                · {summary}
              </span>
            )}
            {duration_ms != null && (
              <span className="text-[10px] text-muted font-mono ml-auto">
                {duration_ms}ms
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Confidence badge ───────────────────────────────────────────────────────────
function ConfidenceBadge({ overall }) {
  if (overall == null) return null
  const color = scoreColor(overall)
  const label = overall >= 4 ? 'High' : overall >= 3 ? 'Medium' : 'Low'
  return (
    <span
      title={`Self-reflection score: ${overall}/5`}
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold"
      style={{ background: color + '1A', color: color }}
    >
      <span className="block w-1.5 h-1.5 rounded-full" style={{ background: color }} />
      {label} confidence · {parseFloat(overall).toFixed(1)}
    </span>
  )
}

// ── Assistant response card ───────────────────────────────────────────────────
function AssistantCard({ message }) {
  const {
    answer, isStreaming, scores, sources,
    route, intent, complexity, rewritten, retries,
    agentTrace, error,
  } = message

  const [copied, setCopied] = useState(false)

  const ROUTE_META = {
    rag:     { bg: 'bg-accent-bg', color: 'text-accent-txt', label: '⚡ RAG' },
    unknown: { bg: 'bg-surface-2', color: 'text-secondary', label: '🚫 Out of Scope' },
  }
  const rm = ROUTE_META[route] || ROUTE_META.rag

  const SCORE_LABELS = {
    faithfulness: 'Faithfulness', completeness: 'Completeness',
    table_accuracy: 'Table Acc.', figure_accuracy: 'Figure Acc.',
    conciseness: 'Conciseness', overall: 'Overall',
  }

  const totalMs = agentTrace?.reduce((s, t) => s + (t.duration_ms || 0), 0)

  const handleCopy = () => {
    if (!answer) return
    navigator.clipboard?.writeText(answer).then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="flex gap-2.5 mb-5 animate-slide-up">
      {/* Avatar */}
      <div className="w-8 h-8 rounded-lg bg-accent-bg border border-accent-bdr flex-shrink-0 flex items-center justify-center text-sm">
        🧠
      </div>

      {/* Card body */}
      <div className="flex-1 bg-surface border border-border rounded-xl rounded-tl-sm px-4 py-3.5 shadow-sm min-w-0">
        {/* Meta row */}
        <div className="flex flex-wrap items-center gap-1.5 mb-3">
          {route && (
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md border ${rm.bg} ${rm.color}`}>
              {rm.label}
            </span>
          )}
          <ConfidenceBadge overall={scores?.overall} />

          {intent && route === 'rag' && (
            <span className="text-[10px] font-semibold text-accent bg-accent-bg border border-accent-bdr rounded-md px-2 py-0.5 font-mono">
              {intent}
            </span>
          )}
          {complexity != null && route === 'rag' && (
            <span className="text-[10px] font-semibold text-secondary bg-surface-2 border border-border rounded-md px-2 py-0.5 font-mono">
              complexity: {complexity}
            </span>
          )}
          {retries > 0 && (
            <span className="text-[10px] font-bold text-warning bg-warning-bg border border-warning/30 rounded-md px-2 py-0.5">
              ↺ {retries} {retries === 1 ? 'retry' : 'retries'}
            </span>
          )}
          {totalMs > 0 && (
            <span className="text-[10px] font-semibold text-muted bg-surface-2 border border-border rounded-md px-2 py-0.5 font-mono">
              {totalMs >= 1000 ? `${(totalMs / 1000).toFixed(1)}s` : `${totalMs}ms`}
            </span>
          )}

          {/* Copy button */}
          {!isStreaming && answer && (
            <button
              onClick={handleCopy}
              title="Copy answer"
              className="ml-auto flex items-center gap-1 px-2 py-0.5 text-[10px] rounded-md border border-border hover:bg-surface-2 transition-colors"
              style={{ color: copied ? 'var(--c-success)' : 'var(--c-text-muted)' }}
            >
              {copied ? <Check size={10} /> : <Copy size={10} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          )}
        </div>

        {/* Rewritten query */}
        {rewritten && rewritten !== message.query && route === 'rag' && (
          <div className="bg-accent-bg border border-accent-bdr rounded-md p-2 mb-3">
            <span className="text-[9px] font-bold text-accent-txt uppercase tracking-wider">
              Rewritten ·
            </span>
            <span className="text-xs text-accent Txt font-mono">
              {' '}
              {rewritten}
            </span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-danger-bg border border-danger/30 rounded-md p-2.5 mb-3">
            <p className="text-xs text-danger">{error}</p>
          </div>
        )}

        {/* Answer text */}
        {isStreaming && !answer ? (
          <div className="flex items-center gap-1 py-1">
            {[0, 1, 2].map(i => (
              <span key={i} className="block w-1.5 h-1.5 rounded-full bg-accent animate-[pulse-dot_1.5s_ease-in-out_infinite]" />
            ))}
            <span className="text-xs text-muted ml-2">Generating…</span>
          </div>
        ) : (
          <p className="text-sm text-primary leading-relaxed whitespace-pre-wrap break-words">
            {answer}
            {isStreaming && <span className="cursor" />}
          </p>
        )}

        {/* Scores */}
        {scores && route === 'rag' && (
          <Expandable
            icon="🪞"
            label="Self-Reflection Scores"
            badge={`${parseFloat(scores.overall ?? 0).toFixed(1)} / 5.0`}
            badgeColor={scoreColor(scores.overall)}
          >
            {Object.entries(SCORE_LABELS).map(([key, label]) => {
              const val = scores[key]
              if (val == null) return null
              return <ScoreBar key={key} label={label} value={parseFloat(val)} />
            })}
            {scores.feedback && (
              <p className="text-[10px] text-secondary italic mt-2 leading-relaxed">
                "{scores.feedback}"
              </p>
            )}
          </Expandable>
        )}

        {/* Sources */}
        {sources?.length > 0 && (
          <Expandable icon="📎" label="Sources" badge={sources.length}>
            <div className="flex flex-col gap-1.5">
              {sources.slice(0, 8).map((src, i) => {
                const sentence = typeof src === 'string' ? src : src.sentence || ''
                const srcList  = typeof src === 'object' ? src.sources || [] : []
                return (
                  <div
                    key={i}
                    className="bg-surface-2 border border-border rounded-md p-2"
                  >
                    {sentence && (
                      <p className="text-xs text-text-2 mb-1 leading-relaxed">{sentence}</p>
                    )}
                    {srcList.length > 0 && (
                      <div className="flex gap-1 flex-wrap">
                        {srcList.map((s, j) => (
                          <span
                            key={j}
                            className="text-[9px] text-accent font-mono bg-accent-bg border border-accent-bdr rounded px-1.5 py-0.25"
                          >
                            {typeof s === 'string' ? s : `chunk_${s}`}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </Expandable>
        )}

        {/* Pipeline trace */}
        {agentTrace?.length > 0 && (
          <Expandable
            icon="🔍"
            label="Pipeline Trace"
            badge={`${agentTrace.length} agents`}
            badgeColor="#8B5CF6"
          >
            <AgentTrace trace={agentTrace} />
          </Expandable>
        )}
      </div>
    </div>
  )
}

// ── Live pipeline strip ────────────────────────────────────────────────────────
function LivePipelineStrip({ agentStates, isRunning, lastRoute }) {
  const hasAny = Object.keys(agentStates || {}).length > 0
  if (!hasAny && !isRunning) return null

  const STATE_COLOR = {
    idle:   '#D4D4D8',
    active: '#6366F1',
    done:   '#059669',
    error:  '#EF4444',
  }

  if (lastRoute === 'unknown') {
    return (
      <div className="px-5 py-2 bg-surface border-b border-border flex items-center gap-2">
        <span className="text-sm">🚫</span>
        <span className="text-xs font-semibold text-secondary">
          Query routed as Out of Scope — no pipeline agents ran
        </span>
      </div>
    )
  }

  return (
    <div className="px-5 py-1.5 bg-surface border-b border-border flex items-center gap-1 flex-shrink-0">
      <span className="text-[10px] font-semibold text-muted mr-1">
        PIPELINE
      </span>
      {AGENTS.map((agent, idx) => {
        const state = agentStates?.[agent.id] || 'idle'
        const color = STATE_COLOR[state]
        return (
          <div key={agent.id} className="flex items-center">
            <div
              title={agent.title}
              className={`
                w-5 h-5 rounded-md flex items-center justify-center
                text-[8px] font-bold font-mono border flex-shrink-0
                ${state === 'idle' ? 'bg-bg border-border text-muted' : 'border'}
              `}
              style={{
                backgroundColor: state === 'idle' ? 'var(--c-bg)' : color + '33',
                borderColor: color,
                color: color,
                animation: state === 'active' ? 'pulse-dot 1.5s ease-in-out infinite' : 'none',
              }}
            >
              {state === 'done' ? '✓' : state === 'error' ? '✗' : agent.number}
            </div>
            {idx < AGENTS.length - 1 && (
              <div
                className="w-1.5 h-px transition-colors"
                style={{ background: state === 'done' ? '#A7F3D0' : 'var(--c-border)' }}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Pipeline step progress (left panel) ───────────────────────────────────────
function PipelineProgress({ agentStates }) {
  const STATE_CONFIG = {
    idle:   { bg: 'bg-bg',       text: 'text-muted',       border: 'border-border' },
    active: { bg: 'bg-accent-bg', text: 'text-accent',     border: 'border-accent-bdr' },
    done:   { bg: 'bg-success-bg', text: 'text-success',    border: 'border-success/30' },
    error:  { bg: 'bg-danger-bg',  text: 'text-danger',     border: 'border-danger/30' },
  }
  return (
    <div>
      <p className="text-[9px] font-bold text-muted uppercase tracking-wider mb-2">
        Pipeline Steps
      </p>
      <div className="flex items-center gap-1 flex-wrap">
        {AGENTS.map((agent, idx) => {
          const state = agentStates?.[agent.id] || 'idle'
          const cfg = STATE_CONFIG[state] || STATE_CONFIG.idle
          return (
            <div key={agent.id} className="flex items-center gap-1">
              <div
                title={agent.title}
                className={`
                  w-6 h-6 rounded-md flex items-center justify-center
                  text-[9px] font-bold font-mono border flex-shrink-0
                  ${cfg.bg} ${cfg.text} ${cfg.border}
                `}
                style={{
                  color: state === 'active' ? agent.color : state === 'done' ? 'var(--c-success)' : state === 'error' ? 'var(--c-danger)' : 'var(--c-text-muted)',
                  animation: state === 'active' ? 'pulse-dot 1.5s ease-in-out infinite' : 'none',
                }}
              >
                {state === 'done' ? '✓' : state === 'error' ? '✗' : agent.number}
              </div>
              {idx < AGENTS.length - 1 && (
                <div
                  className="w-1 h-px"
                  style={{ background: state === 'done' ? '#A7F3D0' : 'var(--c-border)' }}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Multi-doc upload zone ─────────────────────────────────────────────────────
function UploadZone({ docName, docReady, docList, isUploading, uploadError, onFile, onClearAll }) {
  const inputRef       = useRef(null)
  const addInputRef    = useRef(null)

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) onFile(file)
  }

  if (isUploading) {
    return (
      <div className="border border-accent-bdr rounded-lg bg-accent-bg px-3.5 py-3 flex items-center gap-2.5">
        <div className="flex gap-1">
          {[0, 1, 2].map(i => (
            <span key={i} className="block w-1.5 h-1.5 rounded-full bg-accent animate-[pulse-dot_1.5s_ease-in-out_infinite]" />
          ))}
        </div>
        <span className="text-sm font-semibold text-accent">Processing document…</span>
      </div>
    )
  }

  if (docReady && docList?.length > 0) {
    return (
      <div className="flex flex-col gap-2">
        {/* Doc list */}
        {docList.map((name, i) => (
          <div
            key={i}
            className="border border-success/30 rounded-md bg-success-bg px-3 py-2 flex items-center gap-2"
          >
            <span className="text-sm">📄</span>
            <span className="flex-1 text-xs font-semibold text-success truncate">
              {name}
              {i === docList.length - 1 && (
                <span className="ml-1 text-[8px] font-bold bg-success text-white px-1.5 py-0.25 rounded-full">
                  latest
                </span>
              )}
            </span>
          </div>
        ))}

        {/* Actions row */}
        <div className="flex gap-1">
          <button
            onClick={() => addInputRef.current?.click()}
            className="flex-1 text-[10px] font-semibold text-accent bg-accent-bg border border-dashed border-accent-bdr rounded-md py-1.5 cursor-pointer transition-colors hover:bg-accent/15"
          >
            + Add doc
          </button>
          <button
            onClick={onClearAll}
            className="text-[10px] font-semibold text-danger bg-danger-bg border border-danger rounded-md py-1.5 px-2.5 cursor-pointer transition-colors hover:bg-danger/15"
          >
            Clear all
          </button>
        </div>

        {uploadError && (
          <p className="text-[10px] font-medium text-danger">{uploadError}</p>
        )}

        <input
          ref={addInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md"
          className="hidden"
          onChange={e => { if (e.target.files?.[0]) onFile(e.target.files[0]) }}
        />
      </div>
    )
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={e => e.preventDefault()}
      onDrop={handleDrop}
      className={`
        border-2 border-dashed rounded-lg bg-surface-2 px-5 py-7 text-center cursor-pointer
        transition-all text-center
        ${uploadError
          ? 'border-danger'
          : 'border-border hover:border-accent hover:bg-accent-bg'}
      `}
    >
      <div className="text-2xl mb-1.5">📎</div>
      <p className="text-sm font-semibold text-primary mb-0.5">
        Drop a PDF here or click to upload
      </p>
      <p className="text-xs text-muted">
        PDF · DOCX · TXT · MD
      </p>
      {uploadError && (
        <p className="text-xs font-medium text-danger mt-1.5">{uploadError}</p>
      )}
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md"
        className="hidden"
        onChange={e => { if (e.target.files?.[0]) onFile(e.target.files[0]) }}
      />
    </div>
  )
}

// ── Query history dropdown ────────────────────────────────────────────────────
function QueryHistoryDropdown({ history, onSelect, onClose }) {
  if (!history.length) return null
  return (
    <div
      className="absolute top-full left-0 right-0 z-50 bg-surface border border-border rounded-lg shadow-lg overflow-hidden animate-slide-up"
    >
      <div className="px-2.5 py-1.5 border-b border-border-in">
        <span className="text-[10px] font-bold text-muted uppercase tracking-wider">
          Recent queries
        </span>
      </div>
      {history.map((q, i) => (
        <button
          key={i}
          onClick={() => { onSelect(q); onClose() }}
          className="block w-full text-left px-3 py-2 text-xs text-text-2 font-mono truncate hover:bg-bg transition-colors border-b border-border-in last:border-0"
        >
          <span className="text-muted mr-1.5">↑</span>
          {q.length > 80 ? q.slice(0, 80) + '…' : q}
        </button>
      ))}
    </div>
  )
}

// ── Main Playground ────────────────────────────────────────────────────────────
export default function Playground({
  docName, docReady, docList,
  onDocumentReady, onDocumentCleared,
  agentStates, onAgentStateChange, onAgentMetaChange, onResetAll, onRunMetaChange,
  messages, onMessagesChange,
}) {
  const [isUploading,   setIsUploading]   = useState(false)
  const [uploadError,   setUploadError]   = useState(null)
  const [query,         setQuery]         = useState('')
  const [isRunning,     setIsRunning]     = useState(false)
  const [showHistory,   setShowHistory]   = useState(false)
  const [queryHistory,  setQueryHistory]  = useState(() => lsGet('ac_query_history', []))
  const [lastRoute,     setLastRoute]     = useState(null)

  const setMessages = onMessagesChange
  const textareaRef  = useRef(null)
  const scrollAnchor = useRef(null)
  const streamIdRef  = useRef(null)
  const queryAreaRef = useRef(null)

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [query])

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (!showHistory) return
    const handler = (e) => {
      if (!queryAreaRef.current?.contains(e.target)) setShowHistory(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showHistory])

  // ── File upload ────────────────────────────────────────────────────────────
  const handleFile = useCallback(async (file) => {
    setIsUploading(true)
    setUploadError(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      const text = await res.text()
      let data = {}
      try { data = JSON.parse(text) } catch {}
      if (!res.ok) throw new Error(data.detail || text.slice(0, 200) || `HTTP ${res.status}`)
      onDocumentReady(data.doc_name, data.doc_list || [data.doc_name])
      setIsUploading(false)
    } catch (err) {
      setIsUploading(false)
      setUploadError(err.message)
    }
  }, [onDocumentReady])

  const handleClearAll = useCallback(async () => {
    try {
      await fetch('/api/docs', { method: 'DELETE' })
    } catch {}
    onDocumentCleared()
  }, [onDocumentCleared])

  // ── Patch a message ────────────────────────────────────────────────────────
  const patch = useCallback((id, delta) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, ...delta } : m))
  }, [])

  // ── Export conversation ────────────────────────────────────────────────────
  const handleExport = useCallback(() => {
    if (!messages.length) return
    const md = messages.map(m => {
      if (m.type === 'user') return `**You:** ${m.content}\n`
      const ans = m.answer || '(no answer)'
      const scores = m.scores ? `\n\n*Self-reflection: ${m.scores.overall}/5*` : ''
      return `**AC-RAG:** ${ans}${scores}\n`
    }).join('\n---\n\n')

    const blob = new Blob([md], { type: 'text/markdown' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url
    a.download = `ac-rag-conversation-${Date.now()}.md`
    a.click()
    URL.revokeObjectURL(url)
  }, [messages])

  // ── Run pipeline ────────────────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    if (!query.trim() || !docReady || isRunning) return

    const q = query.trim()
    setQuery('')
    setIsRunning(true)
    setLastRoute(null)
    onResetAll()

    const newHistory = [q, ...queryHistory.filter(h => h !== q)].slice(0, 10)
    setQueryHistory(newHistory)
    lsSet('ac_query_history', newHistory)

    const userId = nextId()
    setMessages(prev => [...prev, { id: userId, type: 'user', content: q }])

    const asstId = nextId()
    streamIdRef.current = asstId
    setMessages(prev => [...prev, {
      id: asstId, type: 'assistant', query: q,
      answer: '', isStreaming: true,
      scores: null, sources: [],
      route: null, intent: null, complexity: null,
      rewritten: null, retries: null,
      agentTrace: [], stageLogs: [], error: null,
    }])

    const traceMap = {}

    try {
      const history = messages
        .filter(m => !m.isStreaming && !m.error && (m.content || m.answer))
        .slice(-6)
        .map(m => ({ role: m.type === 'user' ? 'user' : 'assistant', content: m.content || m.answer }))

      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, history }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const ev = JSON.parse(line.slice(6))

            if (ev.type === 'agent_start') {
              onAgentStateChange(ev.agent, 'active')
            }
            else if (ev.type === 'agent_done') {
              onAgentStateChange(ev.agent, 'done')
              onAgentMetaChange(ev.agent, {
                summary: ev.summary || '', output: ev.output, duration_ms: ev.duration_ms,
              })
              traceMap[ev.agent] = {
                agentId: ev.agent, summary: ev.summary || '', duration_ms: ev.duration_ms,
              }
            }
            else if (ev.type === 'start') {
              setLastRoute(ev.route)
              const meta = { route: ev.route, intent: ev.intent, complexity: ev.complexity, rewritten: ev.rewritten, retries: ev.retries, query: q }
              onRunMetaChange?.(meta)
              patch(asstId, {
                route: ev.route, intent: ev.intent, complexity: ev.complexity,
                rewritten: ev.rewritten, retries: ev.retries || 0,
              })
            }
            else if (ev.type === 'token') {
              setMessages(prev => prev.map(m =>
                m.id === asstId ? { ...m, answer: m.answer + ev.text } : m
              ))
            }
            else if (ev.type === 'scores') {
              patch(asstId, { scores: ev.data })
            }
            else if (ev.type === 'sources') {
              patch(asstId, { sources: ev.data })
            }
            else if (ev.type === 'trace') {
              patch(asstId, { stageLogs: ev.data })
            }
            else if (ev.type === 'done') {
              const trace = AGENTS.filter(a => traceMap[a.id]).map(a => traceMap[a.id])
              patch(asstId, { isStreaming: false, agentTrace: trace, error: ev.error || null })
              setIsRunning(false)
            }
            else if (ev.type === 'error') {
              patch(asstId, { isStreaming: false, error: ev.message })
              setIsRunning(false)
            }
          } catch { /* ignore malformed SSE lines */ }
        }
      }
    } catch (err) {
      patch(asstId, { isStreaming: false, error: err.message })
      setIsRunning(false)
      if (/no document loaded/i.test(err.message)) {
        onDocumentCleared()
      }
    }
  }, [query, docReady, isRunning, queryHistory, onResetAll, onAgentStateChange, onAgentMetaChange, onRunMetaChange, onDocumentCleared, patch])

  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') handleRun()
    if (e.key === 'Escape') setShowHistory(false)
  }

  const userCount = messages.filter(m => m.type === 'user').length

  return (
    <div className="flex-1 flex overflow-hidden bg-bg">
      {/* ── LEFT PANEL ─────────────────────────────────────────────-- */}
      <div className="w-72 flex-shrink-0 border-r border-border bg-surface flex flex-col overflow-hidden">
        {/* Setup header */}
        <div className="px-4 py-3.5 border-b border-border flex-shrink-0">
          <span className="text-[10px] font-semibold text-muted uppercase tracking-wider">
            Setup
          </span>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4">
          {/* Document upload */}
          <div>
            <p className="text-[10px] font-semibold text-muted uppercase tracking-wider mb-1.5">
              Documents {docList?.length > 0 && <span className="text-accent">({docList.length})</span>}
            </p>
            <UploadZone
              docName={docName} docReady={docReady} docList={docList}
              isUploading={isUploading} uploadError={uploadError}
              onFile={handleFile} onClearAll={handleClearAll}
            />
          </div>

          {/* Query input */}
          <div>
            <p className="text-[10px] font-semibold text-muted uppercase tracking-wider mb-1.5">
              Query
            </p>
            <div ref={queryAreaRef} className="relative">
              <textarea
                ref={textareaRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="What is AC-RAG's accuracy on the benchmark?"
                disabled={isRunning}
                rows={3}
                className="w-full resize-none overflow-hidden px-3 py-2.5 text-sm rounded-lg
                  bg-surface-2 border border-border text-primary placeholder-muted
                  focus:border-accent focus:bg-surface transition-all outline-none
                  disabled:opacity-50"
                onFocus={() => { if (queryHistory.length) setShowHistory(true) }}
              />
              {showHistory && queryHistory.length > 0 && (
                <QueryHistoryDropdown
                  history={queryHistory}
                  onSelect={q => { setQuery(q); textareaRef.current?.focus() }}
                  onClose={() => setShowHistory(false)}
                />
              )}
            </div>
            <p className="text-[10px] text-text-faint mt-1 text-right">
              ⌘ + Enter to run
            </p>
          </div>

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={!query.trim() || !docReady || isRunning}
            className={`
              w-full py-2.5 text-sm font-semibold rounded-lg
              flex items-center justify-center gap-2 transition-all
              ${(!query.trim() || !docReady || isRunning)
                ? 'bg-text-faint text-white cursor-not-allowed'
                : 'bg-accent hover:bg-accent-hover text-white shadow-sm'}
            `}
          >
            {isRunning ? (
              <>
                <div className="flex gap-1">
                  {[0, 1, 2].map(i => (
                    <span key={i} className="block w-1.5 h-1.5 rounded-full bg-white animate-[pulse-dot_1.5s_ease-in-out_infinite]" />
                  ))}
                </div>
                Running…
              </>
            ) : (
              <>▶ Run Pipeline</>
            )}
          </button>

          {/* Clear conversation */}
          {messages.length > 0 && !isRunning && (
            <button
              onClick={() => setMessages([])}
              className="w-full py-1.5 text-xs font-medium text-secondary hover:text-primary hover:bg-surface-2 border border-border rounded-lg transition-colors"
            >
              🗑 Clear conversation
            </button>
          )}

          {/* Pipeline step progress */}
          <div className="border-t border-border-in pt-3.5">
            <PipelineProgress agentStates={agentStates} />
          </div>

          {isRunning && (
            <div className="bg-accent-bg border border-accent-bdr rounded-lg px-3 py-2">
              <p className="text-xs text-accent">
                💡 Switch to the <strong>Pipeline</strong> tab to inspect each agent live
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── RIGHT PANEL ─────────────────────────────────────────────-- */}
      <div className="flex-1 flex flex-col bg-bg overflow-hidden">
        {/* Chat header */}
        <div className="px-5 py-2.5 border-b border-border bg-surface flex-shrink-0 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-base">💬</span>
            <span className="text-sm font-semibold text-primary">Conversation</span>
            {userCount > 0 && (
              <span className="text-[10px] font-mono text-muted bg-bg px-2 py-0.5 rounded-full">
                {userCount} Q{userCount !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {docName && (
              <span className="text-[10px] text-muted bg-bg border border-border rounded-md px-2 py-0.5 font-mono max-w-[180px] truncate">
                📄 {docName}
              </span>
            )}
            {messages.length > 0 && (
              <button
                onClick={handleExport}
                title="Export conversation as Markdown"
                className="text-xs font-semibold text-secondary hover:text-primary hover:bg-surface-2 border border-border rounded-md px-2.5 py-1 transition-colors"
              >
                ⬇ Export
              </button>
            )}
          </div>
        </div>

        {/* Live pipeline strip */}
        <LivePipelineStrip agentStates={agentStates} isRunning={isRunning} lastRoute={lastRoute} />

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
              <div className="w-14 h-14 rounded-2xl bg-surface border border-border flex items-center justify-center text-2xl mb-4">
                🧠
              </div>
              <p className="text-primary font-medium mb-1">AC-RAG is ready</p>
              <p className="text-secondary text-sm">
                {docReady
                  ? 'Type your question on the left and press ▶ Run Pipeline'
                  : 'Upload a document on the left to get started'}
              </p>

              {docReady && (
                <div className="mt-5 flex gap-2 flex-wrap justify-center">
                  {[
                    'What is AC-RAG?',
                    'Summarize the methodology',
                    'Compare accuracy metrics',
                  ].map(s => (
                    <button
                      key={s}
                      onClick={() => { setQuery(s); textareaRef.current?.focus() }}
                      className="text-xs text-accent bg-accent-bg border border-accent-bdr rounded-full px-3 py-1 cursor-pointer transition-all"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map(msg =>
            msg.type === 'user'
              ? <UserBubble key={msg.id} content={msg.content} />
              : <AssistantCard key={msg.id} message={msg} />
          )}
          <div ref={scrollAnchor} />
        </div>
      </div>
    </div>
  )
}
