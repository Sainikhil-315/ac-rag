import { useState, useRef, useEffect, useCallback } from 'react'
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
    <div style={{ display:'flex', justifyContent:'flex-end', marginBottom:'6px', animation:'slideUp 0.2s ease-out' }}>
      <div style={{
        maxWidth:'72%',
        background:'linear-gradient(135deg,#6366F1,#8B5CF6)',
        color:'#FFFFFF', borderRadius:'18px 18px 4px 18px',
        padding:'10px 16px', fontSize:'13px', lineHeight:1.65,
        boxShadow:'0 2px 8px rgba(99,102,241,0.25)', wordBreak:'break-word',
      }}>
        {content}
      </div>
    </div>
  )
}

// ── Collapsible section ───────────────────────────────────────────────────────
function Expandable({ icon, label, badge, badgeColor='#6366F1', defaultOpen=false, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div style={{ borderTop:'1px solid var(--c-border-in)', marginTop:'10px', paddingTop:'8px' }}>
      <button onClick={() => setOpen(o=>!o)} style={{
        display:'flex', alignItems:'center', gap:'6px', width:'100%',
        background:'none', border:'none', cursor:'pointer', padding:'2px 0',
        fontFamily:'Inter, sans-serif', textAlign:'left',
      }}>
        <span style={{ fontSize:'13px' }}>{icon}</span>
        <span style={{ fontSize:'11px', fontWeight:600, color:'var(--c-text-sec)' }}>{label}</span>
        {badge != null && (
          <span style={{
            fontSize:'10px', fontWeight:700, color:badgeColor,
            background:badgeColor+'18', borderRadius:'999px', padding:'1px 7px', fontFamily:'monospace',
          }}>{badge}</span>
        )}
        <span style={{ marginLeft:'auto', fontSize:'10px', color:'var(--c-text-faint)' }}>
          {open ? '▲' : '▼'}
        </span>
      </button>
      {open && <div style={{ marginTop:'8px' }}>{children}</div>}
    </div>
  )
}

// ── Score bar ─────────────────────────────────────────────────────────────────
function ScoreBar({ label, value }) {
  const pct   = value != null ? Math.min((value/5)*100, 100) : 0
  const color = scoreColor(value)
  return (
    <div style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'6px' }}>
      <span style={{ fontSize:'11px', color:'var(--c-text-sec)', width:'100px', flexShrink:0 }}>{label}</span>
      <div style={{ flex:1, height:'5px', background:'var(--c-border-in)', borderRadius:'999px', overflow:'hidden' }}>
        <div style={{ height:'100%', width:`${pct}%`, background:color, borderRadius:'999px', transition:'width 0.5s ease' }} />
      </div>
      <span style={{ fontSize:'11px', fontWeight:700, color, fontFamily:'monospace', width:'28px', textAlign:'right', flexShrink:0 }}>
        {value != null ? parseFloat(value).toFixed(1) : '—'}
      </span>
    </div>
  )
}

// ── Agent pipeline trace ──────────────────────────────────────────────────────
function AgentTrace({ trace }) {
  if (!trace?.length) return null
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'3px' }}>
      {trace.map(({ agentId, summary, duration_ms }) => {
        const agent = AGENTS.find(a => a.id === agentId)
        if (!agent) return null
        return (
          <div key={agentId} style={{
            display:'flex', alignItems:'center', gap:'8px',
            background:'var(--c-surface-2)', borderRadius:'6px',
            border:'1px solid var(--c-border)', padding:'5px 8px',
          }}>
            <span style={{ fontSize:'11px' }}>{agent.icon}</span>
            <span style={{ fontSize:'10px', fontFamily:'monospace', fontWeight:700, color:agent.color, flexShrink:0 }}>{agent.number}</span>
            <span style={{ fontSize:'11px', color:'var(--c-text-2)', fontWeight:600, flexShrink:0 }}>{agent.title}</span>
            {summary && (
              <span style={{ fontSize:'10px', color:'var(--c-text-muted)', flex:1, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
                · {summary}
              </span>
            )}
            {duration_ms != null && (
              <span style={{ fontSize:'10px', color:'var(--c-text-muted)', fontFamily:'monospace', flexShrink:0 }}>
                {duration_ms}ms
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Confidence badge ──────────────────────────────────────────────────────────
function ConfidenceBadge({ overall }) {
  if (overall == null) return null
  const color = scoreColor(overall)
  const label = overall >= 4 ? 'High' : overall >= 3 ? 'Medium' : 'Low'
  return (
    <span title={`Self-reflection score: ${overall}/5`} style={{
      display:'inline-flex', alignItems:'center', gap:'4px',
      background:color+'18', border:`1px solid ${color}44`,
      borderRadius:'999px', padding:'2px 8px',
      fontSize:'10px', fontWeight:700, color, cursor:'default',
    }}>
      <span style={{ width:'6px', height:'6px', borderRadius:'50%', background:color, display:'inline-block' }}/>
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
    rag:     { bg:'#EEF2FF', color:'#4F46E5', label:'⚡ RAG' },
    unknown: { bg:'#F4F4F5', color:'#71717A', label:'🚫 Out of Scope' },
  }
  const rm = ROUTE_META[route] || ROUTE_META.rag

  const SCORE_LABELS = {
    faithfulness:'Faithfulness', completeness:'Completeness',
    table_accuracy:'Table Acc.', figure_accuracy:'Figure Acc.',
    conciseness:'Conciseness', overall:'Overall',
  }

  const totalMs = agentTrace?.reduce((s,t) => s+(t.duration_ms||0), 0)

  const handleCopy = () => {
    if (!answer) return
    navigator.clipboard?.writeText(answer).then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div style={{ display:'flex', gap:'10px', marginBottom:'20px', animation:'slideUp 0.25s ease-out' }}>
      {/* Avatar */}
      <div style={{
        width:'32px', height:'32px', borderRadius:'10px',
        background:'var(--c-accent-bg)', border:'1px solid var(--c-accent-bdr)',
        display:'flex', alignItems:'center', justifyContent:'center',
        fontSize:'15px', flexShrink:0, marginTop:'2px',
      }}>🧠</div>

      {/* Card body */}
      <div style={{
        flex:1, background:'var(--c-surface)',
        border:'1px solid var(--c-border)',
        borderRadius:'4px 16px 16px 16px',
        padding:'14px 16px',
        boxShadow:'0 1px 4px rgba(0,0,0,0.06)',
        minWidth:0,
      }}>

        {/* Meta row */}
        <div style={{ display:'flex', flexWrap:'wrap', gap:'5px', marginBottom:'10px', alignItems:'center' }}>
          {route && (
            <span style={{
              fontSize:'10px', fontWeight:700, color:rm.color,
              background:rm.bg, borderRadius:'5px', padding:'2px 8px',
            }}>{rm.label}</span>
          )}
          {/* Confidence badge — prominent, shown right after route */}
          <ConfidenceBadge overall={scores?.overall} />

          {intent && route === 'rag' && (
            <span style={{
              fontSize:'10px', fontWeight:600, color:'#6366F1',
              background:'#F5F3FF', borderRadius:'5px', padding:'2px 8px', fontFamily:'monospace',
            }}>{intent}</span>
          )}
          {complexity != null && route === 'rag' && (
            <span style={{
              fontSize:'10px', fontWeight:600, color:'var(--c-text-sec)',
              background:'var(--c-surface-2)', borderRadius:'5px', padding:'2px 8px', fontFamily:'monospace',
            }}>complexity: {complexity}</span>
          )}
          {retries > 0 && (
            <span style={{
              fontSize:'10px', fontWeight:700, color:'#D97706',
              background:'#FFF7ED', borderRadius:'5px', padding:'2px 8px',
            }}>↺ {retries} {retries===1?'retry':'retries'}</span>
          )}
          {totalMs > 0 && (
            <span style={{
              fontSize:'10px', fontWeight:600, color:'var(--c-text-muted)',
              background:'var(--c-surface-2)', borderRadius:'5px', padding:'2px 8px', fontFamily:'monospace',
            }}>
              {totalMs >= 1000 ? `${(totalMs/1000).toFixed(1)}s` : `${totalMs}ms`}
            </span>
          )}

          {/* Copy button — far right */}
          {!isStreaming && answer && (
            <button onClick={handleCopy} title="Copy answer" style={{
              marginLeft:'auto', display:'flex', alignItems:'center', gap:'4px',
              background:'none', border:'1px solid var(--c-border)',
              borderRadius:'5px', padding:'2px 8px',
              fontSize:'10px', color: copied ? '#059669' : 'var(--c-text-muted)',
              cursor:'pointer', transition:'all 0.15s', fontFamily:'Inter, sans-serif',
            }}>
              {copied ? '✓ Copied' : '⎘ Copy'}
            </button>
          )}
        </div>

        {/* Rewritten query */}
        {rewritten && rewritten !== message.query && route === 'rag' && (
          <div style={{
            background:'#F5F3FF', border:'1px solid #DDD6FE',
            borderRadius:'6px', padding:'6px 10px', marginBottom:'12px',
          }}>
            <span style={{ fontSize:'10px', fontWeight:700, color:'#8B5CF6', textTransform:'uppercase', letterSpacing:'0.05em' }}>
              Rewritten ·{' '}
            </span>
            <span style={{ fontSize:'12px', color:'#6D28D9', fontFamily:'monospace' }}>{rewritten}</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            background:'var(--c-danger-bg)', border:'1px solid var(--c-danger)',
            borderRadius:'6px', padding:'8px 12px', marginBottom:'12px',
          }}>
            <p style={{ fontSize:'12px', color:'var(--c-danger)', margin:0 }}>{error}</p>
          </div>
        )}

        {/* Answer text */}
        {isStreaming && !answer ? (
          <div style={{ display:'flex', gap:'4px', alignItems:'center', padding:'4px 0' }}>
            {[0,1,2].map(i=>(
              <span key={i} className="running-dot" style={{ background:'#6366F1' }} />
            ))}
            <span style={{ fontSize:'12px', color:'var(--c-text-muted)', marginLeft:'6px' }}>Generating…</span>
          </div>
        ) : (
          <p style={{
            fontSize:'14px', color:'var(--c-text)', lineHeight:1.75,
            margin:0, whiteSpace:'pre-wrap', wordBreak:'break-word',
          }}>
            {answer}
            {isStreaming && <span className="cursor" />}
          </p>
        )}

        {/* Scores */}
        {scores && route === 'rag' && (
          <Expandable icon="🪞" label="Self-Reflection Scores"
            badge={`${parseFloat(scores.overall??0).toFixed(1)} / 5.0`}
            badgeColor={scoreColor(scores.overall)}
          >
            {Object.entries(SCORE_LABELS).map(([key, label]) => {
              const val = scores[key]
              if (val == null) return null
              return <ScoreBar key={key} label={label} value={parseFloat(val)} />
            })}
            {scores.feedback && (
              <p style={{ fontSize:'11px', color:'var(--c-text-sec)', margin:'8px 0 0', fontStyle:'italic', lineHeight:1.5 }}>
                "{scores.feedback}"
              </p>
            )}
          </Expandable>
        )}

        {/* Sources */}
        {sources?.length > 0 && (
          <Expandable icon="📎" label="Sources" badge={sources.length} badgeColor="#6366F1">
            <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
              {sources.slice(0,8).map((src, i) => {
                const sentence = typeof src==='string' ? src : src.sentence||''
                const srcList  = typeof src==='object' ? src.sources||[] : []
                return (
                  <div key={i} style={{
                    background:'var(--c-surface-2)', border:'1px solid var(--c-border)',
                    borderRadius:'6px', padding:'8px 10px',
                  }}>
                    {sentence && <p style={{ fontSize:'12px', color:'var(--c-text-2)', margin:'0 0 5px', lineHeight:1.5 }}>{sentence}</p>}
                    {srcList.length > 0 && (
                      <div style={{ display:'flex', gap:'4px', flexWrap:'wrap' }}>
                        {srcList.map((s,j)=>(
                          <span key={j} style={{
                            fontSize:'10px', color:'#6366F1', fontFamily:'monospace',
                            background:'#EEF2FF', borderRadius:'4px', padding:'1px 6px',
                          }}>
                            {typeof s==='string' ? s : `chunk_${s}`}
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
          <Expandable icon="🔍" label="Pipeline Trace" badge={`${agentTrace.length} agents`} badgeColor="#8B5CF6">
            <AgentTrace trace={agentTrace} />
          </Expandable>
        )}
      </div>
    </div>
  )
}

// ── Live pipeline strip ────────────────────────────────────────────────────────
function LivePipelineStrip({ agentStates, isRunning, lastRoute }) {
  const hasAny = Object.keys(agentStates||{}).length > 0
  if (!hasAny && !isRunning) return null

  const STATE_COLOR = {
    idle:   '#D4D4D8',
    active: '#6366F1',
    done:   '#059669',
    error:  '#EF4444',
  }

  if (lastRoute === 'unknown') {
    return (
      <div style={{
        padding:'8px 20px', background:'var(--c-surface)',
        borderBottom:'1px solid var(--c-border)',
        display:'flex', alignItems:'center', gap:'8px',
      }}>
        <span style={{ fontSize:'12px' }}>🚫</span>
        <span style={{ fontSize:'11px', fontWeight:600, color:'#71717A' }}>
          Query routed as Out of Scope — no pipeline agents ran
        </span>
      </div>
    )
  }

  return (
    <div style={{
      padding:'8px 20px', background:'var(--c-surface)',
      borderBottom:'1px solid var(--c-border)',
      display:'flex', alignItems:'center', gap:'6px', flexShrink:0,
    }}>
      <span style={{ fontSize:'10px', fontWeight:600, color:'var(--c-text-muted)', marginRight:'4px' }}>
        PIPELINE
      </span>
      {AGENTS.map((agent, idx) => {
        const state = agentStates?.[agent.id] || 'idle'
        const color = STATE_COLOR[state]
        return (
          <div key={agent.id} style={{ display:'flex', alignItems:'center', gap:'4px' }}>
            <div title={agent.title} style={{
              width:'22px', height:'22px', borderRadius:'5px',
              background: state==='idle' ? 'var(--c-bg)' : color+'18',
              border:`1px solid ${color}`,
              display:'flex', alignItems:'center', justifyContent:'center',
              fontSize:'9px', fontWeight:700, color,
              fontFamily:'monospace', flexShrink:0,
              animation: state==='active' ? 'pulse-dot 1.5s ease-in-out infinite' : 'none',
              transition:'all 0.2s',
            }}>
              {state==='done' ? '✓' : state==='error' ? '✗' : agent.number}
            </div>
            {idx < AGENTS.length-1 && (
              <div style={{
                width:'8px', height:'1px',
                background: state==='done' ? '#059669' : 'var(--c-border)',
                transition:'background 0.3s',
              }}/>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Pipeline step progress (left panel) ───────────────────────────────────────
function PipelineProgress({ agentStates }) {
  const STATE_COLOR = {
    idle:   { bg:'var(--c-bg)',       text:'var(--c-text-muted)', border:'var(--c-border)' },
    active: { bg:'#EEF2FF',           text:'#6366F1',             border:'#6366F1' },
    done:   { bg:'var(--c-success-bg)', text:'var(--c-success)',  border:'#A7F3D0' },
    error:  { bg:'var(--c-danger-bg)', text:'var(--c-danger)',    border:'#FECACA' },
  }
  return (
    <div>
      <p style={{
        fontSize:'10px', fontWeight:600, color:'var(--c-text-muted)',
        textTransform:'uppercase', letterSpacing:'0.07em', margin:'0 0 8px',
      }}>Pipeline Steps</p>
      <div style={{ display:'flex', alignItems:'center', gap:'4px', flexWrap:'wrap' }}>
        {AGENTS.map((agent, idx) => {
          const state = agentStates?.[agent.id] || 'idle'
          const s = STATE_COLOR[state]
          return (
            <div key={agent.id} style={{ display:'flex', alignItems:'center', gap:'4px' }}>
              <div title={agent.title} style={{
                width:'28px', height:'28px', borderRadius:'6px',
                background:s.bg, border:`1px solid ${s.border}`,
                display:'flex', alignItems:'center', justifyContent:'center',
                fontSize:'10px', fontWeight:700, color:s.text, fontFamily:'monospace', flexShrink:0,
                animation:state==='active'?'pulse-dot 1.5s ease-in-out infinite':'none',
                transition:'all 0.2s',
              }}>
                {state==='done' ? '✓' : state==='error' ? '✗' : agent.number}
              </div>
              {idx < AGENTS.length-1 && (
                <div style={{
                  width:'8px', height:'1px',
                  background:state==='done' ? '#A7F3D0' : 'var(--c-border)',
                  transition:'background 0.3s',
                }}/>
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

  if (isUploading) return (
    <div style={{
      border:'1px solid var(--c-accent-bdr)', borderRadius:'8px',
      background:'var(--c-accent-bg)', padding:'10px 14px',
      display:'flex', alignItems:'center', gap:'10px',
    }}>
      <div style={{ display:'flex', gap:'4px' }}>
        {[0,1,2].map(i => <span key={i} className="running-dot" style={{ background:'#6366F1' }} />)}
      </div>
      <span style={{ fontSize:'12px', color:'#6366F1', fontWeight:600 }}>Processing document…</span>
    </div>
  )

  if (docReady && docList?.length > 0) return (
    <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
      {/* Doc list */}
      {docList.map((name, i) => (
        <div key={i} style={{
          border:'1px solid #A7F3D0', borderRadius:'6px',
          background:'var(--c-success-bg)', padding:'7px 12px',
          display:'flex', alignItems:'center', gap:'8px',
        }}>
          <span style={{ fontSize:'13px' }}>📄</span>
          <span style={{
            flex:1, fontSize:'12px', fontWeight:600, color:'var(--c-success)',
            whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
          }}>
            {name}
            {i === docList.length-1 && (
              <span style={{
                marginLeft:'6px', fontSize:'9px', fontWeight:700,
                background:'#A7F3D0', color:'#065F46', borderRadius:'999px', padding:'1px 6px',
              }}>latest</span>
            )}
          </span>
        </div>
      ))}

      {/* Actions row */}
      <div style={{ display:'flex', gap:'6px' }}>
        <button onClick={() => addInputRef.current?.click()} style={{
          flex:1, fontSize:'11px', fontWeight:600, color:'#6366F1',
          background:'var(--c-accent-bg)', border:'1px dashed var(--c-accent-bdr)',
          borderRadius:'6px', padding:'6px', cursor:'pointer', fontFamily:'Inter, sans-serif',
        }}>+ Add doc</button>
        <button onClick={onClearAll} style={{
          fontSize:'11px', fontWeight:600, color:'var(--c-danger)',
          background:'var(--c-danger-bg)', border:'1px solid var(--c-danger)',
          borderRadius:'6px', padding:'6px 10px', cursor:'pointer', fontFamily:'Inter, sans-serif',
        }}>Clear all</button>
      </div>

      {uploadError && (
        <p style={{ fontSize:'11px', color:'var(--c-danger)', fontWeight:500 }}>{uploadError}</p>
      )}

      <input ref={addInputRef} type="file" accept=".pdf,.docx,.txt,.md"
        style={{ display:'none' }}
        onChange={e => { if (e.target.files?.[0]) onFile(e.target.files[0]) }} />
    </div>
  )

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={e => e.preventDefault()}
      onDrop={handleDrop}
      style={{
        border:`1.5px dashed ${uploadError ? 'var(--c-danger)' : 'var(--c-border)'}`,
        borderRadius:'8px', background:'var(--c-surface-2)',
        padding:'20px 14px', textAlign:'center', cursor:'pointer', transition:'all 0.15s',
      }}
    >
      <div style={{ fontSize:'22px', marginBottom:'6px' }}>📎</div>
      <p style={{ fontSize:'12px', fontWeight:600, color:'var(--c-text-2)', margin:'0 0 3px' }}>
        Drop a PDF here or click to upload
      </p>
      <p style={{ fontSize:'11px', color:'var(--c-text-muted)', margin:0 }}>PDF · DOCX · TXT · MD</p>
      {uploadError && (
        <p style={{ fontSize:'11px', color:'var(--c-danger)', margin:'6px 0 0', fontWeight:500 }}>{uploadError}</p>
      )}
      <input ref={inputRef} type="file" accept=".pdf,.docx,.txt,.md"
        style={{ display:'none' }}
        onChange={e => { if (e.target.files?.[0]) onFile(e.target.files[0]) }} />
    </div>
  )
}

// ── Query history dropdown ────────────────────────────────────────────────────
function QueryHistoryDropdown({ history, onSelect, onClose }) {
  if (!history.length) return null
  return (
    <div style={{
      position:'absolute', top:'calc(100% + 4px)', left:0, right:0, zIndex:100,
      background:'var(--c-surface)', border:'1px solid var(--c-border)',
      borderRadius:'8px', boxShadow:'0 8px 24px rgba(0,0,0,0.12)',
      overflow:'hidden', animation:'slideUp 0.15s ease-out',
    }}>
      <div style={{ padding:'6px 10px', borderBottom:'1px solid var(--c-border-in)' }}>
        <span style={{ fontSize:'10px', fontWeight:600, color:'var(--c-text-muted)', textTransform:'uppercase', letterSpacing:'0.06em' }}>
          Recent queries
        </span>
      </div>
      {history.map((q, i) => (
        <button key={i} onClick={() => { onSelect(q); onClose() }} style={{
          display:'block', width:'100%', textAlign:'left',
          padding:'8px 12px', fontSize:'12px', color:'var(--c-text-2)',
          background:'none', border:'none', cursor:'pointer',
          borderBottom: i < history.length-1 ? '1px solid var(--c-border-in)' : 'none',
          fontFamily:'Inter, sans-serif', lineHeight:1.4,
          transition:'background 0.1s',
        }}
          onMouseEnter={e => e.currentTarget.style.background='var(--c-bg)'}
          onMouseLeave={e => e.currentTarget.style.background='none'}
        >
          <span style={{ color:'var(--c-text-muted)', marginRight:'6px' }}>↑</span>
          {q.length > 80 ? q.slice(0,80)+'…' : q}
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
    el.style.height = Math.min(el.scrollHeight, 160)+'px'
  }, [query])

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior:'smooth' })
  }, [messages])

  // Close history dropdown on outside click
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
      const res  = await fetch('/api/upload', { method:'POST', body:form })
      const text = await res.text()
      let data = {}
      try { data = JSON.parse(text) } catch {}
      if (!res.ok) throw new Error(data.detail || text.slice(0,200) || `HTTP ${res.status}`)
      onDocumentReady(data.doc_name, data.doc_list || [data.doc_name])
      setIsUploading(false)
    } catch (err) {
      setIsUploading(false)
      setUploadError(err.message)
    }
  }, [onDocumentReady])

  const handleClearAll = useCallback(async () => {
    try {
      await fetch('/api/docs', { method:'DELETE' })
    } catch {}
    onDocumentCleared()
  }, [onDocumentCleared])

  // ── Patch a message ────────────────────────────────────────────────────────
  const patch = useCallback((id, delta) => {
    setMessages(prev => prev.map(m => m.id===id ? { ...m, ...delta } : m))
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

    const blob = new Blob([md], { type:'text/markdown' })
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

    // Save to query history
    const newHistory = [q, ...queryHistory.filter(h => h !== q)].slice(0,10)
    setQueryHistory(newHistory)
    lsSet('ac_query_history', newHistory)

    const userId = nextId()
    setMessages(prev => [...prev, { id:userId, type:'user', content:q }])

    const asstId = nextId()
    streamIdRef.current = asstId
    setMessages(prev => [...prev, {
      id:asstId, type:'assistant', query:q,
      answer:'', isStreaming:true,
      scores:null, sources:[],
      route:null, intent:null, complexity:null,
      rewritten:null, retries:null,
      agentTrace:[], stageLogs:[], error:null,
    }])

    const traceMap = {}

    try {
      // Last few completed turns, for follow-up/pronoun resolution server-side
      // (query_analyzer.py caps how many it actually uses — safe to send more).
      const history = messages
        .filter(m => !m.isStreaming && !m.error && (m.content || m.answer))
        .slice(-6)
        .map(m => ({ role: m.type === 'user' ? 'user' : 'assistant', content: m.content || m.answer }))

      const res = await fetch('/api/ask', {
        method:'POST',
        headers:{ 'Content-Type':'application/json' },
        body:JSON.stringify({ query:q, history }),
      })
      if (!res.ok) {
        const data = await res.json().catch(()=>({}))
        throw new Error(data.detail || `HTTP ${res.status}`)
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream:true })
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
                summary:ev.summary||'', output:ev.output, duration_ms:ev.duration_ms,
              })
              traceMap[ev.agent] = {
                agentId:ev.agent, summary:ev.summary||'', duration_ms:ev.duration_ms,
              }
            }

            else if (ev.type === 'start') {
              setLastRoute(ev.route)
              const meta = { route:ev.route, intent:ev.intent, complexity:ev.complexity, rewritten:ev.rewritten, retries:ev.retries, query:q }
              onRunMetaChange?.(meta)
              patch(asstId, {
                route:ev.route, intent:ev.intent, complexity:ev.complexity,
                rewritten:ev.rewritten, retries:ev.retries||0,
              })
            }

            else if (ev.type === 'token') {
              setMessages(prev => prev.map(m =>
                m.id===asstId ? { ...m, answer:m.answer+ev.text } : m
              ))
            }

            else if (ev.type === 'scores') {
              patch(asstId, { scores:ev.data })
            }

            else if (ev.type === 'sources') {
              patch(asstId, { sources:ev.data })
            }

            else if (ev.type === 'trace') {
              patch(asstId, { stageLogs:ev.data })
            }

            else if (ev.type === 'done') {
              const trace = AGENTS.filter(a => traceMap[a.id]).map(a => traceMap[a.id])
              patch(asstId, { isStreaming:false, agentTrace:trace, error:ev.error||null })
              setIsRunning(false)
            }

            else if (ev.type === 'error') {
              patch(asstId, { isStreaming:false, error:ev.message })
              setIsRunning(false)
            }

          } catch { /* ignore malformed SSE lines */ }
        }
      }
    } catch (err) {
      patch(asstId, { isStreaming:false, error:err.message })
      setIsRunning(false)
      // Backend restarts (e.g. dev server auto-reload) wipe its in-memory document
      // state, but this client's localStorage still says a doc is loaded — self-heal
      // instead of leaving the UI stuck showing a document that no longer exists
      // server-side (was previously a confusing dead-end, see 2026-08-20 bug report).
      if (/no document loaded/i.test(err.message)) {
        onDocumentCleared()
      }
    }
  }, [query, docReady, isRunning, queryHistory, onResetAll, onAgentStateChange, onAgentMetaChange, onRunMetaChange, onDocumentCleared, patch])

  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key==='Enter') handleRun()
    if (e.key === 'Escape') setShowHistory(false)
  }

  const userCount = messages.filter(m => m.type==='user').length

  return (
    <div style={{ flex:1, display:'flex', overflow:'hidden', background:'var(--c-bg)' }}>

      {/* ── LEFT PANEL ─────────────────────────────────────────────── */}
      <div style={{
        width:'310px', flexShrink:0,
        borderRight:'1px solid var(--c-border)',
        background:'var(--c-surface)',
        display:'flex', flexDirection:'column', overflow:'hidden',
      }}>
        <div style={{ padding:'14px 16px 12px', borderBottom:'1px solid var(--c-border)', flexShrink:0 }}>
          <span style={{ fontSize:'11px', fontWeight:600, color:'var(--c-text-muted)', textTransform:'uppercase', letterSpacing:'0.07em' }}>
            Setup
          </span>
        </div>

        <div style={{ flex:1, overflowY:'auto', padding:'14px', display:'flex', flexDirection:'column', gap:'14px' }}>

          {/* Document upload */}
          <div>
            <p style={{ fontSize:'10px', fontWeight:600, color:'var(--c-text-muted)', textTransform:'uppercase', letterSpacing:'0.07em', margin:'0 0 7px' }}>
              Documents {docList?.length > 0 && <span style={{ color:'var(--c-accent)' }}>({docList.length})</span>}
            </p>
            <UploadZone
              docName={docName} docReady={docReady} docList={docList}
              isUploading={isUploading} uploadError={uploadError}
              onFile={handleFile} onClearAll={handleClearAll}
            />
          </div>

          {/* Query textarea with history */}
          <div>
            <p style={{ fontSize:'10px', fontWeight:600, color:'var(--c-text-muted)', textTransform:'uppercase', letterSpacing:'0.07em', margin:'0 0 7px' }}>
              Query
            </p>
            <div ref={queryAreaRef} style={{ position:'relative' }}>
              <textarea
                ref={textareaRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your document…"
                disabled={isRunning}
                rows={3}
                style={{
                  width:'100%', boxSizing:'border-box',
                  resize:'none', overflow:'hidden',
                  padding:'10px 12px', fontSize:'13px',
                  color:'var(--c-text)', background:'var(--c-surface-2)',
                  border:'1px solid var(--c-border)', borderRadius:'8px',
                  outline:'none', fontFamily:'Inter, sans-serif', lineHeight:1.6,
                  transition:'border-color 0.15s',
                }}
                onFocus={e => { e.target.style.borderColor='#6366F1'; if (queryHistory.length) setShowHistory(true) }}
                onBlur={e => { e.target.style.borderColor='var(--c-border)' }}
              />
              {showHistory && queryHistory.length > 0 && (
                <QueryHistoryDropdown
                  history={queryHistory}
                  onSelect={q => { setQuery(q); textareaRef.current?.focus() }}
                  onClose={() => setShowHistory(false)}
                />
              )}
            </div>
            <p style={{ fontSize:'10px', color:'var(--c-text-faint)', margin:'4px 0 0', textAlign:'right' }}>
              ⌘ + Enter to run
            </p>
          </div>

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={!query.trim() || !docReady || isRunning}
            style={{
              width:'100%', padding:'10px',
              fontSize:'13px', fontWeight:600, color:'#FFFFFF',
              background: (!query.trim()||!docReady||isRunning) ? 'var(--c-text-faint)' : '#6366F1',
              border:'none', borderRadius:'8px',
              cursor: (!query.trim()||!docReady||isRunning) ? 'not-allowed' : 'pointer',
              display:'flex', alignItems:'center', justifyContent:'center', gap:'8px',
              transition:'background 0.15s', fontFamily:'Inter, sans-serif',
            }}
          >
            {isRunning ? (
              <><div style={{ display:'flex', gap:'3px' }}>{[0,1,2].map(i=><span key={i} className="running-dot" style={{ background:'#FFFFFF' }}/>)}</div>Running…</>
            ) : <>▶ Run Pipeline</>}
          </button>

          {/* Clear conversation */}
          {messages.length > 0 && !isRunning && (
            <button
              onClick={() => setMessages([])}
              style={{
                width:'100%', padding:'7px',
                fontSize:'12px', fontWeight:500, color:'var(--c-text-sec)',
                background:'transparent', border:'1px solid var(--c-border)',
                borderRadius:'8px', cursor:'pointer',
                fontFamily:'Inter, sans-serif', transition:'all 0.15s',
              }}
              onMouseOver={e => { e.currentTarget.style.borderColor='var(--c-text-muted)'; e.currentTarget.style.color='var(--c-text)' }}
              onMouseOut={e => { e.currentTarget.style.borderColor='var(--c-border)'; e.currentTarget.style.color='var(--c-text-sec)' }}
            >
              🗑 Clear conversation
            </button>
          )}

          {/* Pipeline step progress */}
          <div style={{ borderTop:'1px solid var(--c-border-in)', paddingTop:'14px' }}>
            <PipelineProgress agentStates={agentStates} />
          </div>

          {isRunning && (
            <div style={{ background:'var(--c-accent-bg)', border:'1px solid var(--c-accent-bdr)', borderRadius:'8px', padding:'10px 12px' }}>
              <p style={{ fontSize:'11px', color:'var(--c-accent)', margin:0, lineHeight:1.5 }}>
                💡 Switch to the <strong>Pipeline</strong> tab to inspect each agent live
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── RIGHT PANEL ────────────────────────────────────────────── */}
      <div style={{ flex:1, display:'flex', flexDirection:'column', background:'var(--c-bg)', overflow:'hidden' }}>

        {/* Chat header */}
        <div style={{
          padding:'11px 20px', borderBottom:'1px solid var(--c-border)',
          background:'var(--c-surface)', flexShrink:0,
          display:'flex', alignItems:'center', justifyContent:'space-between',
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
            <span style={{ fontSize:'15px' }}>💬</span>
            <span style={{ fontSize:'13px', fontWeight:600, color:'var(--c-text)' }}>Conversation</span>
            {userCount > 0 && (
              <span style={{
                fontSize:'10px', fontFamily:'monospace', color:'var(--c-text-muted)',
                background:'var(--c-bg)', borderRadius:'999px', padding:'1px 8px',
              }}>{userCount} Q{userCount!==1?'s':''}</span>
            )}
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
            {docName && (
              <span style={{
                fontSize:'10px', color:'var(--c-text-muted)',
                background:'var(--c-bg)', border:'1px solid var(--c-border)',
                borderRadius:'6px', padding:'2px 8px', fontFamily:'monospace',
                maxWidth:'200px', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap',
              }}>
                📄 {docName}
              </span>
            )}
            {messages.length > 0 && (
              <button onClick={handleExport} title="Export conversation as Markdown"
                style={{
                  fontSize:'11px', fontWeight:600, color:'var(--c-text-sec)',
                  background:'var(--c-bg)', border:'1px solid var(--c-border)',
                  borderRadius:'6px', padding:'3px 10px',
                  cursor:'pointer', fontFamily:'Inter, sans-serif', transition:'all 0.15s',
                }}
                onMouseOver={e => { e.currentTarget.style.color='var(--c-text)'; e.currentTarget.style.borderColor='var(--c-text-muted)' }}
                onMouseOut={e => { e.currentTarget.style.color='var(--c-text-sec)'; e.currentTarget.style.borderColor='var(--c-border)' }}
              >
                ⬇ Export
              </button>
            )}
          </div>
        </div>

        {/* Live pipeline strip */}
        <LivePipelineStrip agentStates={agentStates} isRunning={isRunning} lastRoute={lastRoute} />

        {/* Messages */}
        <div style={{ flex:1, overflowY:'auto', padding:'20px 24px' }}>

          {messages.length === 0 && (
            <div style={{
              display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
              minHeight:'60%', gap:'14px', opacity:0.75,
            }}>
              <div style={{
                width:'56px', height:'56px', borderRadius:'18px',
                background:'var(--c-surface)', border:'1px solid var(--c-border)',
                display:'flex', alignItems:'center', justifyContent:'center', fontSize:'26px',
                boxShadow:'0 2px 8px rgba(0,0,0,0.06)',
              }}>🧠</div>
              <div style={{ textAlign:'center' }}>
                <p style={{ fontSize:'15px', fontWeight:700, color:'var(--c-text)', margin:'0 0 6px' }}>AC-RAG ready</p>
                <p style={{ fontSize:'13px', color:'var(--c-text-muted)', margin:0, lineHeight:1.6 }}>
                  {docReady
                    ? 'Type your question on the left and press ▶ Run Pipeline'
                    : 'Upload a document on the left to get started'}
                </p>
              </div>
              <div style={{ display:'flex', gap:'8px', flexWrap:'wrap', justifyContent:'center' }}>
                {['What is AC-RAG?', 'Summarize the methodology', 'Compare accuracy metrics'].map(s=>(
                  <button key={s} disabled={!docReady||isRunning}
                    onClick={() => { if (docReady&&!isRunning) { setQuery(s); textareaRef.current?.focus() } }}
                    style={{
                      fontSize:'11px', color:'#6366F1',
                      background:'var(--c-accent-bg)', border:'1px solid var(--c-accent-bdr)',
                      borderRadius:'999px', padding:'4px 12px',
                      cursor: docReady ? 'pointer' : 'default',
                      fontFamily:'Inter, sans-serif', opacity: docReady ? 1 : 0.5,
                    }}
                  >{s}</button>
                ))}
              </div>
            </div>
          )}

          {messages.map(msg =>
            msg.type==='user'
              ? <UserBubble key={msg.id} content={msg.content} />
              : <AssistantCard key={msg.id} message={msg} />
          )}
          <div ref={scrollAnchor} />
        </div>
      </div>
    </div>
  )
}
