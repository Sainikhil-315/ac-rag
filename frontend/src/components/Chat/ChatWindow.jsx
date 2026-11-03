import { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'
import ChatMessage from './ChatMessage'

let idCounter = 0
const nextId = () => `m${Date.now()}_${idCounter++}`

/** Parse one SSE event into a partial patch for the in-progress assistant message. */
function patchFromEvent(prev, evt) {
  switch (evt.type) {
    case 'start':
      return { route: evt.route, intent: evt.intent, complexity: evt.complexity, rewritten: evt.rewritten, retries: evt.retries }
    case 'agent_start':
      return { agents: { ...prev.agents, [evt.agent]: { ...(prev.agents[evt.agent] || {}), status: 'active' } } }
    case 'agent_done':
      return {
        agents: {
          ...prev.agents,
          [evt.agent]: { status: 'done', output: evt.output, summary: evt.summary, duration_ms: evt.duration_ms },
        },
      }
    case 'token':
      return { content: prev.content + evt.text }
    case 'scores':
      return { scores: evt.data }
    case 'sources':
      return { sources: evt.data }
    case 'trace':
      return { trace: evt.data }
    case 'cost':
      return { cost: evt.data }
    case 'error':
      return { error: evt.message, streaming: false }
    case 'done':
      return { streaming: false, error: evt.error || prev.error }
    default:
      return {}
  }
}

export default function ChatWindow({ session, onUpdateSession, docReady }) {
  const [input, setInput] = useState('')
  const scrollRef = useRef(null)
  const streamingRef = useRef(false)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [session.messages])

  const updateMessage = (id, patcher) => {
    onUpdateSession(s => ({
      ...s,
      messages: s.messages.map(m => (m.id === id ? { ...m, ...patcher(m) } : m)),
    }))
  }

  const send = async () => {
    const query = input.trim()
    if (!query || streamingRef.current) return
    if (!docReady) return

    const userMsg = { id: nextId(), role: 'user', content: query }
    const assistantId = nextId()
    const assistantMsg = {
      id: assistantId, role: 'assistant', content: '', agents: {}, streaming: true,
      route: null, intent: null, complexity: null, rewritten: null, retries: 0,
      scores: null, sources: null, trace: null, cost: null, error: null,
    }

    const history = session.messages.map(m => ({ role: m.role, content: m.content }))
    onUpdateSession(s => ({ ...s, messages: [...s.messages, userMsg, assistantMsg] }))
    setInput('')
    streamingRef.current = true

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history }),
      })
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}))
        updateMessage(assistantId, () => ({ error: data.detail || `Request failed (${res.status})`, streaming: false }))
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buffer.indexOf('\n\n')) >= 0) {
          const raw = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          if (!raw.startsWith('data: ')) continue
          try {
            const evt = JSON.parse(raw.slice(6))
            updateMessage(assistantId, (prev) => patchFromEvent(prev, evt))
          } catch {
            // malformed SSE chunk — skip rather than crash the stream
          }
        }
      }
    } catch (e) {
      updateMessage(assistantId, () => ({ error: e.message || 'Connection error', streaming: false }))
    } finally {
      streamingRef.current = false
      updateMessage(assistantId, (prev) => ({ streaming: false }))
    }
  }

  return (
    <main className="flex-1 flex flex-col overflow-hidden bg-bg">
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
        {session.messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <p className="text-sm text-secondary">
              {docReady ? 'Ask a question about the loaded document(s).' : 'Upload a document on the right to begin.'}
            </p>
          </div>
        )}
        {session.messages.map(m => <ChatMessage key={m.id} message={m} />)}
      </div>

      <div className="border-t border-border bg-surface p-4">
        <div className="flex items-end gap-2 max-w-3xl mx-auto">
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder={docReady ? 'Ask a question…' : 'Upload a document first…'}
            disabled={!docReady}
            className="flex-1 resize-none rounded-xl border border-border bg-surface-2 px-4 py-2.5 text-sm text-text placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
          />
          <button
            onClick={send}
            disabled={!docReady || !input.trim() || streamingRef.current}
            className="h-10 w-10 flex-shrink-0 flex items-center justify-center rounded-xl bg-accent text-white hover:bg-accent-hover disabled:opacity-40 disabled:pointer-events-none transition-colors"
            aria-label="Send"
          >
            {streamingRef.current ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
          </button>
        </div>
      </div>
    </main>
  )
}
