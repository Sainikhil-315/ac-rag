import { useState, useRef, useEffect } from 'react'
import { FileText, RefreshCw, Send } from 'lucide-react'
import UserMessage from './UserMessage'
import AssistantMessage from './AssistantMessage'

export default function Chat({ messages, docName, onAsk, onChange }) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef()
  const inputRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const isStreaming = messages.some(m => m.role === 'assistant' && m.isStreaming)

  const submit = async () => {
    const q = input.trim()
    if (!q || isStreaming) return
    setInput('')
    setLoading(true)
    await onAsk(q)
    setLoading(false)
    inputRef.current?.focus()
  }

  const onKey = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }

  return (
    <div className="h-screen bg-bg flex flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-border flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-accent/20 border border-accent/30 flex items-center justify-center text-sm">🧠</div>
          <span className="font-semibold text-primary tracking-tight">AC-RAG</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-surface border border-border rounded-lg px-3 py-1.5">
            <FileText size={13} className="text-secondary" />
            <span className="text-sm text-secondary max-w-[200px] truncate">{docName}</span>
          </div>
          <button onClick={onChange}
            className="flex items-center gap-1.5 text-sm text-secondary hover:text-primary transition-colors px-3 py-1.5 rounded-lg hover:bg-surface border border-transparent hover:border-border">
            <RefreshCw size={13} />
            Change
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
            <div className="w-12 h-12 rounded-2xl bg-surface border border-border flex items-center justify-center text-xl mb-4">💬</div>
            <p className="text-primary font-medium mb-1">Ask about your document</p>
            <p className="text-secondary text-sm">The pipeline will route and answer your question</p>
          </div>
        )}
        {messages.map(msg =>
          msg.role === 'user'
            ? <UserMessage key={msg.id} content={msg.content} />
            : <AssistantMessage key={msg.id} message={msg} />
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-6 pb-6 flex-shrink-0">
        <div className="flex gap-3 bg-surface border border-border rounded-2xl p-3 focus-within:border-accent/50 transition-colors">
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ask a question about your document…"
            disabled={isStreaming}
            className="flex-1 bg-transparent text-primary placeholder-muted text-sm outline-none disabled:opacity-50"
          />
          <button onClick={submit} disabled={!input.trim() || isStreaming}
            className="w-8 h-8 rounded-xl bg-accent hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center flex-shrink-0 transition-colors">
            <Send size={14} className="text-white" />
          </button>
        </div>
        <p className="text-center text-muted text-xs mt-2">
          Powered by AC-RAG · {docName}
        </p>
      </div>
    </div>
  )
}
