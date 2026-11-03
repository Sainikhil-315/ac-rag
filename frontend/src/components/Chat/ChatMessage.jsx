import { useState } from 'react'
import { ChevronDown, ChevronRight, AlertTriangle, User2 } from 'lucide-react'
import { Badge } from '../../ui/Badge'
import PipelineDetailPanel from './PipelineDetailPanel'

export default function ChatMessage({ message }) {
  const [expanded, setExpanded] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[70%] flex items-start gap-2">
          <div className="rounded-2xl bg-accent text-white px-4 py-2.5 text-sm">
            {message.content}
          </div>
          <div className="w-7 h-7 rounded-full bg-surface-2 border border-border flex items-center justify-center flex-shrink-0 mt-0.5">
            <User2 size={13} className="text-muted" />
          </div>
        </div>
      </div>
    )
  }

  const hasDetail = Object.keys(message.agents || {}).length > 0 || message.scores || message.trace

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] w-full">
        <div className="flex items-center gap-2 mb-1.5">
          {message.route && <Badge variant={message.route === 'rag' ? 'primary' : 'default'}>{message.route}</Badge>}
          {message.intent && <Badge variant="default">{message.intent}</Badge>}
          {typeof message.complexity === 'number' && <Badge variant="default">complexity {message.complexity}</Badge>}
          {message.retries > 0 && <Badge variant="warning">{message.retries} retr{message.retries === 1 ? 'y' : 'ies'}</Badge>}
        </div>

        <div
          className="rounded-2xl bg-surface border border-border px-4 py-3 text-sm text-text"
          aria-live="polite"
        >
          {message.error ? (
            <div className="flex items-start gap-2 text-danger">
              <AlertTriangle size={15} className="flex-shrink-0 mt-0.5" />
              <span>{message.error}</span>
            </div>
          ) : (
            <p className="whitespace-pre-wrap leading-relaxed">
              {message.content}
              {message.streaming && <span className="cursor" aria-hidden="true" />}
            </p>
          )}
        </div>

        {hasDetail && (
          <button
            onClick={() => setExpanded(e => !e)}
            className="mt-2 flex items-center gap-1 text-xs text-secondary hover:text-text transition-colors"
          >
            {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            Pipeline Detail
          </button>
        )}

        {expanded && hasDetail && <PipelineDetailPanel message={message} />}
      </div>
    </div>
  )
}
