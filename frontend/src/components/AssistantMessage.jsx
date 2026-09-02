import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import ScoreCards from './ScoreCards'
import PipelineTrace from './PipelineTrace'

const ROUTE_STYLES = {
  rag:     { label: 'RAG Pipeline', className: 'bg-accent/15 text-accent border-accent/25' },
  unknown: { label: 'Out of Scope', className: 'bg-surface-2/80 text-muted border-border' },
}

const INTENT_ICONS = { factual: '🔍', analytical: '📊', comparative: '⚖️', summarization: '📝' }

export default function AssistantMessage({ message }) {
  const { route, intent, complexity, rewritten, text, isStreaming, scores, sources, trace, error, retries } = message
  const [showSources, setShowSources] = useState(false)
  const [showTrace, setShowTrace] = useState(false)

  const routeStyle = ROUTE_STYLES[route] || ROUTE_STYLES.rag

  return (
    <div className="flex flex-col gap-2 animate-slide-up max-w-[85%]">
      {/* Main card */}
      <div className="bg-surface border border-border rounded-2xl rounded-tl-sm p-5">
        {/* Meta row */}
        {route && (
          <div className="flex items-center gap-2 mb-4 flex-wrap">
            <span className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border tracking-wide uppercase ${routeStyle.className}`}>
              {routeStyle.label}
            </span>
            {intent && (
              <span className="text-[11px] text-secondary flex items-center gap-1">
                <span>{INTENT_ICONS[intent] || '•'}</span> {intent}
              </span>
            )}
            {retries > 0 && (
              <span className="text-[11px] text-muted ml-auto">↺ {retries} {retries === 1 ? 'retry' : 'retries'}</span>
            )}
          </div>
        )}

        {/* Complexity bar */}
        {complexity != null && route === 'rag' && (
          <div className="flex items-center gap-3 mb-4">
            <span className="text-[11px] text-muted w-20">Complexity</span>
            <div className="flex-1 h-1 bg-surface-2 rounded-full overflow-hidden">
              <div className="h-full bg-accent rounded-full transition-all duration-700" style={{ width: `${complexity * 100}%` }} />
            </div>
            <span className="text-[11px] text-secondary font-mono w-8 text-right">{complexity.toFixed(2)}</span>
          </div>
        )}

        {/* Rewrite hint */}
        {rewritten && (
          <div className="flex gap-2 mb-4 p-2.5 bg-surface-2 rounded-lg border-l-2 border-accent/50">
            <span className="text-[11px] text-muted flex-shrink-0 mt-0.5">Rewritten</span>
            <span className="text-[11px] text-secondary leading-relaxed">{rewritten}</span>
          </div>
        )}

        {/* Answer text */}
        <div className="text-sm text-primary leading-relaxed">
          {text}
          {isStreaming && <span className="cursor" />}
          {!isStreaming && !text && <span className="text-muted italic">No answer generated.</span>}
        </div>

        {/* Scores */}
        {!isStreaming && scores && (
          <div className="mt-5 pt-5 border-t border-border">
            <ScoreCards scores={scores} />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-4 p-3 bg-danger/10 border border-danger/20 rounded-lg">
            <p className="text-xs text-danger">{error}</p>
          </div>
        )}
      </div>

      {/* Expand buttons */}
      {!isStreaming && (sources?.length > 0 || trace?.length > 0) && (
        <div className="flex gap-2 px-1">
          {sources?.length > 0 && (
            <button onClick={() => setShowSources(s => !s)}
              className="flex items-center gap-1.5 text-xs text-secondary hover:text-primary transition-colors py-1">
              {showSources ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              Sources ({sources.length})
            </button>
          )}
          {trace?.length > 0 && (
            <button onClick={() => setShowTrace(s => !s)}
              className="flex items-center gap-1.5 text-xs text-secondary hover:text-primary transition-colors py-1 ml-2">
              {showTrace ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              Pipeline trace ({trace.length} stages)
            </button>
          )}
        </div>
      )}

      {/* Sources panel */}
      {showSources && sources?.length > 0 && (
        <div className="bg-surface border border-border rounded-xl p-4 space-y-3 animate-slide-up">
          {sources.map((entry, i) => {
            const srcs = (entry.sources || []).map(s => `${s.file || '?'} p.${s.page || '?'}`).join(' · ')
            return (
              <div key={i} className="border-b border-border pb-3 last:border-0 last:pb-0">
                <p className="text-[11px] text-accent font-mono mb-1">{srcs || 'uncited'}</p>
                <p className="text-xs text-secondary leading-relaxed">{entry.sentence}</p>
              </div>
            )
          })}
        </div>
      )}

      {/* Pipeline trace */}
      {showTrace && trace?.length > 0 && (
        <PipelineTrace stages={trace} />
      )}
    </div>
  )
}
