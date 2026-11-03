import { useState } from 'react'
import * as Icons from 'lucide-react'
import { ChevronDown, ChevronRight, DollarSign, FileSearch, BookMarked } from 'lucide-react'
import { StatusDot } from '../../ui/Badge'
import { AGENTS } from '../../constants/agents'
import ValueView from './ValueView'

function Section({ icon, title, right, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-xl border border-border bg-surface overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3.5 py-2.5 text-left hover:bg-surface-2 transition-colors"
      >
        {open ? <ChevronDown size={13} className="text-muted flex-shrink-0" /> : <ChevronRight size={13} className="text-muted flex-shrink-0" />}
        <span className="text-muted flex-shrink-0">{icon}</span>
        <span className="text-xs font-semibold text-text flex-1 truncate">{title}</span>
        {right}
      </button>
      {open && <div className="px-3.5 pb-3.5 pt-1 border-t border-border">{children}</div>}
    </div>
  )
}

export default function PipelineDetailPanel({ message }) {
  const agents = message.agents || {}

  return (
    <div className="mt-2 space-y-2">
      {AGENTS.map(a => {
        const data = agents[a.id]
        if (!data) return null
        const Icon = Icons[a.icon] || Icons.Circle
        return (
          <Section
            key={a.id}
            icon={<Icon size={14} />}
            title={a.title}
            right={
              <div className="flex items-center gap-2 flex-shrink-0">
                {data.duration_ms != null && <span className="text-[10px] font-mono text-muted">{data.duration_ms}ms</span>}
                <StatusDot status={data.status === 'done' ? 'done' : 'active'} />
              </div>
            }
          >
            <p className="text-[11px] text-secondary mb-2">{data.summary}</p>
            <ValueView value={data.output} />
          </Section>
        )
      })}

      {message.scores && (
        <Section icon={<FileSearch size={14} />} title="Critic Scores" defaultOpen>
          <ValueView value={message.scores} />
        </Section>
      )}

      {message.sources && message.sources.length > 0 && (
        <Section icon={<BookMarked size={14} />} title={`Sources (${message.sources.length})`}>
          <ValueView value={message.sources} />
        </Section>
      )}

      {message.cost && (
        <Section icon={<DollarSign size={14} />} title="Cost / Tokens">
          <ValueView value={message.cost} />
        </Section>
      )}

      {message.trace && message.trace.length > 0 && (
        <Section icon={<Icons.Terminal size={14} />} title={`Raw Stage Log (${message.trace.length} entries)`}>
          <ValueView value={message.trace} />
        </Section>
      )}
    </div>
  )
}
