import { Plus, MessageSquare, Trash2 } from 'lucide-react'

function titleFor(session) {
  const firstUser = session.messages.find(m => m.role === 'user')
  if (!firstUser) return 'New chat'
  return firstUser.content.length > 40 ? firstUser.content.slice(0, 40) + '…' : firstUser.content
}

export default function ChatList({ sessions, activeId, onSelect, onNew, onDelete }) {
  return (
    <aside className="w-64 flex-shrink-0 border-r border-border bg-surface flex flex-col">
      <div className="p-3">
        <button
          onClick={onNew}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors"
        >
          <Plus size={15} /> New chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {sessions.length === 0 && (
          <p className="text-xs text-muted text-center mt-6 px-3">No chats yet — start one above.</p>
        )}
        {sessions.map(s => {
          const active = s.id === activeId
          return (
            <div
              key={s.id}
              onClick={() => onSelect(s.id)}
              className={`group flex items-center gap-2 px-3 py-2 rounded-lg text-sm cursor-pointer transition-colors
                ${active ? 'bg-accent-bg text-accent-text border border-accent-bdr' : 'text-secondary hover:bg-surface-2 border border-transparent'}`}
            >
              <MessageSquare size={14} className="flex-shrink-0 opacity-70" />
              <span className="flex-1 truncate">{titleFor(s)}</span>
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(s.id) }}
                className="opacity-0 group-hover:opacity-100 text-muted hover:text-danger transition-opacity flex-shrink-0"
                aria-label="Delete chat"
              >
                <Trash2 size={13} />
              </button>
            </div>
          )
        })}
      </div>

      <div className="p-3 border-t border-border">
        <p className="text-[10px] text-muted leading-relaxed">
          Chats are session-local (this browser only) — not saved to the server.
        </p>
      </div>
    </aside>
  )
}
