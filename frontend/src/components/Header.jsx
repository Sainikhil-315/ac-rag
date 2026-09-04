import { useTheme } from '../ThemeContext'
import { Toggle } from '../ui/Toggle'

const TABS = [
  { id: 'Pipeline', label: 'Pipeline' },
  { id: 'Playground', label: 'Playground' },
  { id: 'Results', label: 'Results' },
]

export default function Header({ activeTab, onTabChange }) {
  const { isDark, toggle } = useTheme()

  return (
    <header className="h-14 border-b border-border bg-surface flex-shrink-0">
      <div className="h-full max-w-[1400px] mx-auto flex items-center justify-between px-5">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-accent-bg border border-accent-bdr flex items-center justify-center text-sm">
            🧠
          </div>
          <span className="text-sm font-semibold tracking-tight text-primary">
            AC-RAG
          </span>
          <span className="text-[10px] font-medium text-accent bg-accent-bg border border-accent-bdr rounded-full px-2 py-0.5 tracking-wide">
            RESEARCH
          </span>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1 bg-surface-2 rounded-lg p-1 border border-border">
          {TABS.map(({ id, label }) => {
            const active = activeTab === id
            return (
              <button
                key={id}
                onClick={() => onTabChange(id)}
                className={`
                  px-3.5 py-1.5 text-xs font-medium rounded-md transition-all
                  ${active
                    ? 'bg-surface text-primary shadow-sm border border-border'
                    : 'text-secondary hover:text-primary'}
                `}
              >
                {label}
              </button>
            )
          })}
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3">
          <Toggle isDark={isDark} onToggle={toggle} />
          <span className="text-[11px] text-muted font-mono">
            SRKR · 2025–26
          </span>
        </div>
      </div>
    </header>
  )
}
