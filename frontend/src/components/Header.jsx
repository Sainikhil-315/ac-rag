import { BrainCircuit } from 'lucide-react'
import { useTheme } from '../ThemeContext'
import { Toggle } from '../ui/Toggle'
import { Badge } from '../ui/Badge'

export default function Header() {
  const { isDark, toggle } = useTheme()

  return (
    <header className="h-14 border-b border-border bg-surface flex-shrink-0">
      <div className="h-full flex items-center justify-between px-4">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-accent-bg border border-accent-bdr flex items-center justify-center text-accent">
            <BrainCircuit size={16} />
          </div>
          <span className="text-sm font-semibold tracking-tight text-text">AC-RAG</span>
          <Badge variant="primary">RESEARCH CONSOLE</Badge>
        </div>

        <div className="flex items-center gap-3">
          <Toggle isDark={isDark} onToggle={toggle} />
          <span className="text-[11px] text-muted font-mono">SRKR · 2025–26</span>
        </div>
      </div>
    </header>
  )
}
