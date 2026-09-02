import { useTheme } from '../ThemeContext'

export default function Header({ activeTab, onTabChange }) {
  const { isDark, toggle } = useTheme()
  const tabs = ['Pipeline', 'Playground', 'Results']

  return (
    <div style={{
      height: '48px',
      background: 'var(--c-surface)',
      borderBottom: '1px solid var(--c-border)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 20px', flexShrink: 0,
      transition: 'background 0.2s, border-color 0.2s',
    }}>

      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{
          width: '26px', height: '26px', borderRadius: '7px',
          background: 'var(--c-accent-bg)', border: '1px solid var(--c-accent-bdr)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '13px',
        }}>🧠</div>
        <span style={{ fontSize: '14px', fontWeight: 600, letterSpacing: '-0.01em', color: 'var(--c-text)' }}>
          AC-RAG
        </span>
        <span style={{
          fontSize: '10px', fontWeight: 500, color: 'var(--c-accent)',
          background: 'var(--c-accent-bg)', border: '1px solid var(--c-accent-bdr)',
          borderRadius: '999px', padding: '1px 8px', letterSpacing: '0.02em',
        }}>RESEARCH</span>
      </div>

      {/* Tabs */}
      <div style={{
        display: 'flex', gap: '2px',
        background: 'var(--c-bg)', borderRadius: '8px', padding: '3px',
      }}>
        {tabs.map(tab => {
          const active = activeTab === tab
          return (
            <button key={tab} onClick={() => onTabChange(tab)} style={{
              padding: '4px 14px', fontSize: '13px',
              fontWeight: active ? 600 : 400,
              color: active ? 'var(--c-text)' : 'var(--c-text-sec)',
              background: active ? 'var(--c-surface)' : 'transparent',
              border: active ? '1px solid var(--c-border)' : '1px solid transparent',
              borderRadius: '6px', cursor: 'pointer',
              transition: 'all 0.15s', fontFamily: 'Inter, sans-serif',
            }}>
              {tab}
            </button>
          )
        })}
      </div>

      {/* Right: theme toggle + label */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* Dark mode toggle */}
        <button
          onClick={toggle}
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            background: 'none', border: 'none', cursor: 'pointer', padding: '4px',
          }}
        >
          <div style={{
            width: '34px', height: '18px', borderRadius: '9px',
            background: isDark ? 'var(--c-accent)' : 'var(--c-border)',
            position: 'relative', transition: 'background 0.25s', flexShrink: 0,
          }}>
            <div style={{
              position: 'absolute', top: '2px',
              left: isDark ? '18px' : '2px',
              width: '14px', height: '14px', borderRadius: '50%',
              background: '#FFFFFF',
              boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
              transition: 'left 0.25s',
            }} />
          </div>
          <span style={{ fontSize: '13px', lineHeight: 1 }}>
            {isDark ? '🌙' : '☀️'}
          </span>
        </button>

        <span style={{
          fontSize: '11px', color: 'var(--c-text-muted)', fontFamily: 'monospace',
        }}>
          SRKR · 2025–26
        </span>
      </div>
    </div>
  )
}
