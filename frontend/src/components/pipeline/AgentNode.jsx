// AgentNode — one node in the pipeline canvas
// state: 'idle' | 'active' | 'done' | 'error'
// selected: whether the inspector is open for this node

export default function AgentNode({ agent, state = 'idle', selected, onClick, meta }) {
  const isIdle   = state === 'idle'
  const isActive = state === 'active'
  const isDone   = state === 'done'
  const isError  = state === 'error'

  // Border + left accent colour per state
  const stateStyle = {
    idle:   { border: '1px solid var(--c-border)',        accent: 'transparent',       bg: 'var(--c-surface)'   },
    active: { border: '1px solid var(--c-accent)',        accent: 'var(--c-accent)',   bg: 'var(--c-accent-bg)' },
    done:   { border: '1px solid var(--c-success)',       accent: 'var(--c-success)',  bg: 'var(--c-success-bg)'},
    error:  { border: '1px solid var(--c-danger)',        accent: 'var(--c-danger)',   bg: 'var(--c-danger-bg)' },
  }[state]

  return (
    <div
      onClick={onClick}
      style={{
        position: 'relative',
        background: stateStyle.bg,
        border: selected ? `1.5px solid ${agent.color}` : stateStyle.border,
        borderRadius: '10px',
        padding: '12px 14px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        boxShadow: selected
          ? `0 0 0 3px ${agent.color}28`
          : isActive
          ? '0 2px 8px rgba(99,102,241,0.12)'
          : '0 1px 2px rgba(0,0,0,0.04)',
        transition: 'box-shadow 0.15s, border-color 0.15s',
        overflow: 'hidden',
      }}
    >
      {/* Left accent bar */}
      <div style={{
        position: 'absolute',
        left: 0, top: 0, bottom: 0,
        width: '3px',
        background: stateStyle.accent,
        borderRadius: '10px 0 0 10px',
        transition: 'background 0.2s',
      }} />

      {/* Icon */}
      <div style={{
        width: '34px', height: '34px', borderRadius: '8px',
        background: isIdle ? 'var(--c-bg)' : agent.lightBg,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '16px', flexShrink: 0,
        transition: 'background 0.2s',
      }}>
        {isDone ? '✓' : isError ? '✗' : agent.icon}
      </div>

      {/* Text */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            fontSize: '10px', fontFamily: 'monospace', fontWeight: 600,
            color: isActive ? agent.color : isDone ? 'var(--c-success)' : 'var(--c-text-muted)',
          }}>
            {agent.number}
          </span>
          <span style={{
            fontSize: '13px', fontWeight: 600,
            color: isIdle ? 'var(--c-text-2)' : isError ? 'var(--c-danger)' : 'var(--c-text)',
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>
            {agent.title}
          </span>
        </div>
        <div style={{
          fontSize: '11px', color: 'var(--c-text-muted)', marginTop: '1px',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}>
          {isDone && meta ? meta : agent.subtitle}
        </div>
      </div>

      {/* State indicator */}
      <div style={{ flexShrink: 0 }}>
        {isActive && (
          <div style={{
            width: '8px', height: '8px', borderRadius: '50%',
            background: 'var(--c-accent)',
            boxShadow: '0 0 0 3px rgba(99,102,241,0.2)',
            animation: 'pulse-dot 1.5s ease-in-out infinite',
          }} />
        )}
        {isDone && (
          <div style={{
            width: '18px', height: '18px', borderRadius: '50%',
            background: 'var(--c-success-bg)', border: '1px solid var(--c-success)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '10px', color: 'var(--c-success)', fontWeight: 700,
          }}>✓</div>
        )}
        {isError && (
          <div style={{
            width: '18px', height: '18px', borderRadius: '50%',
            background: 'var(--c-danger-bg)', border: '1px solid var(--c-danger)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '10px', color: 'var(--c-danger)', fontWeight: 700,
          }}>✗</div>
        )}
      </div>
    </div>
  )
}
