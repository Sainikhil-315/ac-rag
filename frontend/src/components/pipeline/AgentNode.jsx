export default function AgentNode({ agent, state = 'idle', selected, onClick, meta }) {
  const isIdle   = state === 'idle'
  const isActive = state === 'active'
  const isDone   = state === 'done'
  const isError  = state === 'error'

  const stateConfig = {
    idle:   { border: 'border-border', accent: 'accent-transparent', bg: 'bg-surface' },
    active: { border: 'border-accent', accent: 'border-l-accent', bg: 'bg-accent-bg' },
    done:   { border: 'border-success', accent: 'border-l-success', bg: 'bg-success-bg' },
    error:  { border: 'border-danger', accent: 'border-l-danger', bg: 'bg-danger-bg' },
  }[state]

  const nodeBorder = selected ? 'border-2' : stateConfig.border
  const nodeColor  = selected ? agent.color : (state === 'active' ? 'var(--c-accent)' : state === 'done' ? 'var(--c-success)' : state === 'error' ? 'var(--c-danger)' : 'var(--c-border)')

  return (
    <div
      onClick={onClick}
      className={`
        relative rounded-lg px-3.5 py-3 cursor-pointer transition-all duration-200
        flex items-center gap-3 overflow-hidden
        ${stateConfig.bg} ${nodeBorder} ${stateConfig.accent}
        ${selected ? 'ring-2' : 'hover:bg-surface'}
      `}
      style={{
        borderColor: nodeColor,
        ...(selected ? { boxShadow: `0 0 0 3px ${agent.color}28` } : {}),
      }}
    >
      {/* Icon */}
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center text-base flex-shrink-0 border"
        style={{
          backgroundColor: agent.lightBg,
          borderColor: agent.color + '33',
          color: agent.color,
        }}
      >
        {isDone ? '✓' : isError ? '✗' : agent.icon}
      </div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-[9px] font-mono font-semibold" style={{
            color: isActive ? agent.color : isDone ? 'var(--c-success)' : 'var(--c-text-muted)'
          }}>
            {agent.number}
          </span>
          <span className={`text-sm font-semibold truncate transition-colors
            ${isIdle ? 'text-text-2' : isError ? 'text-danger' : 'text-primary'}`}>
            {agent.title}
          </span>
        </div>
        <div className="text-[10px] text-muted mt-0.5 truncate max-w-[180px]">
          {isDone && meta ? meta : agent.subtitle}
        </div>
      </div>

      {/* State indicator */}
      <div className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
        {isActive && (
          <div className="w-2 h-2 rounded-full bg-accent shadow-[0_0_0_3px_rgba(99,102,241,0.2)] animate-[pulse-dot_1.5s_ease-in-out_infinite]" />
        )}
        {isDone && (
          <div className="w-4 h-4 rounded-full bg-success-bg border border-success flex items-center justify-center text-[9px] font-bold text-success">✓</div>
        )}
        {isError && (
          <div className="w-4 h-4 rounded-full bg-danger-bg border border-danger flex items-center justify-center text-[9px] font-bold text-danger">✗</div>
        )}
      </div>
    </div>
  )
}
