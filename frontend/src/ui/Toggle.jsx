export function Toggle({ isDark, onToggle, className = "" }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className={`
        relative inline-flex h-5 w-9 items-center rounded-full
        transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg
        ${isDark ? 'bg-accent' : 'bg-border'}
        ${className}
      `}
    >
      <span className="sr-only">Toggle dark mode</span>
      <span
        className={`
          absolute top-0.5 h-4 w-4 rounded-full bg-white shadow
          transition-transform duration-200
          ${isDark ? 'translate-x-5' : 'translate-x-0.5'}
        `}
      />
    </button>
  )
}

export function SegmentedControl({ options, value, onChange, className = "" }) {
  return (
    <div className={`inline-flex items-center rounded-lg bg-surface-2 p-1 border border-border ${className}`}>
      {options.map(opt => {
        const active = value === opt.value
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`
              px-3 py-1.5 text-xs font-medium rounded-md transition-all
              ${active
                ? 'bg-surface text-primary shadow-sm'
                : 'text-secondary hover:text-primary'}
            `}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}
