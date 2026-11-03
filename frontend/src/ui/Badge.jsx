export function Badge({ children, variant = "default", size = "sm", className = "" }) {
  const variants = {
    default:  "bg-surface-2 text-secondary border border-border",
    primary:  "bg-accent-bg text-accent-txt border border-accent-bdr",
    success:  "bg-success-bg text-success border border-success-bdr",
    warning:  "bg-warning-bg text-warning border border-warning-bdr",
    danger:   "bg-danger-bg text-danger border border-danger-bdr",
    dark:     "bg-surface-2 text-text-faint border border-border",
  }
  const sizes = {
    sm: "px-2 py-0.5 text-[10px] font-semibold",
    md: "px-2.5 py-1 text-xs font-semibold",
  }
  return (
    <span className={`inline-flex items-center rounded-full ${variants[variant] || variants.default} ${sizes[size] || sizes.sm} ${className}`}>
      {children}
    </span>
  )
}

export function StatusDot({ status = "idle", size = "sm" }) {
  const color = {
    idle:   "bg-text-faint",
    active: "bg-accent shadow-[0_0_0_3px_var(--c-accent-bg)]",
    done:   "bg-success",
    error:  "bg-danger",
  }[status] || "bg-text-faint"
  const dims = { sm: "w-2 h-2", md: "w-3 h-3" }
  return (
    <span className={`inline-block rounded-full ${dims[size] || dims.sm} ${color} ${status === 'active' ? 'animate-pulse' : ''}`} />
  )
}
