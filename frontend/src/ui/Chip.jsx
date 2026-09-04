export function Chip({ label, value, valueColor = "text-accent", variant = "subtle", className = "" }) {
  const variants = {
    subtle: "bg-surface-2 text-secondary border border-border",
    accent: "bg-accent-bg text-accent border border-accent-bdr",
  }
  const v = variants[variant] || variants.subtle
  return (
    <div className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-mono ${v} ${className}`}>
      <span className="text-[9px] font-semibold opacity-50">{label}</span>
      <span className={`font-semibold ${valueColor}`}>{value}</span>
    </div>
  )
}

export function Progress({ value = 0, max = 100, color = "accent", size = "sm", animated = true, className = "" }) {
  const pct = Math.min((value / max) * 100, 100)
  const colors = {
    accent:  "bg-accent",
    success: "bg-success",
    danger:  "bg-danger",
    warning: "bg-warning",
  }
  const bgColors = {
    accent:  "bg-accent-bg",
    success: "bg-success-bg",
    danger:  "bg-danger-bg",
    warning: "bg-warning-bg",
  }
  const heights = { sm: "h-1", md: "h-1.5" }
  return (
    <div className={`w-full overflow-hidden rounded-full ${bgColors[color] || bgColors.accent} ${heights[size] || heights.sm} ${className}`}>
      <div
        className={`h-full rounded-full transition-all duration-500 ${colors[color] || colors.accent}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
