import { forwardRef } from 'react'

export const Card = forwardRef(function Card(
  { className = "", title, subtitle, children, ...props },
  ref
) {
  const cls = "bg-surface border border-border rounded-xl p-5 shadow-sm"
  if (!title && !subtitle) {
    return (
      <div ref={ref} className={`${cls} ${className}`} {...props}>
        {children}
      </div>
    )
  }
  return (
    <div ref={ref} className={`bg-surface border border-border rounded-xl shadow-sm overflow-hidden ${className}`} {...props}>
      <div className="px-5 py-4 border-b border-border">
        {title && <h3 className="text-xs font-semibold text-muted uppercase tracking-wider">{title}</h3>}
        {subtitle && <p className="text-xs text-secondary mt-0.5">{subtitle}</p>}
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
})

export const CardPlain = forwardRef(function CardPlain({ className = "", children, ...props }, ref) {
  return (
    <div ref={ref} className={`bg-surface border border-border rounded-xl p-5 shadow-sm ${className}`} {...props}>
      {children}
    </div>
  )
})
