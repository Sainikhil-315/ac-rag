import { forwardRef } from 'react'

const base = "inline-flex items-center justify-center rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"

const variants = {
  primary: "bg-accent text-white hover:bg-accent-hover shadow-sm",
  secondary: "bg-surface border border-border text-primary hover:bg-surface-2",
  ghost: "text-secondary hover:bg-surface-2 border border-transparent hover:border-border",
  danger: "bg-danger text-white hover:bg-danger/90",
  accent: "bg-accent text-white hover:bg-accent-hover",
}

const sizes = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-4 py-2 text-sm",
  lg: "px-5 py-2.5 text-sm",
}

export const Button = forwardRef(function Button(
  { className = "", variant = "primary", size = "md", children, leftIcon, rightIcon, ...props },
  ref
) {
  const cls = `${base} ${variants[variant] || variants.primary} ${sizes[size] || sizes.md} ${className}`
  return (
    <button ref={ref} className={cls} {...props}>
      {leftIcon && <span className="mr-1.5 flex items-center">{leftIcon}</span>}
      {children}
      {rightIcon && <span className="ml-1.5 flex items-center">{rightIcon}</span>}
    </button>
  )
})
