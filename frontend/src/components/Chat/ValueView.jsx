// Generic recursive renderer for arbitrary pipeline output (dicts/lists/scalars).
// One component handles every field shape in the research-tester detail panel,
// instead of a bespoke renderer per field — keeps this lean while still showing
// every value, since the pipeline's internal state shapes vary a lot by node.

function Scalar({ value }) {
  if (value === null || value === undefined || value === '') {
    return <span className="text-muted italic">—</span>
  }
  if (typeof value === 'boolean') {
    return <span className={value ? 'text-success' : 'text-danger'}>{String(value)}</span>
  }
  if (typeof value === 'number') {
    return <span className="text-accent font-mono">{Number.isInteger(value) ? value : value.toFixed(4)}</span>
  }
  return <span className="text-text break-words">{String(value)}</span>
}

export default function ValueView({ value, depth = 0 }) {
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-muted italic">empty</span>
    return (
      <div className="space-y-1.5">
        {value.map((item, i) => (
          <div key={i} className="rounded-md border border-border bg-surface-2 px-2.5 py-1.5">
            <ValueView value={item} depth={depth + 1} />
          </div>
        ))}
      </div>
    )
  }

  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value)
    if (entries.length === 0) return <span className="text-muted italic">empty</span>
    return (
      <div className={depth > 0 ? 'space-y-1' : 'grid grid-cols-[minmax(120px,auto)_1fr] gap-x-3 gap-y-1.5'}>
        {entries.map(([k, v]) => (
          <div key={k} className={depth > 0 ? 'flex gap-2 text-xs' : 'contents'}>
            <span className="text-[11px] font-mono text-secondary whitespace-nowrap">{k}</span>
            <div className="text-xs">
              <ValueView value={v} depth={depth + 1} />
            </div>
          </div>
        ))}
      </div>
    )
  }

  return <Scalar value={value} />
}
