const DIMS = [
  { key: 'faithfulness',    label: 'Faith' },
  { key: 'completeness',    label: 'Complete' },
  { key: 'table_accuracy',  label: 'Tables' },
  { key: 'figure_accuracy', label: 'Figures' },
  { key: 'conciseness',     label: 'Concise' },
]

function scoreColor(v) {
  if (v >= 4) return 'text-success'
  if (v === 3) return 'text-warning'
  return 'text-danger'
}

function scoreBg(v) {
  if (v >= 4) return 'bg-success-bg'
  if (v === 3) return 'bg-warning-bg'
  return 'bg-danger-bg'
}

export default function ScoreCards({ scores }) {
  const overall = scores?.overall ?? 0
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-muted uppercase tracking-wider font-medium">
          Self-Reflection
        </span>
        <span className={`text-[11px] font-semibold ${scoreColor(overall)}`}>
          {overall.toFixed(1)}/5 overall
        </span>
      </div>
      <div className="grid grid-cols-5 gap-1.5">
        {DIMS.map(({ key, label }) => {
          const v = scores?.[key] ?? 0
          return (
            <div
              key={key}
              className={`rounded-lg p-3 text-center border ${scoreBg(v)} ${scoreColor(v)}`}
            >
              <div className="text-base font-bold">{v}</div>
              <div className="text-[10px] text-muted mt-0.5">{label}</div>
            </div>
          )
        })}
      </div>
      {scores?.feedback && (
        <p className="text-[11px] text-secondary italic leading-relaxed">
          "{scores.feedback}"
        </p>
      )}
    </div>
  )
}
