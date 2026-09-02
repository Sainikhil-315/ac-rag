const STAGE_COLORS = {
  entry_router:      'bg-purple-500/20 text-purple-400 border-purple-500/30',
  query_analyzer:    'bg-blue-500/20 text-blue-400 border-blue-500/30',
  retrieval_planner: 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30',
  retriever:         'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  validator:         'bg-orange-500/20 text-orange-400 border-orange-500/30',
  context_refiner:   'bg-teal-500/20 text-teal-400 border-teal-500/30',
  generator:         'bg-green-500/20 text-green-400 border-green-500/30',
  critic:            'bg-red-500/20 text-red-400 border-red-500/30',
  direct_responder:  'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
}

const STATUS_ICONS = { completed: '✓', failed: '✗', skipped: '○', started: '·' }
const STATUS_COLORS = { completed: 'text-success', failed: 'text-danger', skipped: 'text-muted', started: 'text-secondary' }

export default function PipelineTrace({ stages }) {
  return (
    <div className="bg-surface border border-border rounded-xl p-4 animate-slide-up">
      <div className="space-y-2">
        {stages.map((stage, i) => {
          const stageColor = STAGE_COLORS[stage.stage] || 'bg-surface-2 text-secondary border-border'
          return (
            <div key={i} className="flex items-center gap-3">
              <span className={`${STATUS_COLORS[stage.status] || 'text-secondary'} w-4 text-xs font-mono`}>
                {STATUS_ICONS[stage.status] || '·'}
              </span>
              <span className={`text-[11px] font-mono px-2 py-0.5 rounded border ${stageColor}`}>
                {stage.stage}
              </span>
              <span className={`text-[11px] ml-auto ${STATUS_COLORS[stage.status] || 'text-secondary'}`}>
                {stage.status}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
