import { useState, useRef, useEffect } from 'react'
import { AGENTS, STATS } from '../constants/agents'

function useVisible(ref) {
  const [vis, setVis] = useState(false)
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setVis(true) }, { threshold: 0.15 })
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [ref])
  return vis
}

function Section({ children, delay = 0 }) {
  const ref = useRef()
  const vis = useVisible(ref)
  return (
    <div
      ref={ref}
      style={{
        opacity: vis ? 1 : 0,
        transform: vis ? 'translateY(0)' : 'translateY(28px)',
        transition: `opacity .6s ease ${delay}ms, transform .6s ease ${delay}ms`,
      }}
    >
      {children}
    </div>
  )
}

function AgentFlowNode({ agent, index, hovered, setHovered }) {
  return (
    <div
      className="relative flex flex-col items-center"
      onMouseEnter={() => setHovered(index)}
      onMouseLeave={() => setHovered(null)}
    >
      {/* Node */}
      <div
        className={`
          w-24 h-24 rounded-xl flex flex-col items-center justify-center text-2xl
          transition-all duration-300 cursor-default
          ${hovered === index
            ? 'shadow-lg ring-2 ring-offset-2'
            : 'border border-border'}
        `}
        style={{
          backgroundColor: agent.lightBg,
          borderColor: agent.color + '33',
          ringColor: agent.color,
          ringOffsetColor: hovered === index ? 'var(--c-bg)' : undefined,
        }}
      >
        {agent.icon}
      </div>

      {/* Number */}
      <span className="mt-2 text-xs font-mono font-semibold" style={{ color: agent.color }}>
        {agent.number}
      </span>

      {/* Title */}
      <span className="mt-1 text-xs text-center text-secondary font-medium max-w-[100px]">
        {agent.title}
      </span>

      {/* Hover tooltip */}
      {hovered === index && (
        <div className="absolute top-full mt-3 left-1/2 -translate-x-1/2 w-56 z-20">
          <div className="bg-surface border border-border rounded-lg p-3 shadow-lg animate-slide-up">
            <p className="text-[11px] text-secondary leading-relaxed">
              {agent.description}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default function Landing({ onTryMe }) {
  const [hovered, setHovered] = useState(null)

  return (
    <div className="min-h-screen bg-bg text-text flex flex-col font-sans">
      {/* ── NAV ─────────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-40 bg-surface/80 backdrop-blur border-b border-border">
        <div className="max-w-5xl mx-auto h-14 flex items-center justify-between px-6">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-accent-bg border border-accent-bdr flex items-center justify-center text-sm">
              🧠
            </div>
            <span className="font-bold text-base text-primary">AC-RAG</span>
            <span className="text-[10px] font-medium text-accent bg-accent-bg border border-accent-bdr rounded-full px-2 py-0.5">
              RESEARCH
            </span>
          </div>
          <div className="flex items-center gap-6">
            <a href="#pipeline" className="text-sm text-secondary hover:text-primary transition-colors">
              Pipeline
            </a>
            <a href="#results" className="text-sm text-secondary hover:text-primary transition-colors">
              Results
            </a>
            <a href="#agents" className="text-sm text-secondary hover:text-primary transition-colors">
              Agents
            </a>
            <button
              onClick={onTryMe}
              className="px-4 py-1.5 text-sm font-semibold text-white bg-accent rounded-lg hover:bg-accent-hover transition-colors shadow-sm"
            >
              Try it →
            </button>
          </div>
        </div>
      </nav>

      {/* ── HERO ─────────────────────────────────────────────------ */}
      <section className="max-w-3xl mx-auto pt-16 pb-12 px-6 text-center">
        <p className="text-xs font-semibold text-accent uppercase tracking-wider mb-4">
          B.Tech Final Year Project · SRKR Engineering College · Dept. of IT
        </p>

        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-primary mb-6">
          Document QA that{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent to-secondary">
            verifies its own answers
          </span>
        </h1>

        <p className="text-lg text-secondary mb-8 max-w-xl mx-auto leading-relaxed">
          AC-RAG routes your query through 7 specialised agents that retrieve
          evidence, filter noise, generate grounded answers, and self-reflect —
          all before presenting a confidence-scored result.
        </p>

        <div className="flex gap-3 justify-center">
          <button
            onClick={onTryMe}
            className="px-6 py-2.5 text-base font-semibold text-white bg-accent rounded-lg hover:bg-accent-hover transition-colors shadow-sm"
          >
            Try me — upload a PDF
          </button>
          <a
            href="#pipeline"
            className="px-6 py-2.5 text-base font-medium text-primary bg-surface border border-border rounded-lg hover:bg-surface-2 transition-colors"
          >
            See the pipeline ↓
          </a>
        </div>

        {/* Trust badge */}
        <div className="mt-8 flex items-center justify-center gap-4 text-xs text-muted">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-success"></span>
            <span>Faithfulness 1.0 (RAGAS)</span>
          </div>
          <span>·</span>
          <span>2× answer accuracy vs baseline</span>
        </div>
      </section>

      {/* ── STATS ────────────────────────────────────────────── */}
      <section id="results" className="max-w-5xl mx-auto px-6 pb-12">
        <Section delay={0}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {STATS.map((s, i) => (
              <div
                key={i}
                className="bg-surface border border-border rounded-xl p-6 text-center transition-all hover:shadow-md"
              >
                <div
                  className="text-3xl font-extrabold mb-1 bg-gradient-to-r from-accent to-secondary bg-clip-text text-transparent"
                >
                  {s.val}
                </div>
                <div className="text-sm font-semibold text-primary mb-0.5">
                  {s.label}
                </div>
                <div className="text-xs text-muted">
                  {s.sub}
                </div>
              </div>
            ))}
          </div>
        </Section>
      </section>

      {/* ── PIPELINE FLOW ────────────────────────────────────── */}
      <section id="pipeline" className="max-w-5xl mx-auto px-6 pb-16">
        <Section delay={100}>
          <div className="text-center mb-10">
            <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">
              How it works
            </p>
            <h2 className="text-2xl font-bold text-primary">
              One query. Seven agents. Zero hallucination.
            </h2>
          </div>

          <div className="flex items-center justify-center gap-1 overflow-x-auto py-4">
            {/* Entry */}
            <div className="flex flex-col items-center flex-shrink-0">
              <div className="w-10 h-10 rounded-lg bg-surface border border-border flex items-center justify-center text-lg">
                👤
              </div>
              <span className="mt-1.5 text-[10px] font-mono text-muted">User</span>
            </div>

            {AGENTS.map((agent, i) => (
              <div key={agent.id} className="flex items-center flex-shrink-0">
                {/* Connector arrow */}
                <div className="flex items-center">
                  <div className="w-5 h-px bg-border" />
                  <svg width="6" height="5" viewBox="0 0 8 5" fill="none" className="ml-0.5">
                    <path d="M4 5L0 0H8L4 5Z" fill="var(--c-border)" />
                  </svg>
                </div>

                {/* Agent node */}
                <AgentFlowNode
                  agent={agent}
                  index={i}
                  hovered={hovered}
                  setHovered={setHovered}
                />
              </div>
            ))}

            {/* Exit */}
            <div className="flex items-center flex-shrink-0">
              <div className="flex items-center">
                <div className="w-5 h-px bg-border" />
                <svg width="6" height="5" viewBox="0 0 8 5" fill="none" className="ml-0.5">
                  <path d="M4 5L0 0H8L4 5Z" fill="var(--c-border)" />
                </svg>
              </div>
              <div className="w-10 h-10 rounded-lg bg-success-bg border border-success flex items-center justify-center text-lg">
                ✅
              </div>
              <span className="mt-1.5 text-[10px] font-mono text-success">Answer</span>
            </div>
          </div>
        </Section>
      </section>

      {/* ── AGENTS GRID ─────────────────────────────────────── */}
      <section id="agents" className="max-w-5xl mx-auto px-6 pb-16">
        <Section>
          <div className="mb-8">
            <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-3">
              The Pipeline
            </p>
            <h2 className="text-2xl font-bold text-primary">
              7 agents, one reliable answer
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {AGENTS.map((agent, i) => (
              <Section key={agent.id} delay={i * 60}>
                <div className="bg-surface border border-border rounded-xl p-5 h-full transition-all hover:shadow-md">
                  <div className="flex items-center justify-between mb-3">
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center text-lg border"
                      style={{ backgroundColor: agent.lightBg, borderColor: agent.color + '40' }}
                    >
                      {agent.icon}
                    </div>
                    <span className="text-xs font-mono" style={{ color: agent.color }}>
                      {agent.number}
                    </span>
                  </div>
                  <h3 className="text-sm font-bold text-primary mb-1.5">{agent.title}</h3>
                  <p className="text-xs text-muted leading-relaxed">
                    {agent.description}
                  </p>
                </div>
              </Section>
            ))}

            {/* CTA card */}
            <Section delay={7 * 60}>
              <div
                onClick={onTryMe}
                className="bg-gradient-to-br from-accent-bg to-secondary/10 border border-accent-bdr rounded-xl p-5 h-full flex flex-col items-center justify-center gap-3 cursor-pointer transition-all hover:shadow-lg"
              >
                <div className="text-2xl">🚀</div>
                <div className="text-sm font-bold text-primary">See it in action</div>
                <div className="text-xs text-muted text-center">
                  Upload your PDF and ask questions
                </div>
                <div className="px-3 py-1 bg-accent/10 border border-accent-bdr rounded-md text-xs font-semibold" style={{ color: 'var(--c-accent-txt)' }}>
                  Try me →
                </div>
              </div>
            </Section>
          </div>
        </Section>
      </section>

      {/* ── FOOTER ───────────────────────────────────────────── */}
      <footer className="border-t border-border bg-surface/50 mt-auto">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between text-xs text-muted">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-accent to-secondary flex items-center justify-center text-xs">
              🧠
            </div>
            <span>AC-RAG · Agent-Controlled Retrieval-Augmented Generation</span>
          </div>
          <span>SRKR Engineering College · Dept. of IT · 2025–26</span>
        </div>
      </footer>
    </div>
  )
}
