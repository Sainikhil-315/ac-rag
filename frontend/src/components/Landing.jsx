import { useEffect, useRef, useState } from 'react'

/* ─── keyframes injected once ─────────────────────────────────────── */
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

@keyframes float   { 0%,100%{transform:translateY(0)}  50%{transform:translateY(-18px)} }
@keyframes glow    { 0%,100%{opacity:.35} 50%{opacity:.7} }
@keyframes fadeUp  { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
@keyframes spin    { to{transform:rotate(360deg)} }
@keyframes shimmer { 0%{background-position:-400px 0} 100%{background-position:400px 0} }
@keyframes pulseRing {
  0%   { transform:scale(.95); box-shadow:0 0 0 0 rgba(99,102,241,.5); }
  70%  { transform:scale(1);   box-shadow:0 0 0 12px rgba(99,102,241,0); }
  100% { transform:scale(.95); box-shadow:0 0 0 0 rgba(99,102,241,0); }
}
@keyframes gradShift {
  0%  {background-position:0% 50%}
  50% {background-position:100% 50%}
  100%{background-position:0% 50%}
}
@keyframes counterUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }

.landing-root * { box-sizing:border-box; margin:0; padding:0; }
.landing-root a { text-decoration:none; color:inherit; }

/* gradient headline */
.grad-text {
  background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 30%, #c4b5fd 60%, #f0abfc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* glass card */
.glass {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.09);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
.glass:hover {
  background: rgba(255,255,255,.07);
  border-color: rgba(165,180,252,.25);
  transform: translateY(-4px);
  box-shadow: 0 20px 40px rgba(0,0,0,.35), 0 0 0 1px rgba(165,180,252,.15);
}

/* agent card hover glow */
.agent-glass:hover { box-shadow: 0 0 0 1px rgba(165,180,252,.3), 0 24px 48px rgba(0,0,0,.4); }

/* nav link */
.nav-link { color:rgba(255,255,255,.55); font-size:14px; transition:color .2s; }
.nav-link:hover { color:#fff; }

/* primary button */
.btn-hero {
  background: linear-gradient(135deg,#6366f1,#8b5cf6);
  color:#fff; border:none; border-radius:12px;
  padding:14px 32px; font-size:15px; font-weight:700;
  cursor:pointer; letter-spacing:-.01em;
  box-shadow: 0 8px 24px rgba(99,102,241,.4);
  transition: transform .15s, box-shadow .15s;
}
.btn-hero:hover {
  transform:translateY(-2px);
  box-shadow: 0 12px 32px rgba(99,102,241,.55);
}
.btn-ghost {
  background:rgba(255,255,255,.07); color:rgba(255,255,255,.8);
  border:1px solid rgba(255,255,255,.12); border-radius:12px;
  padding:14px 28px; font-size:15px; font-weight:500;
  cursor:pointer; transition:all .15s;
}
.btn-ghost:hover { background:rgba(255,255,255,.12); color:#fff; border-color:rgba(255,255,255,.22); }

/* stat card */
.stat-card {
  background:rgba(255,255,255,.045);
  border:1px solid rgba(255,255,255,.08);
  border-radius:20px; padding:28px 24px; text-align:center;
  transition: all .25s;
}
.stat-card:hover {
  background:rgba(99,102,241,.12);
  border-color:rgba(165,180,252,.25);
  transform:translateY(-3px);
}

/* pipeline connector */
.pipe-line { width:1px; height:20px; background:rgba(99,102,241,.3); margin:0 auto; }
.pipe-arrow {
  width:0; height:0; margin:0 auto;
  border-left:4px solid transparent;
  border-right:4px solid transparent;
  border-top:5px solid rgba(99,102,241,.4);
}

/* scrollbar */
.landing-root::-webkit-scrollbar { width:5px; }
.landing-root::-webkit-scrollbar-track { background:transparent; }
.landing-root::-webkit-scrollbar-thumb { background:#3f3f46; border-radius:99px; }
`

const AGENTS = [
  { n:'01', icon:'🔍', color:'#818cf8', bg:'rgba(99,102,241,.15)', title:'Query Understanding',
    desc:'Rewrites vague queries, detects intent (factual / analytical / comparative / summarization), scores complexity 0–1.' },
  { n:'02', icon:'📋', color:'#a78bfa', bg:'rgba(139,92,246,.15)', title:'Retrieval Planning',
    desc:'Dynamically sets retrieval depth, passage count (4–12), modality filter and MMR parameters based on complexity.' },
  { n:'03', icon:'📚', color:'#38bdf8', bg:'rgba(56,189,248,.12)', title:'Document Retrieval',
    desc:'Executes Maximum Marginal Relevance search. For complex queries, runs sub-queries independently and pools results.' },
  { n:'04', icon:'✅', color:'#fbbf24', bg:'rgba(251,191,36,.12)', title:'Evidence Validation',
    desc:'Scores each passage for relevance. Passages below threshold are discarded. Triggers auto-retry if too few survive.' },
  { n:'05', icon:'✂️', color:'#34d399', bg:'rgba(52,211,153,.12)', title:'Context Refinement',
    desc:'Deduplicates near-identical passages, compresses verbose text, re-ranks by score before passing to the generator.' },
  { n:'06', icon:'✍️', color:'#60a5fa', bg:'rgba(96,165,250,.12)', title:'Answer Generation',
    desc:'LLM generates a grounded answer strictly from refined context. Every claim is cited back to a source passage.' },
  { n:'07', icon:'🪞', color:'#f87171', bg:'rgba(248,113,113,.12)', title:'Self-Reflection',
    desc:'Scores the answer on faithfulness, completeness, table accuracy, figure accuracy and conciseness (each 1–5).' },
]

const STATS = [
  { val:'1.0',  label:'Faithfulness',    sub:'RAGAS score' },
  { val:'0.88', label:'Answer Relevancy', sub:'RAGAS score' },
  { val:'2×',   label:'Answer Accuracy',  sub:'vs baseline' },
  { val:'7',    label:'Agents',           sub:'in the pipeline' },
]

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
    <div ref={ref} style={{
      opacity: vis ? 1 : 0,
      transform: vis ? 'translateY(0)' : 'translateY(28px)',
      transition: `opacity .6s ease ${delay}ms, transform .6s ease ${delay}ms`,
    }}>
      {children}
    </div>
  )
}

export default function Landing({ onTryMe }) {
  const [hovered, setHovered] = useState(null)

  return (
    <div className="landing-root" style={{
      background: '#09090b',
      minHeight: '100vh',
      overflowY: 'auto',
      fontFamily: "'Inter', -apple-system, sans-serif",
      color: '#fff',
      position: 'relative',
    }}>
      <style>{STYLE}</style>

      {/* ── Background orbs ─────────────────────────────────────── */}
      <div style={{ position:'fixed', inset:0, overflow:'hidden', pointerEvents:'none', zIndex:0 }}>
        <div style={{
          position:'absolute', width:700, height:700, borderRadius:'50%',
          background:'radial-gradient(circle, rgba(99,102,241,.18) 0%, transparent 70%)',
          top:-200, left:-150, animation:'float 8s ease-in-out infinite',
        }}/>
        <div style={{
          position:'absolute', width:500, height:500, borderRadius:'50%',
          background:'radial-gradient(circle, rgba(139,92,246,.15) 0%, transparent 70%)',
          top:'30%', right:-100, animation:'float 10s ease-in-out infinite 2s',
        }}/>
        <div style={{
          position:'absolute', width:400, height:400, borderRadius:'50%',
          background:'radial-gradient(circle, rgba(56,189,248,.1) 0%, transparent 70%)',
          bottom:100, left:'30%', animation:'float 7s ease-in-out infinite 4s',
        }}/>
        {/* grid overlay */}
        <div style={{
          position:'absolute', inset:0,
          backgroundImage:'linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px)',
          backgroundSize:'48px 48px',
        }}/>
      </div>

      {/* ── NAV ─────────────────────────────────────────────────── */}
      <nav style={{
        position:'sticky', top:0, zIndex:50,
        background:'rgba(9,9,11,.8)', backdropFilter:'blur(20px)',
        borderBottom:'1px solid rgba(255,255,255,.07)',
        padding:'0 2rem',
      }}>
        <div style={{ maxWidth:1140, margin:'0 auto', height:60, display:'flex', alignItems:'center', justifyContent:'space-between' }}>
          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
            <div style={{
              width:34, height:34, borderRadius:9,
              background:'linear-gradient(135deg,#6366f1,#8b5cf6)',
              display:'flex', alignItems:'center', justifyContent:'center', fontSize:16,
              boxShadow:'0 4px 12px rgba(99,102,241,.4)',
            }}>🧠</div>
            <span style={{ fontWeight:800, fontSize:16, letterSpacing:'-.02em' }}>AC-RAG</span>
            <span style={{
              fontSize:10, fontWeight:700, color:'#818cf8',
              background:'rgba(99,102,241,.15)', border:'1px solid rgba(99,102,241,.3)',
              borderRadius:999, padding:'2px 10px', letterSpacing:'.06em',
            }}>RESEARCH</span>
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:32 }}>
            <a href="#pipeline" className="nav-link">Pipeline</a>
            <a href="#results"  className="nav-link">Results</a>
            <a href="#agents"   className="nav-link">Agents</a>
            <button className="btn-hero" style={{ padding:'8px 20px', fontSize:13 }} onClick={onTryMe}>
              Try it →
            </button>
          </div>
        </div>
      </nav>

      <div style={{ position:'relative', zIndex:1 }}>

        {/* ── HERO ────────────────────────────────────────────────── */}
        <section style={{ maxWidth:1140, margin:'0 auto', padding:'100px 2rem 80px', textAlign:'center' }}>

          {/* badge */}
          <div style={{
            display:'inline-flex', alignItems:'center', gap:8,
            background:'rgba(99,102,241,.12)', border:'1px solid rgba(99,102,241,.3)',
            borderRadius:999, padding:'6px 18px', marginBottom:36,
            fontSize:12, color:'#a5b4fc', fontWeight:600, letterSpacing:'.04em',
            animation:'pulseRing 2.5s infinite',
          }}>
            <span style={{ width:6, height:6, borderRadius:'50%', background:'#818cf8', display:'inline-block' }}/>
            B.Tech Final Year Project · SRKR Engineering College · Dept. of IT
          </div>

          {/* headline */}
          <h1 style={{
            fontSize:'clamp(38px, 6vw, 68px)', fontWeight:900,
            letterSpacing:'-.04em', lineHeight:1.08, marginBottom:24,
          }}>
            Document QA that<br />
            <span className="grad-text">verifies its own answers</span>
          </h1>

          <p style={{
            fontSize:18, color:'rgba(255,255,255,.55)', lineHeight:1.8,
            maxWidth:560, margin:'0 auto 48px',
          }}>
            AC-RAG runs 7 specialised agents to retrieve evidence, filter noise,
            generate grounded answers and self-reflect — all in one pipeline.
          </p>

          <div style={{ display:'flex', gap:14, justifyContent:'center', flexWrap:'wrap' }}>
            <button className="btn-hero" onClick={onTryMe}>
              Try me — upload a PDF
            </button>
            <a href="#pipeline" className="btn-ghost" style={{ display:'inline-block' }}>
              See the pipeline ↓
            </a>
          </div>

          {/* floating demo badge */}
          <div style={{
            marginTop:64, display:'inline-flex', alignItems:'center', gap:10,
            background:'rgba(255,255,255,.04)', border:'1px solid rgba(255,255,255,.08)',
            borderRadius:14, padding:'12px 20px',
          }}>
            <div style={{
              width:8, height:8, borderRadius:'50%', background:'#34d399',
              boxShadow:'0 0 8px #34d399', flexShrink:0,
            }}/>
            <span style={{ fontSize:13, color:'rgba(255,255,255,.6)' }}>
              Faithfulness score <strong style={{ color:'#34d399' }}>1.0</strong> on RAGAS benchmark
              &nbsp;·&nbsp; Answer accuracy <strong style={{ color:'#818cf8' }}>2× baseline</strong>
            </span>
          </div>
        </section>

        {/* ── STATS ───────────────────────────────────────────────── */}
        <section id="results" style={{ maxWidth:1140, margin:'0 auto', padding:'0 2rem 80px' }}>
          <Section>
            <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:16 }}>
              {STATS.map((s, i) => (
                <div key={i} className="stat-card">
                  <div style={{
                    fontSize:44, fontWeight:900, letterSpacing:'-.04em', marginBottom:8,
                    background:'linear-gradient(135deg,#a5b4fc,#c4b5fd)',
                    WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent',
                  }}>{s.val}</div>
                  <div style={{ fontSize:14, fontWeight:600, color:'rgba(255,255,255,.8)', marginBottom:4 }}>{s.label}</div>
                  <div style={{ fontSize:12, color:'rgba(255,255,255,.35)' }}>{s.sub}</div>
                </div>
              ))}
            </div>
          </Section>
        </section>

        {/* ── PIPELINE FLOW ───────────────────────────────────────── */}
        <section id="pipeline" style={{ maxWidth:1140, margin:'0 auto', padding:'0 2rem 80px' }}>
          <Section>
            <div style={{ textAlign:'center', marginBottom:52 }}>
              <p style={{ fontSize:11, fontWeight:700, color:'rgba(255,255,255,.3)', textTransform:'uppercase', letterSpacing:'.12em', marginBottom:12 }}>
                How it works
              </p>
              <h2 style={{ fontSize:'clamp(26px,4vw,40px)', fontWeight:800, letterSpacing:'-.03em', marginBottom:14 }}>
                One query. Seven agents. Zero hallucination.
              </h2>
              <p style={{ fontSize:15, color:'rgba(255,255,255,.45)', maxWidth:520, margin:'0 auto' }}>
                Every agent owns exactly one responsibility and passes its output to the next — with automatic retry when quality thresholds aren't met.
              </p>
            </div>

            {/* horizontal flow */}
            <div style={{ overflowX:'auto', paddingBottom:12 }}>
              <div style={{ display:'flex', alignItems:'center', gap:0, minWidth:900, justifyContent:'center' }}>
                {/* User query node */}
                <div style={{
                  background:'rgba(255,255,255,.06)', border:'1px solid rgba(255,255,255,.12)',
                  borderRadius:12, padding:'10px 16px', textAlign:'center', flexShrink:0,
                }}>
                  <div style={{ fontSize:18, marginBottom:4 }}>👤</div>
                  <div style={{ fontSize:11, color:'rgba(255,255,255,.5)', fontWeight:600 }}>User Query</div>
                </div>

                {AGENTS.map((a, i) => (
                  <div key={i} style={{ display:'flex', alignItems:'center' }}>
                    {/* connector */}
                    <div style={{ width:28, height:1, background:'rgba(99,102,241,.35)', flexShrink:0 }}/>
                    <div style={{ width:0, height:0, borderTop:'4px solid transparent', borderBottom:'4px solid transparent', borderLeft:'5px solid rgba(99,102,241,.5)', flexShrink:0 }}/>

                    {/* agent node */}
                    <div
                      onMouseEnter={() => setHovered(i)}
                      onMouseLeave={() => setHovered(null)}
                      style={{
                        width:90, flexShrink:0, background: hovered === i ? a.bg : 'rgba(255,255,255,.04)',
                        border: `1px solid ${hovered === i ? a.color + '55' : 'rgba(255,255,255,.08)'}`,
                        borderRadius:12, padding:'10px 8px', textAlign:'center',
                        cursor:'default', transition:'all .25s',
                        boxShadow: hovered === i ? `0 0 20px ${a.color}33` : 'none',
                      }}
                    >
                      <div style={{ fontSize:20, marginBottom:4 }}>{a.icon}</div>
                      <div style={{ fontSize:9.5, color: hovered === i ? a.color : 'rgba(255,255,255,.45)', fontWeight:700, fontFamily:'monospace' }}>{a.n}</div>
                      <div style={{ fontSize:9.5, color:'rgba(255,255,255,.6)', fontWeight:600, lineHeight:1.3, marginTop:2 }}>{a.title.split(' ').join('\n')}</div>
                    </div>
                  </div>
                ))}

                {/* connector + answer node */}
                <div style={{ display:'flex', alignItems:'center' }}>
                  <div style={{ width:28, height:1, background:'rgba(52,211,153,.35)', flexShrink:0 }}/>
                  <div style={{ width:0, height:0, borderTop:'4px solid transparent', borderBottom:'4px solid transparent', borderLeft:'5px solid rgba(52,211,153,.5)', flexShrink:0 }}/>
                </div>
                <div style={{
                  background:'rgba(52,211,153,.12)', border:'1px solid rgba(52,211,153,.3)',
                  borderRadius:12, padding:'10px 16px', textAlign:'center', flexShrink:0,
                }}>
                  <div style={{ fontSize:18, marginBottom:4 }}>✅</div>
                  <div style={{ fontSize:11, color:'#34d399', fontWeight:600 }}>Final Answer</div>
                </div>
              </div>

              {/* hover tooltip */}
              {hovered !== null && (
                <div style={{
                  marginTop:24, textAlign:'center',
                  animation:'fadeUp .2s ease-out',
                }}>
                  <div style={{
                    display:'inline-block',
                    background:'rgba(255,255,255,.06)', border:`1px solid ${AGENTS[hovered].color}44`,
                    borderRadius:12, padding:'12px 24px', maxWidth:480,
                  }}>
                    <span style={{ fontSize:12, color:'rgba(255,255,255,.5)' }}>{AGENTS[hovered].desc}</span>
                  </div>
                </div>
              )}
            </div>
          </Section>
        </section>

        {/* ── AGENTS GRID ─────────────────────────────────────────── */}
        <section id="agents" style={{ maxWidth:1140, margin:'0 auto', padding:'0 2rem 80px' }}>
          <Section>
            <div style={{ marginBottom:44 }}>
              <p style={{ fontSize:11, fontWeight:700, color:'rgba(255,255,255,.3)', textTransform:'uppercase', letterSpacing:'.12em', marginBottom:12 }}>
                The Pipeline
              </p>
              <h2 style={{ fontSize:'clamp(26px,4vw,36px)', fontWeight:800, letterSpacing:'-.03em' }}>
                7 agents, one reliable answer
              </h2>
            </div>

            <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:16 }}>
              {AGENTS.map((a, i) => (
                <Section key={i} delay={i * 60}>
                  <div className="glass agent-glass" style={{
                    borderRadius:18, padding:24, height:'100%',
                    transition:'all .25s', cursor:'default',
                  }}>
                    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:16 }}>
                      <div style={{
                        width:44, height:44, borderRadius:12, background:a.bg,
                        border:`1px solid ${a.color}33`,
                        display:'flex', alignItems:'center', justifyContent:'center', fontSize:20,
                      }}>{a.icon}</div>
                      <span style={{ fontSize:11, fontFamily:'monospace', color:'rgba(255,255,255,.25)', fontWeight:600 }}>{a.n}</span>
                    </div>
                    <h3 style={{ fontSize:14.5, fontWeight:700, color:'#fff', marginBottom:8 }}>{a.title}</h3>
                    <p style={{ fontSize:12.5, color:'rgba(255,255,255,.45)', lineHeight:1.75 }}>{a.desc}</p>
                    <div style={{ marginTop:14, height:2, borderRadius:999, background:`linear-gradient(90deg, ${a.color}55, transparent)` }}/>
                  </div>
                </Section>
              ))}

              {/* CTA card */}
              <Section delay={7 * 60}>
                <div onClick={onTryMe} style={{
                  borderRadius:18, padding:24, height:'100%',
                  background:'linear-gradient(135deg, rgba(99,102,241,.25), rgba(139,92,246,.2))',
                  border:'1px solid rgba(165,180,252,.2)',
                  display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
                  gap:14, cursor:'pointer', textAlign:'center',
                  transition:'all .25s', boxShadow:'0 8px 32px rgba(99,102,241,.2)',
                  minHeight:200,
                }}
                  onMouseEnter={e => { e.currentTarget.style.transform='translateY(-4px)'; e.currentTarget.style.boxShadow='0 16px 48px rgba(99,102,241,.35)' }}
                  onMouseLeave={e => { e.currentTarget.style.transform=''; e.currentTarget.style.boxShadow='0 8px 32px rgba(99,102,241,.2)' }}
                >
                  <div style={{ fontSize:32 }}>🚀</div>
                  <div style={{ fontSize:15, fontWeight:700, color:'#fff' }}>See it in action</div>
                  <div style={{ fontSize:12.5, color:'rgba(255,255,255,.5)', lineHeight:1.5 }}>
                    Upload your PDF and ask questions
                  </div>
                  <div style={{
                    marginTop:4, padding:'8px 22px',
                    background:'rgba(255,255,255,.12)', border:'1px solid rgba(255,255,255,.2)',
                    borderRadius:9, fontSize:13, color:'#fff', fontWeight:600,
                  }}>Try me →</div>
                </div>
              </Section>
            </div>
          </Section>
        </section>

        {/* ── FOOTER ──────────────────────────────────────────────── */}
        <footer style={{
          borderTop:'1px solid rgba(255,255,255,.06)',
          background:'rgba(255,255,255,.02)',
        }}>
          <div style={{
            maxWidth:1140, margin:'0 auto', padding:'24px 2rem',
            display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:12,
          }}>
            <div style={{ display:'flex', alignItems:'center', gap:10 }}>
              <div style={{
                width:26, height:26, borderRadius:7,
                background:'linear-gradient(135deg,#6366f1,#8b5cf6)',
                display:'flex', alignItems:'center', justifyContent:'center', fontSize:12,
              }}>🧠</div>
              <span style={{ fontSize:13, color:'rgba(255,255,255,.4)', fontWeight:500 }}>
                AC-RAG · Agent-Controlled Retrieval-Augmented Generation
              </span>
            </div>
            <span style={{ fontSize:12, color:'rgba(255,255,255,.25)' }}>
              SRKR Engineering College · Dept. of IT · 2025–26
            </span>
          </div>
        </footer>

      </div>
    </div>
  )
}
