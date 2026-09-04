import { useState, useEffect } from 'react'
import { ThemeProvider } from './ThemeContext'
import Header from './components/Header'
import PipelineCanvas from './components/pipeline/PipelineCanvas'
import AgentInspector from './components/inspector/AgentInspector'
import Playground from './components/playground/Playground'
import Results from './components/results/Results'
import Landing from './components/Landing'

function lsGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v !== null ? JSON.parse(v) : fallback }
  catch { return fallback }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch {}
}

export default function App() {
  const [showLanding,   setShowLanding]   = useState(() => !lsGet('ac_doc_ready', false))
  const [activeTab,     setActiveTab]     = useState('Playground')
  const [selectedAgent, setSelectedAgent] = useState(null)

  // Pipeline live state
  const [agentStates, setAgentStates] = useState({})
  const [agentMeta,   setAgentMeta]   = useState({})

  // Run history — last 10 runs, newest first
  const [runHistory, setRunHistory] = useState(() => lsGet('ac_run_history', []))

  // Document state
  const [docName,  setDocName]  = useState(() => lsGet('ac_doc_name',  null))
  const [docReady, setDocReady] = useState(() => lsGet('ac_doc_ready', false))
  const [docList,  setDocList]  = useState(() => lsGet('ac_doc_list',  []))

  // Conversation history
  const [messages, setMessages] = useState(() => lsGet('ac_messages', []))

  useEffect(() => { lsSet('ac_doc_name',    docName)    }, [docName])
  useEffect(() => { lsSet('ac_doc_ready',   docReady)   }, [docReady])
  useEffect(() => { lsSet('ac_doc_list',    docList)    }, [docList])
  useEffect(() => { lsSet('ac_messages',    messages)   }, [messages])
  useEffect(() => { lsSet('ac_run_history', runHistory) }, [runHistory])

  // Reconcile persisted "document ready" state with the backend on load.
  // The backend keeps its loaded-document state in memory only (backend/api.py
  // _state dict) — a server restart (e.g. uvicorn --reload picking up a code
  // change) wipes it, but localStorage still says a document is loaded. Without
  // this check the UI shows a document as ready while every /ask call 400s with
  // "No document loaded" — confusing and looked like a hang/bug rather than what
  // it was (stale client state after a server restart).
  useEffect(() => {
    if (!docReady) return
    fetch('/api/status')
      .then(res => res.json())
      .then(data => {
        if (!data.ready) {
          setDocName(null); setDocReady(false); setDocList([])
        }
      })
      .catch(() => {
        // Backend unreachable — leave state as-is rather than clearing on a
        // transient network blip; the next /ask attempt will surface the error.
      })
    // Only on mount — subsequent doc state changes come from explicit
    // upload/clear actions, which already keep frontend and backend in sync.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleDocumentReady = (name, allDocs) => {
    setDocName(name)
    setDocReady(true)
    setDocList(allDocs || [name])
    // Don't clear messages on additional uploads — only on first upload or explicit clear
  }

  const handleDocumentCleared = () => {
    setDocName(null); setDocReady(false); setDocList([]); setMessages([])
    lsSet('ac_doc_name', null); lsSet('ac_doc_ready', false)
    lsSet('ac_doc_list', []); lsSet('ac_messages', [])
  }

  const handleAgentStateChange = (id, state) =>
    setAgentStates(prev => ({ ...prev, [id]: state }))

  const handleAgentMetaChange = (id, meta) =>
    setAgentMeta(prev => ({ ...prev, [id]: meta }))

  const handleResetAll = () => {
    setAgentStates({})
    setAgentMeta({})
    setSelectedAgent(null)
  }

  const handleRunMetaChange = (meta) =>
    setRunHistory(prev => [meta, ...prev].slice(0, 10))

  const lastRoute = runHistory[0]?.route ?? null

  if (showLanding) {
    return (
      <ThemeProvider>
        <Landing onTryMe={() => { setShowLanding(false); setActiveTab('Playground') }} />
      </ThemeProvider>
    )
  }

  return (
    <ThemeProvider>
      <div className="h-screen flex flex-col bg-bg font-sans overflow-hidden">
        <Header activeTab={activeTab} onTabChange={setActiveTab} />

        <div className="flex-1 flex overflow-hidden">
          {activeTab === 'Pipeline' && (
            <>
              <PipelineCanvas
                agentStates={agentStates}
                agentMeta={agentMeta}
                selectedAgent={selectedAgent}
                onSelectAgent={setSelectedAgent}
                lastRoute={lastRoute}
              />
              <AgentInspector
                selectedAgentId={selectedAgent}
                agentStates={agentStates}
                agentMeta={agentMeta}
              />
            </>
          )}

          {activeTab === 'Playground' && (
            <Playground
              docName={docName}
              docReady={docReady}
              docList={docList}
              onDocumentReady={handleDocumentReady}
              onDocumentCleared={handleDocumentCleared}
              agentStates={agentStates}
              onAgentStateChange={handleAgentStateChange}
              onAgentMetaChange={handleAgentMetaChange}
              onResetAll={handleResetAll}
              onRunMetaChange={handleRunMetaChange}
              messages={messages}
              onMessagesChange={setMessages}
            />
          )}

          {activeTab === 'Results' && (
            <Results agentMeta={agentMeta} runHistory={runHistory} />
          )}
        </div>
      </div>
    </ThemeProvider>
  )
}
