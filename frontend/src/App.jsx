import { useState } from 'react'
import { ThemeProvider } from './ThemeContext'
import Header from './components/Header'
import ChatList from './components/Sidebar/ChatList'
import ChatWindow from './components/Chat/ChatWindow'
import FilesPanel from './components/FilesPanel/FilesPanel'

function lsGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v !== null ? JSON.parse(v) : fallback }
  catch { return fallback }
}
function lsSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch {}
}

let sessionCounter = 0
const newSession = () => ({ id: `s${Date.now()}_${sessionCounter++}`, messages: [] })

export default function App() {
  const [sessions, setSessions] = useState(() => {
    const saved = lsGet('ac_sessions', [])
    return saved.length ? saved : [newSession()]
  })
  const [activeId, setActiveId] = useState(() => lsGet('ac_active_session', null) || sessions[0]?.id)

  const [docName, setDocName] = useState(() => lsGet('ac_doc_name', null))
  const [docReady, setDocReady] = useState(() => lsGet('ac_doc_ready', false))
  const [docList, setDocList] = useState(() => lsGet('ac_doc_list', []))

  const persist = (nextSessions) => { setSessions(nextSessions); lsSet('ac_sessions', nextSessions) }

  const activeSession = sessions.find(s => s.id === activeId) || sessions[0]

  const handleUpdateSession = (updater) => {
    persist(sessions.map(s => (s.id === activeSession.id ? updater(s) : s)))
  }

  const handleNewChat = () => {
    const s = newSession()
    persist([s, ...sessions])
    setActiveId(s.id)
    lsSet('ac_active_session', s.id)
  }

  const handleSelect = (id) => { setActiveId(id); lsSet('ac_active_session', id) }

  const handleDelete = (id) => {
    const remaining = sessions.filter(s => s.id !== id)
    const next = remaining.length ? remaining : [newSession()]
    persist(next)
    if (id === activeId) { setActiveId(next[0].id); lsSet('ac_active_session', next[0].id) }
  }

  const handleDocumentReady = (name, allDocs) => {
    setDocName(name); lsSet('ac_doc_name', name)
    setDocReady(true); lsSet('ac_doc_ready', true)
    const list = allDocs || [name]
    setDocList(list); lsSet('ac_doc_list', list)
  }

  const handleDocumentCleared = () => {
    setDocName(null); lsSet('ac_doc_name', null)
    setDocReady(false); lsSet('ac_doc_ready', false)
    setDocList([]); lsSet('ac_doc_list', [])
  }

  return (
    <ThemeProvider>
      <div className="h-screen flex flex-col bg-bg font-sans overflow-hidden">
        <Header />
        <div className="flex-1 flex overflow-hidden">
          <ChatList
            sessions={sessions}
            activeId={activeSession?.id}
            onSelect={handleSelect}
            onNew={handleNewChat}
            onDelete={handleDelete}
          />
          {activeSession && (
            <ChatWindow
              session={activeSession}
              onUpdateSession={handleUpdateSession}
              docReady={docReady}
            />
          )}
          <FilesPanel
            docList={docList}
            docReady={docReady}
            onDocumentReady={handleDocumentReady}
            onDocumentCleared={handleDocumentCleared}
          />
        </div>
      </div>
    </ThemeProvider>
  )
}
