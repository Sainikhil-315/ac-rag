import { useState, useRef } from 'react'
import { Upload as UploadIcon, FileText, CheckCircle, AlertCircle } from 'lucide-react'

export default function Upload({ onUploaded }) {
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('idle') // idle | uploading | processing | done | error
  const [error, setError] = useState('')
  const [progress, setProgress] = useState('')
  const inputRef = useRef()

  const processFile = async (f) => {
    setFile(f)
    setStatus('uploading')
    setError('')

    const steps = ['Reading document...', 'Chunking content...', 'Building embeddings...', 'Loading pipeline...']
    let step = 0
    const interval = setInterval(() => {
      if (step < steps.length) { setProgress(steps[step]); step++ }
    }, 800)

    try {
      const form = new FormData()
      form.append('file', f)
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      clearInterval(interval)
      if (!res.ok) {
        const d = await res.json()
        throw new Error(d.detail || 'Upload failed')
      }
      const d = await res.json()
      setStatus('done')
      setTimeout(() => onUploaded(d.doc_name), 600)
    } catch (e) {
      clearInterval(interval)
      setStatus('error')
      setError(e.message)
    }
  }

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) processFile(f)
  }

  const onPick = (e) => { const f = e.target.files[0]; if (f) processFile(f) }

  const isIdle = status === 'idle'

  return (
    <div className="min-h-screen bg-bg flex flex-col items-center justify-center px-4">
      {/* Logo */}
      <div className="mb-12 text-center">
        <div className="flex items-center justify-center gap-3 mb-3">
          <div className="w-9 h-9 rounded-xl bg-accent/20 border border-accent/30 flex items-center justify-center text-lg">🧠</div>
          <span className="text-2xl font-semibold tracking-tight text-primary">AC-RAG</span>
        </div>
        <p className="text-secondary text-sm">Agent-Controlled Retrieval-Augmented Generation</p>
      </div>

      {/* Upload card */}
      <div className="w-full max-w-md">
        <div
          onClick={() => isIdle && inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); if (isIdle) setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`
            relative rounded-2xl border-2 border-dashed transition-all duration-200 p-12 text-center
            ${isIdle ? 'cursor-pointer hover:border-accent/60 hover:bg-accent/5' : ''}
            ${dragging ? 'border-accent bg-accent/10 scale-[1.01]' : 'border-border'}
            ${status === 'done' ? 'border-success/40 bg-success/5' : ''}
            ${status === 'error' ? 'border-danger/40 bg-danger/5' : ''}
          `}
        >
          <input ref={inputRef} type="file" accept=".pdf,.docx,.txt,.html,.md" className="hidden" onChange={onPick} />

          {status === 'idle' && (
            <div className="animate-fade-in">
              <div className="w-14 h-14 rounded-2xl bg-surface-2 border border-border flex items-center justify-center mx-auto mb-5">
                <UploadIcon size={24} className="text-secondary" />
              </div>
              <p className="text-primary font-medium mb-1">Drop your document here</p>
              <p className="text-secondary text-sm mb-4">or click to browse</p>
              <div className="flex gap-2 justify-center">
                {['PDF', 'DOCX', 'TXT', 'MD'].map(f => (
                  <span key={f} className="text-xs text-muted bg-surface border border-border rounded px-2 py-1 font-mono">{f}</span>
                ))}
              </div>
            </div>
          )}

          {(status === 'uploading' || status === 'processing') && (
            <div className="animate-fade-in">
              <div className="w-14 h-14 rounded-2xl bg-accent/10 border border-accent/20 flex items-center justify-center mx-auto mb-5">
                <FileText size={24} className="text-accent animate-pulse" />
              </div>
              <p className="text-primary font-medium mb-1">{file?.name}</p>
              <p className="text-secondary text-sm mb-5">{progress || 'Processing...'}</p>
              <div className="w-full bg-surface rounded-full h-1 overflow-hidden">
                <div className="h-full bg-accent rounded-full animate-pulse" style={{ width: '60%' }} />
              </div>
            </div>
          )}

          {status === 'done' && (
            <div className="animate-fade-in">
              <div className="w-14 h-14 rounded-2xl bg-success/10 border border-success/20 flex items-center justify-center mx-auto mb-5">
                <CheckCircle size={24} className="text-success" />
              </div>
              <p className="text-primary font-medium mb-1">Ready</p>
              <p className="text-secondary text-sm">{file?.name}</p>
            </div>
          )}

          {status === 'error' && (
            <div className="animate-fade-in">
              <div className="w-14 h-14 rounded-2xl bg-danger/10 border border-danger/20 flex items-center justify-center mx-auto mb-5">
                <AlertCircle size={24} className="text-danger" />
              </div>
              <p className="text-primary font-medium mb-1">Upload failed</p>
              <p className="text-secondary text-sm mb-4">{error}</p>
              <button onClick={(e) => { e.stopPropagation(); setStatus('idle'); setFile(null) }}
                className="text-xs text-accent hover:text-accent-hover transition-colors">Try again</button>
            </div>
          )}
        </div>

        <p className="text-center text-muted text-xs mt-4">
          Your document is processed locally — nothing is sent to external servers except the LLM API
        </p>
      </div>
    </div>
  )
}
