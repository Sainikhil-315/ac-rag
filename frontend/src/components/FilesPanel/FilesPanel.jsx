import { useRef, useState, useEffect } from 'react'
import { FileText, Upload as UploadIcon, Trash2, Loader2, CheckCircle2 } from 'lucide-react'

export default function FilesPanel({ docList, docReady, onDocumentReady, onDocumentCleared }) {
  const inputRef = useRef(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/status')
      .then(r => r.json())
      .then(data => { if (data.ready) onDocumentReady(data.doc_name, data.doc_list) })
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleFiles = async (files) => {
    if (!files || !files.length) return
    setError(null)
    setUploading(true)
    try {
      for (const file of files) {
        const form = new FormData()
        form.append('file', file)
        const res = await fetch('/api/upload', { method: 'POST', body: form })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || 'Upload failed')
        onDocumentReady(data.doc_name, data.doc_list)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const handleClear = async () => {
    try {
      await fetch('/api/docs', { method: 'DELETE' })
    } finally {
      onDocumentCleared()
    }
  }

  return (
    <aside className="w-72 flex-shrink-0 border-l border-border bg-surface flex flex-col">
      <div className="p-4 border-b border-border">
        <h2 className="text-xs font-semibold text-muted uppercase tracking-wider">Knowledge Base</h2>
      </div>

      <div className="p-4">
        <label
          onDrop={(e) => { e.preventDefault(); handleFiles(e.dataTransfer.files) }}
          onDragOver={(e) => e.preventDefault()}
          className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-border rounded-xl py-6 text-center cursor-pointer hover:border-accent hover:bg-accent-bg transition-colors"
        >
          {uploading ? <Loader2 size={20} className="animate-spin text-accent" /> : <UploadIcon size={20} className="text-muted" />}
          <span className="text-xs text-secondary">
            {uploading ? 'Ingesting…' : 'Drop a document or click to upload'}
          </span>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.docx,.txt,.html,.md"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </label>
        {error && <p className="text-xs text-danger mt-2">{error}</p>}
      </div>

      <div className="flex-1 overflow-y-auto px-4 space-y-1.5">
        {(docList || []).map(name => (
          <div key={name} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-2 border border-border text-xs">
            <FileText size={13} className="text-muted flex-shrink-0" />
            <span className="flex-1 truncate text-secondary">{name}</span>
            <CheckCircle2 size={13} className="text-success flex-shrink-0" />
          </div>
        ))}
        {(!docList || docList.length === 0) && (
          <p className="text-xs text-muted text-center mt-4">No documents loaded.</p>
        )}
      </div>

      {docReady && (
        <div className="p-4 border-t border-border">
          <button
            onClick={handleClear}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-danger-bdr text-danger text-xs font-medium hover:bg-danger-bg transition-colors"
          >
            <Trash2 size={13} /> Clear all documents
          </button>
        </div>
      )}
    </aside>
  )
}
