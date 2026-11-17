import { useState } from 'react'
import { DEFAULT_API } from './api.js'

const STATUS = {
  checking: 'Connecting to the search engine',
  live: 'Connected',
  offline: 'Not reachable',
}

// Which search API this page talks to, and a way to point it elsewhere (a fresh tunnel
// URL, or a server on this machine).
export default function Connection({ apiBase, status, error, onChange }) {
  const [draft, setDraft] = useState(apiBase)

  return (
    <section className="panel" aria-labelledby="conn-h">
      <h2 id="conn-h">Connection</h2>
      <p className={`conn-status conn-${status}`}>
        <span className="conn-dot" aria-hidden="true" />
        {STATUS[status]}
      </p>
      <p className="conn-url">{apiBase}</p>
      {status === 'offline' && error && <p className="conn-error">{error}</p>}
      <details className="conn-edit">
        <summary>Use a different server</summary>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            if (draft.trim()) onChange(draft.trim())
          }}
        >
          <label htmlFor="api-url">Search API address</label>
          <input
            id="api-url"
            type="url"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="https://example.trycloudflare.com"
            spellCheck="false"
          />
          <div className="conn-actions">
            <button type="submit">Connect</button>
            <button
              type="button"
              className="quiet"
              onClick={() => {
                setDraft(DEFAULT_API)
                onChange(null)
              }}
            >
              Reset to {DEFAULT_API.replace(/^https?:\/\//, '')}
            </button>
          </div>
        </form>
      </details>
    </section>
  )
}
