import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchHealth, fetchSamples, initialApiBase, saveApiBase, search, DEFAULT_API } from './api.js'
import { fmtDate, fmtInt, queryTerms } from './format.js'
import Connection from './Connection.jsx'
import EngineInfo from './EngineInfo.jsx'
import History from './History.jsx'
import Result from './Result.jsx'
import Trace from './Trace.jsx'

const FALLBACK_EXAMPLES = [
  'how do vaccines train the immune system',
  'what causes the northern lights',
  'difference between a virus and bacteria',
]
const K_OPTIONS = [5, 10, 20]
const CANDIDATE_OPTIONS = [10, 20, 50, 100]
const REPO = 'https://github.com/Rishabh763/Search-engine-Neural-Ranking'

export default function App() {
  const [apiBase, setApiBase] = useState(initialApiBase)
  // The server check for one address; while it runs for a new address, status reads "checking".
  const [check, setCheck] = useState({ apiBase: null, status: 'checking', health: null, error: null })
  const [samples, setSamples] = useState(null)
  const [query, setQuery] = useState(() => new URLSearchParams(window.location.search).get('q') || '')
  const [params, setParams] = useState({ k: 10, candidates: 20, rerank: true, dedup: true })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])
  const pendingQuery = useRef(query)
  const seq = useRef(0)

  const status = check.apiBase === apiBase ? check.status : 'checking'
  const live = status === 'live'
  const health = live ? check.health : samples?.health ?? null

  useEffect(() => {
    let cancelled = false
    fetchHealth(apiBase)
      .then((h) => !cancelled && setCheck({ apiBase, status: 'live', health: h, error: null }))
      .catch((e) => {
        if (cancelled) return
        const why = e.name === 'AbortError' ? 'no answer within 5 seconds' : e.message
        setCheck({ apiBase, status: 'offline', health: null, error: why })
      })
    return () => {
      cancelled = true
    }
  }, [apiBase])

  useEffect(() => {
    let cancelled = false
    fetchSamples().then((s) => !cancelled && setSamples(s))
    return () => {
      cancelled = true
    }
  }, [])

  const runSearch = useCallback(
    async (q, p = params) => {
      const text = q.trim()
      if (!text) return
      const url = new URL(window.location.href)
      url.searchParams.set('q', text)
      window.history.replaceState(null, '', url)
      setError(null)

      if (!live) {
        const hit = samples?.samples.find((s) => s.query.toLowerCase() === text.toLowerCase())
        if (hit) {
          const recordedParams = { ...p, ...hit.params, k: Number(hit.params.k), candidates: Number(hit.params.candidates) }
          setResult({ response: hit.response, clientMs: hit.client_ms, params: recordedParams, recordedAt: samples.captured_utc })
          setHistory((h) => [
            ...h,
            { id: Date.now(), q: text, serverMs: hit.response.timings.total_ms, clientMs: hit.client_ms, recorded: true },
          ])
        } else {
          setResult(null)
          setError(
            samples
              ? 'The search engine is offline, so only the recorded searches are available. Pick one of them above the results.'
              : 'The search engine is offline and no recorded searches are bundled with this page.',
          )
        }
        return
      }

      const mine = ++seq.current
      setLoading(true)
      try {
        const { response, clientMs } = await search(apiBase, { q: text, ...p })
        if (mine !== seq.current) return // a newer search has started
        setResult({ response, clientMs, params: p, recordedAt: null })
        setHistory((h) => [...h, { id: Date.now(), q: text, serverMs: response.timings.total_ms, clientMs }])
      } catch (e) {
        if (mine !== seq.current) return
        const why = e.name === 'AbortError' ? 'no answer within 60 seconds' : e.message
        setError(`The search did not complete: ${why}. Check that the server and the tunnel are still running.`)
      } finally {
        if (mine === seq.current) setLoading(false)
      }
    },
    [apiBase, live, params, samples],
  )

  // Run a query that arrived in the link (?q=) once we know whether the server is up.
  useEffect(() => {
    if (status === 'checking' || !pendingQuery.current) return
    if (!live && !samples) return
    const q = pendingQuery.current
    pendingQuery.current = ''
    runSearch(q)
  }, [status, live, samples, runSearch])

  const changeParams = (patch) => {
    const next = { ...params, ...patch }
    if (next.candidates < next.k) next.candidates = CANDIDATE_OPTIONS.find((c) => c >= next.k)
    setParams(next)
    if (live && result) runSearch(result.response.query, next)
  }

  const changeApi = (url) => {
    saveApiBase(url)
    setApiBase(url ? url.replace(/\/+$/, '') : DEFAULT_API)
  }

  const pick = (q) => {
    setQuery(q)
    runSearch(q)
  }

  const examples = samples?.samples.map((s) => s.query) ?? FALLBACK_EXAMPLES
  const terms = result ? queryTerms(result.response.query) : []
  const nDocs = health?.n_docs

  return (
    <div className="page">
      <header className="masthead">
        <h1>Passage search</h1>
        <p className="tagline">
          Ask a question in plain words. A neural retriever narrows {nDocs ? fmtInt(nDocs) : 'millions of'} web passages to a few
          dozen, then a cross-encoder reads each one against your question and ranks them.
        </p>
      </header>

      <form
        className="searchbar"
        role="search"
        onSubmit={(e) => {
          e.preventDefault()
          runSearch(query)
        }}
      >
        <label htmlFor="q" className="sr-only">
          Your question
        </label>
        <input
          id="q"
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="how do vaccines train the immune system"
          autoComplete="off"
          maxLength={1000}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? 'Searching' : 'Search'}
        </button>
      </form>

      <div className="examples">
        <span className="examples-label">{live ? 'Try' : 'Recorded searches'}</span>
        {examples.map((q) => (
          <button key={q} type="button" className="chip" onClick={() => pick(q)}>
            {q}
          </button>
        ))}
      </div>

      <fieldset className="controls" disabled={!live}>
        <legend className="sr-only">Search settings</legend>
        <label>
          Results
          <select value={params.k} onChange={(e) => changeParams({ k: Number(e.target.value) })}>
            {K_OPTIONS.map((k) => (
              <option key={k}>{k}</option>
            ))}
          </select>
        </label>
        <label>
          Candidates to rerank
          <select value={params.candidates} onChange={(e) => changeParams({ candidates: Number(e.target.value) })}>
            {CANDIDATE_OPTIONS.filter((c) => c >= params.k).map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={params.rerank} onChange={(e) => changeParams({ rerank: e.target.checked })} />
          Rerank with the cross-encoder
        </label>
        <label className="check">
          <input type="checkbox" checked={params.dedup} onChange={(e) => changeParams({ dedup: e.target.checked })} />
          Fold near-identical copies
        </label>
      </fieldset>

      {status === 'offline' && (
        <p className="notice" role="status">
          The search engine runs on a laptop and is not reachable right now
          {samples ? `. These recorded searches were captured from it on ${fmtDate(samples.captured_utc)}, with their real timings.` : '.'}
        </p>
      )}

      <div className="layout">
        <main aria-live="polite" aria-busy={loading}>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {result && (
            <>
              <Trace
                key={`${result.response.query}-${result.clientMs}`}
                response={result.response}
                clientMs={result.clientMs}
                params={result.params}
                health={health}
                recordedAt={result.recordedAt}
              />
              {result.response.results.length ? (
                <ol className="results">
                  {result.response.results.map((r) => (
                    <Result key={r.id} r={r} terms={terms} reranked={result.response.reranked} />
                  ))}
                </ol>
              ) : (
                <p className="empty">No passages matched. Try fewer or more common words.</p>
              )}
            </>
          )}
          {!result && !error && (
            <section className="intro" aria-labelledby="intro-h">
              <h2 id="intro-h">What happens when you search</h2>
              <ol>
                <li>
                  <strong>Dense retrieval.</strong> Your question becomes a vector of 384 numbers. A compressed index finds the
                  passages whose vectors point the same way, without reading their words.
                </li>
                <li>
                  <strong>Reranking.</strong> A cross-encoder reads your question together with each candidate and scores how well
                  it answers. This is the slow, accurate step, so it only sees the top candidates.
                </li>
                <li>
                  <strong>Folding copies.</strong> The web repeats itself. Passages that are near-identical copies are folded into
                  one, so every result on the page says something new.
                </li>
              </ol>
              <p>Each answer shows how many passages survived each step and where the time went.</p>
            </section>
          )}
        </main>

        <aside className="rail">
          <Connection key={apiBase} apiBase={apiBase} status={status} error={check.error} onChange={changeApi} />
          <EngineInfo health={health} />
          <History items={history} onPick={pick} />
        </aside>
      </div>

      <footer className="footer">
        <a href={REPO}>Source code and benchmarks on GitHub</a>
        <span>Built by Rishabh Singh</span>
      </footer>
    </div>
  )
}
