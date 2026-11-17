import { fmtCount, fmtDate, fmtInt, fmtMs } from './format.js'

// How one query was answered: the cascade of counts (8.8M passages narrowed to the few on
// screen) and where its time went, stage by stage, including the network round trip the
// server cannot see.
export default function Trace({ response, clientMs, params, health, recordedAt }) {
  const t = response.timings
  const info = health?.retriever_info
  const n = health?.n_docs
  const isDense = response.retriever.startsWith('dense')
  const firstStage = response.reranked || response.dedup ? params.candidates : params.k

  const steps = [{ key: 'index', count: fmtCount(n, true), what: 'passages in the index' }]
  if (info?.nlist && info?.nprobe && n) {
    steps.push({
      key: 'scan',
      count: `~${fmtCount((n * info.nprobe) / info.nlist)}`,
      what: `scanned in ${fmtInt(info.nprobe)} of ${fmtInt(info.nlist)} index cells`,
    })
  }
  if (info?.refine) {
    steps.push({ key: 'refine', count: fmtInt(Math.max(info.refine, firstStage)), what: 'rescored with exact vectors' })
  }
  steps.push({
    key: 'rerank',
    count: fmtInt(firstStage),
    what: response.reranked ? 'read by the cross-encoder' : 'kept in dense order',
  })
  steps.push({
    key: 'shown',
    count: fmtInt(response.results.length),
    what: response.dedup && response.collapsed
      ? `shown, after folding in ${response.collapsed} ${response.collapsed === 1 ? 'copy' : 'copies'}`
      : 'shown',
  })

  const server = t.total_ms
  const total = clientMs ?? server
  const segments = [
    {
      key: 'retrieve',
      label: isDense ? 'Dense retrieval' : 'Retrieval',
      note: isDense ? 'encode the query, scan, rescore' : response.retriever,
      ms: t.retrieve_ms,
    },
    { key: 'fetch', label: 'Fetch passage text', note: 'from the document store', ms: t.fetch_ms },
    { key: 'rerank', label: 'Rerank', note: response.reranked ? 'cross-encoder on every candidate' : 'skipped', ms: t.rerank_ms },
    { key: 'dedup', label: 'Collapse copies', note: response.dedup ? 'near-duplicate check' : 'off', ms: t.dedup_ms ?? 0 },
  ]
  if (clientMs != null) {
    segments.push({ key: 'net', label: 'Network and browser', note: 'round trip minus server time', ms: Math.max(0, clientMs - server) })
  }
  const spoken = segments.map((s) => `${s.label} ${fmtMs(s.ms)}`).join(', ')

  return (
    <section className="trace" aria-label="How this query was answered">
      <ol className="cascade" style={{ '--n': steps.length }}>
        {steps.map((s) => (
          <li key={s.key}>
            <span className="cascade-count">{s.count}</span>
            <span className="cascade-what">{s.what}</span>
          </li>
        ))}
      </ol>

      <div className="timebar" role="img" aria-label={`Time per stage: ${spoken}`}>
        {segments.map((s) =>
          s.ms > 0 ? (
            <span
              key={s.key}
              className={`seg seg-${s.key}`}
              style={{ flexGrow: s.ms / total }}
              title={`${s.label}: ${fmtMs(s.ms)}`}
            />
          ) : null,
        )}
      </div>

      <dl className="legend">
        {segments.map((s) => (
          <div key={s.key} className="legend-item">
            <dt>
              <span className={`swatch seg-${s.key}`} aria-hidden="true" />
              {s.label}
            </dt>
            <dd>
              <span className="legend-ms">{fmtMs(s.ms)}</span>
              <span className="legend-note">{s.note}</span>
            </dd>
          </div>
        ))}
      </dl>

      <p className="totals">
        <strong>{fmtMs(server)}</strong> on the server
        {clientMs != null && (
          <>
            , <strong>{fmtMs(clientMs)}</strong> round trip from this browser
          </>
        )}
        {recordedAt && <span className="recorded-note">Recorded {fmtDate(recordedAt)}</span>}
      </p>
    </section>
  )
}
