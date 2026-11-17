import { fmtMs, percentile } from './format.js'

// Latency of every search in this visit: server time and browser round trip, with the
// median and 95th percentile once there are enough searches for them to mean something.
export default function History({ items, onPick }) {
  if (!items.length) return null
  const trips = items.map((h) => h.clientMs).filter((v) => v != null)
  const max = Math.max(...items.map((h) => h.clientMs ?? h.serverMs))

  return (
    <section className="panel" aria-labelledby="history-h">
      <h2 id="history-h">This session</h2>
      {trips.length >= 3 && (
        <p className="history-summary">
          Round trip: median <strong>{fmtMs(percentile(trips, 0.5))}</strong>, 95th percentile{' '}
          <strong>{fmtMs(percentile(trips, 0.95))}</strong> over {trips.length} searches
        </p>
      )}
      <ol className="history" reversed>
        {[...items].reverse().map((h) => (
          <li key={h.id}>
            <button type="button" className="history-q" onClick={() => onPick(h.q)} title="Search again">
              {h.q}
            </button>
            <span className="history-bar" aria-hidden="true">
              <span className="history-server" style={{ width: `${(h.serverMs / max) * 100}%` }} />
              {h.clientMs != null && (
                <span
                  className="history-net"
                  style={{ width: `${(Math.max(0, h.clientMs - h.serverMs) / max) * 100}%` }}
                />
              )}
            </span>
            <span className="history-ms">
              {fmtMs(h.serverMs)}
              {h.clientMs != null && <span className="history-trip"> / {fmtMs(h.clientMs)}</span>}
            </span>
          </li>
        ))}
      </ol>
      <p className="history-key">
        <span className="swatch seg-server" aria-hidden="true" /> server
        <span className="swatch seg-net" aria-hidden="true" /> network and browser
        {items.some((h) => h.recorded) && <span className="history-note">Recorded searches show the times measured when they were recorded.</span>}
      </p>
    </section>
  )
}
