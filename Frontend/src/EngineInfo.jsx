import { fmtInt, shortModel } from './format.js'

const DEVICE = { dml: 'GPU via DirectML', cuda: 'NVIDIA GPU', mps: 'Apple GPU', cpu: 'CPU' }

// What the engine is running, as reported by the server's /health.
export default function EngineInfo({ health }) {
  if (!health) return null
  const info = health.retriever_info
  const rr = health.reranker
  const isDense = health.retriever?.startsWith('dense')
  const indexName = health.retriever?.split(':')[0].replace(/^dense-/, '')

  const rows = [
    ['Collection', `${fmtInt(health.n_docs)} passages (${health.collection || 'unknown'})`],
  ]
  if (isDense && info) {
    rows.push(['Retriever', `${shortModel(info.encoder?.model)}, ${info.encoder?.dim}-d vectors`])
    const parts = [`${indexName} index`]
    if (info.nlist) parts.push(`${fmtInt(info.nprobe)} of ${fmtInt(info.nlist)} cells probed`)
    if (info.refine) parts.push(`top ${fmtInt(info.refine)} rescored exactly`)
    rows.push(['Vector search', parts.join(', ')])
  } else {
    rows.push(['Retriever', health.retriever])
  }
  if (rr) {
    const how = rr.runtime === 'onnxruntime' ? `ONNX ${rr.variant}` : rr.fp16 ? 'PyTorch fp16' : 'PyTorch'
    rows.push(['Reranker', `${shortModel(rr.model)}, ${how}, ${DEVICE[rr.device] || rr.device}`])
  } else {
    rows.push(['Reranker', 'none'])
  }
  if (health.machine) rows.push(['Machine', health.machine])
  if (health.defaults) {
    rows.push([
      'Defaults',
      `${health.defaults.candidates} candidates, copies ${health.defaults.dedup ? 'collapsed' : 'kept'}`,
    ])
  }

  return (
    <section className="panel" aria-labelledby="engine-h">
      <h2 id="engine-h">The engine</h2>
      <dl className="facts">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
