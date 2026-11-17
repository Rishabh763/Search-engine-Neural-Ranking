import { useState } from 'react'
import { highlight } from './format.js'

// One passage. The passage reads as a document (serif, query words marked); the line
// under it gives the reranker's score and how far reranking moved the passage. The raw
// score is shown rather than sigmoid(score): the probability saturates near 1 for every
// good passage (8.30 and 6.64 both read "99.9%"), the score still tells them apart.
export default function Result({ r, terms, reranked }) {
  const [showCopies, setShowCopies] = useState(false)
  const moved = r.first_stage_rank - r.rank
  const copies = r.duplicates?.length || 0

  let movement = null
  if (reranked) {
    if (moved > 0) movement = <span className="move up">up from #{r.first_stage_rank} in the dense ranking</span>
    else if (moved < 0) movement = <span className="move down">down from #{r.first_stage_rank}</span>
    else movement = <span className="move same">same place as the dense ranking</span>
  }

  return (
    <li className="result">
      <span className="rank" aria-label={`Rank ${r.rank}`}>
        {r.rank}
      </span>
      <div className="result-body">
        {r.title && <h3 className="result-title">{r.title}</h3>}
        <p className="passage">
          {highlight(r.snippet, terms).map((part, i) =>
            part.hit ? <mark key={i}>{part.text}</mark> : <span key={i}>{part.text}</span>,
          )}
        </p>
        <div className="result-meta">
          {r.rerank_score != null && (
            <span
              className="score"
              title={`Probability of relevance ${r.relevance}: near 1 for every good passage, so the raw score separates them better`}
            >
              reranker score <strong>{r.rerank_score.toFixed(2)}</strong>
            </span>
          )}
          {movement}
          {!reranked && <span className="move same">dense score {r.first_stage_score.toFixed(3)}</span>}
          <span className="doc-id">passage {r.id}</span>
          {copies > 0 && (
            <button
              type="button"
              className="copies-toggle"
              aria-expanded={showCopies}
              onClick={() => setShowCopies((v) => !v)}
            >
              {copies} {copies === 1 ? 'copy' : 'copies'} folded in
            </button>
          )}
        </div>
        {showCopies && (
          <p className="copies-list">
            Near-identical passages hidden from this list: {r.duplicates.join(', ')}
          </p>
        )}
      </div>
    </li>
  )
}
