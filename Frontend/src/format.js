// Formatting and small calculations shared by the components.

const int = new Intl.NumberFormat('en-US')

export const fmtInt = (n) => (n == null ? '?' : int.format(Math.round(n)))

// Milliseconds with precision that matches their size: 0.43 ms, 14.3 ms, 249 ms, 1.11 s.
export function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return '?'
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`
  if (ms >= 100) return `${Math.round(ms)} ms`
  if (ms >= 10) return `${ms.toFixed(1)} ms`
  return `${ms.toFixed(2)} ms`
}

// Compact counts for the cascade: 8,841,823 stays exact; 1,105,228 becomes "1.1M".
export function fmtCount(n, exact = false) {
  if (n == null) return '?'
  if (exact || n < 10000) return fmtInt(n)
  if (n < 1e6) return `${Math.round(n / 1000)}K`
  return `${(n / 1e6).toFixed(1)}M`
}

export function percentile(values, p) {
  if (!values.length) return null
  const s = [...values].sort((a, b) => a - b)
  const i = (s.length - 1) * p
  const lo = Math.floor(i)
  const hi = Math.ceil(i)
  return s[lo] + (s[hi] - s[lo]) * (i - lo)
}

const STOP = new Set(
  'a an and are as at be by do does for from how in is it of on or the to was what when where which who why with'.split(' '),
)

export function queryTerms(q) {
  return [...new Set((q.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => t.length > 1 && !STOP.has(t)))]
}

// Split text into [{text, hit}] pieces where hit = a word starting with a query term
// (so "vaccine" also marks "vaccines"). Used to show why a passage matched.
export function highlight(text, terms) {
  if (!terms.length) return [{ text, hit: false }]
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const re = new RegExp(`\\b(${escaped.join('|')})[a-z0-9]*`, 'gi')
  const out = []
  let last = 0
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), hit: false })
    out.push({ text: m[0], hit: true })
    last = m.index + m[0].length
  }
  if (last < text.length) out.push({ text: text.slice(last), hit: false })
  return out
}

export const shortModel = (name) => (name ? name.split('/').pop() : '?')

export function fmtDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
}
