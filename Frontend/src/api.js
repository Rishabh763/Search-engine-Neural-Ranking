// Talking to the search API, and the recorded fallback for when it is offline.
//
// Where the API lives, in order: a ?api= link (a tunnel URL changes on every run, so
// sharing https://<site>/?api=https://<tunnel> is the way to point the deployed site
// at a laptop), the address saved from the settings panel, the build-time
// VITE_API_URL, then the local server.

const STORAGE_KEY = 'search-api-url'
export const DEFAULT_API = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

const clean = (url) => url.trim().replace(/\/+$/, '')

function readStored() {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export function saveApiBase(url) {
  try {
    if (url) localStorage.setItem(STORAGE_KEY, clean(url))
    else localStorage.removeItem(STORAGE_KEY)
  } catch {
    // storage blocked (private mode): the address still applies for this visit
  }
}

export function initialApiBase() {
  const fromLink = new URLSearchParams(window.location.search).get('api')
  if (fromLink) {
    saveApiBase(fromLink)
    return clean(fromLink)
  }
  return clean(readStored() || DEFAULT_API)
}

async function getJson(url, timeoutMs) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: ctrl.signal })
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`
      try {
        const body = await res.json()
        if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      } catch {
        // not JSON: keep the status line
      }
      throw new Error(detail)
    }
    return await res.json()
  } finally {
    clearTimeout(timer)
  }
}

export function fetchHealth(apiBase) {
  return getJson(`${apiBase}/health`, 5000)
}

// One search; the round trip is timed in the browser so it can be set against the
// server's own per-stage timings (the difference is network plus browser work).
export async function search(apiBase, { q, k, candidates, rerank, dedup }) {
  const params = new URLSearchParams({ q, k, candidates, rerank, dedup })
  const t0 = performance.now()
  const response = await getJson(`${apiBase}/search?${params}`, 60000)
  return { response, clientMs: performance.now() - t0 }
}

export async function fetchSamples() {
  try {
    return await getJson(`${import.meta.env.BASE_URL}samples.json`, 5000)
  } catch {
    return null
  }
}
