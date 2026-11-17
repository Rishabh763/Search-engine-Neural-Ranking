# Passage search: frontend

A single page for the search engine in this repository: ask a question, see the ranked
passages, and see what each stage did and cost (how many passages survived each step,
per-stage server time, network round trip, and a running median / p95 for the session).

React 19 + Vite, no other runtime dependencies. It talks to the FastAPI server
(`searchengine/api.py`): `GET /health` for the engine's configuration and `GET /search`
for results.

## Run it locally

```bat
:: terminal 1: the search API (from the project root, venv active, SEARCH_* / DENSE_* / RERANK_* set)
uvicorn searchengine.api:app --port 8000

:: terminal 2: the frontend
cd Frontend
npm install
npm run dev
```

Open http://localhost:5173. The API allows the local Vite servers by default
(`CORS_ORIGINS` in `searchengine/api.py`).

## Which server it talks to

In order: `?api=<url>` in the page link (also remembered by the browser), the address saved
under **Connection, Use a different server**, the build-time `VITE_API_URL`, then
`http://127.0.0.1:8000`.

## When the server is off: recorded searches

The engine needs ~11 GB of index and data and a GPU-backed reranker, so it runs on a laptop,
not on a free host. When the page cannot reach it, it shows recorded responses from
`public/samples.json` instead, labelled as recorded with their date and real timings.
Refresh them with the API running:

```bat
python scripts/capture_samples.py
```

## Deploy

1. **Frontend on Vercel** (free Hobby plan): push to GitHub, then in Vercel choose
   *Add New, Project*, import the repository, set **Root Directory** to `Frontend`
   (Vite is detected: build `npm run build`, output `dist`) and deploy. No environment
   variable is needed; set `VITE_API_URL` only if the API has a permanent address.
2. **API from the laptop through a Cloudflare quick tunnel** (no account needed):

   ```bat
   winget install --id Cloudflare.cloudflared
   set CORS_ORIGINS=https://<your-app>.vercel.app,http://localhost:5173
   uvicorn searchengine.api:app --port 8000
   :: in a second terminal
   cloudflared tunnel --url http://localhost:8000
   ```

   cloudflared prints `https://<random-words>.trycloudflare.com`. Share
   `https://<your-app>.vercel.app/?api=https://<random-words>.trycloudflare.com`.
   The tunnel lives while cloudflared runs and gets a new address every time, so send a
   fresh link for each demo. Anyone with the link can search (the API is read-only);
   the reranker serves one request at a time, so heavy traffic queues.
