# Large-Scale Search Engine with Neural Ranking

Hybrid retrieval over the MS MARCO passage corpus (8.8M passages): BM25 and dense
retrieval for recall, rank fusion, then a cross-encoder reranker for precision.
Every claim is backed by standard IR metrics, significance tests, and measured
latency percentiles.

```text
Query
  ├── BM25 (Elasticsearch)        → top 1000
  └── Dense (bi-encoder + FAISS)  → top 1000
            ↓ Reciprocal Rank Fusion
       top-N candidates
            ↓ Cross-encoder (fine-tuned, ONNX int8)
          top 10
```

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Data pipeline, BM25 (Elasticsearch + local reference), evaluation harness | **done** |
| 2 | Cross-encoder reranking, FastAPI search endpoint | next |
| 3 | Dense retrieval (FAISS IVF-PQ), RRF fusion, BEIR zero-shot | planned |
| 4 | Cross-encoder fine-tuning with BM25 hard negatives | planned |
| 5 | ONNX + int8, latency vs candidate-depth curve, minimal UI | planned |

## Results

Filled in only from real runs. Reference points for a correct BM25 on the **full**
corpus (Anserini, k1=0.9, b=0.4), used as a sanity check:
MS MARCO dev MRR@10 ≈ 0.184, TREC DL19 nDCG@10 ≈ 0.506, DL20 nDCG@10 ≈ 0.480.

| System | Collection | Dev MRR@10 | Dev R@1000 | DL19 nDCG@10 | DL20 nDCG@10 | p95 latency |
|---|---|---:|---:|---:|---:|---:|
| BM25 (ES) | msmarco-full | | | | | |

## Quickstart (Windows PowerShell; same commands on Linux/macOS)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest                                   # 30 tests, no network or server needed

# 1. Sanity-check the install with an offline synthetic collection
python scripts/prepare_data.py --name toy --toy
python scripts/build_index.py --collection toy --backend local
python scripts/evaluate.py --collection toy --retriever bm25-local

# 2. Start Elasticsearch (Docker Desktop)
docker compose up -d
#   If ES exits with "max virtual memory areas too low":
#   wsl -d docker-desktop sysctl -w vm.max_map_count=262144

# 3. Build a 100K MS MARCO subset (first run downloads ~1 GB via ir_datasets)
python scripts/prepare_data.py --name msmarco-100k --corpus msmarco-passage `
    --eval-sets msmarco-dev dl19 dl20 --n-docs 100000 --n-queries 1000

# 4. Index and evaluate with both backends, then compare
python scripts/build_index.py --collection msmarco-100k --backend es
python scripts/build_index.py --collection msmarco-100k --backend local
python scripts/evaluate.py --collection msmarco-100k --retriever bm25-es
python scripts/evaluate.py --collection msmarco-100k --retriever bm25-local
python scripts/compare_runs.py runs/msmarco-100k/msmarco-dev/bm25-es.json `
                               runs/msmarco-100k/msmarco-dev/bm25-local.json

# 5. Full corpus (set ES heap to 4g in docker-compose.yml first)
python scripts/prepare_data.py --name msmarco-full --corpus msmarco-passage --eval-sets msmarco-dev dl19 dl20
python scripts/build_index.py --collection msmarco-full --backend es
python scripts/evaluate.py --collection msmarco-full --retriever bm25-es
```

`evaluate.py` writes a TREC run file and a JSON report (mean and per-query
metrics, batch QPS, single-query p50/p95/p99) to `runs/<collection>/<eval_set>/`.

## Layout

```text
searchengine/
  data/         registry (datasets + official metrics), collection format, subset builder, toy data
  retrieval/    Retriever interface, LocalBM25 (SciPy impact index), ElasticsearchBM25 (msearch)
  index/es.py   index definition, bulk loading, k1/b updates without reindexing
  eval/         metrics (validated against trec_eval), randomization test, TREC IO, latency
  text.py       analyzer mirroring ES `english` (for BM25 only; neural models get raw text)
scripts/        prepare_data, build_index, evaluate, compare_runs
tests/          unit tests; `pytest -m es` adds an ES-vs-local integration test
Frontend/       UI (reworked in phase 5)
```

## Design decisions

- **Two BM25 implementations.** Elasticsearch serves the full corpus. A local
  SciPy implementation (precomputed BM25 impacts in a CSR matrix; a query is one
  sparse product) is a transparent reference to cross-check ES and to test without
  a server. Both use Lucene's BM25 variant, so scores are directly comparable.
- **One ES shard.** BM25 statistics are per shard; a single shard keeps IDF exact.
- **BM25 parameters in the index's default similarity**, changeable by
  close/update/open, so k1/b tuning needs no reindexing (`evaluate.py --k1 --b`).
- **Subsets keep every judged passage** (including judged non-relevant ones,
  which are hard distractors on TREC DL) plus a seeded uniform fill. Subsets are
  for iteration only; headline numbers come from the full corpus.
- **Official metric conventions per dataset** (`searchengine/data/registry.py`):
  MRR@10 for MS MARCO dev (sparse labels make precision meaningless there),
  nDCG@10 with relevance ≥ 2 for binary metrics on TREC DL.
- **Metrics match `trec_eval -c`**, including tie-breaking, and are tested
  per query against `pytrec_eval`. Queries missing from a run score 0.
- **Paired randomization test + bootstrap CI** for every system comparison.

## Author

Rishabh Singh · M.Tech, Geoinformatics and Natural Resources Engineering, IIT Bombay
