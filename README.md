# Large-Scale Search Engine with Neural Ranking

Hybrid retrieval over the MS MARCO passage corpus (8.8M passages): BM25 and dense
retrieval for recall, rank fusion, then a cross-encoder reranker for precision.
Every claim is backed by standard IR metrics, significance tests, and measured
latency percentiles, with the hardware recorded alongside every number.

```text
Query
  ├── BM25 (Elasticsearch)        → top 1000
  └── Dense (bi-encoder + FAISS)  → top 1000
            ↓ Reciprocal Rank Fusion
       top-N candidates
            ↓ Cross-encoder (PyTorch fp32/fp16/int8 or ONNX fp32/int8)
          top 10
```

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Data pipeline, BM25 (Elasticsearch + local reference, score-level parity), evaluation harness | **done** |
| 2 | Cross-encoder reranking, FastAPI search endpoint | **done** |
| 3 | Dense retrieval (FAISS flat / HNSW / IVF-PQ), RRF hybrid fusion | **built**, measuring |
| 5 | Latency: depth sweep, ONNX + int8, PyTorch int8, cross-machine benchmarks | **built**, measuring |
| 4 | Cross-encoder fine-tuning with BM25 hard negatives | planned |

## Results so far

All on the 100K-passage subset (`msmarco-100k`: every judged passage for the
sampled queries plus a random fill). TREC DL nDCG on this subset is close to
full-corpus values because all judged passages are kept; dev MRR is inflated and
used for relative comparisons only.

| System | DL19 nDCG@10 | DL20 nDCG@10 | DL19 R@1000 | Rerank latency p50 |
|---|---:|---:|---:|---:|
| BM25 (Elasticsearch, k1=0.9 b=0.4) | 0.482 | 0.477 | 0.912 | n/a (first stage ~7 ms at k=100) |
| BM25 → MiniLM-L6 cross-encoder @100 | **0.752** | **0.733** | 0.912 | 2,059 ms (6-core laptop CPU, PyTorch fp32) |

The reranking gain is significant (DL19: +0.270, 95% CI [+0.20, +0.34], p < 1e-4,
paired randomization test). Full-corpus sanity references (Anserini BM25):
dev MRR@10 ≈ 0.184, DL19 ≈ 0.506, DL20 ≈ 0.480; MiniLM-L6 model card ≈ 0.743 on DL19.

## Quickstart (Windows; same commands on Linux/macOS)

```bat
python -m venv .venv
.venv\Scripts\activate.bat              :: PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[all]"                 :: GPU machine: install CUDA torch FIRST (see below)
pytest

:: Elasticsearch (Docker Desktop)
docker compose up -d

:: 100K MS MARCO subset (first run downloads ~1 GB via ir_datasets)
python scripts/prepare_data.py --name msmarco-100k --corpus msmarco-passage --eval-sets msmarco-dev dl19 dl20 --n-docs 100000 --n-queries 1000
python scripts/build_index.py --collection msmarco-100k --backend es
python scripts/evaluate.py --collection msmarco-100k --retriever bm25-es
```

### Reranking, latency, and the quality/latency tradeoff

```bat
python scripts/rerank.py --collection msmarco-100k --first-stage bm25-es --depth 100 --eval-sets dl19 dl20
python scripts/depth_sweep.py --collection msmarco-100k --reranked "bm25-es+ms-marco-MiniLM-L-6-v2@100"

:: Faster CPU variants (which one wins depends on the CPU: measure, don't assume)
python scripts/rerank.py --collection msmarco-100k --first-stage bm25-es --depth 100 --eval-sets dl19 dl20 --quantize int8
python scripts/export_onnx.py
python scripts/rerank.py --collection msmarco-100k --first-stage bm25-es --depth 100 --eval-sets dl19 dl20 --model models/ms-marco-MiniLM-L-6-v2-onnx --onnx-variant int8

python scripts/plot_tradeoff.py --collection msmarco-100k --eval-set dl19
```

`depth_sweep.py` gets quality at every depth ≤ 100 from the one depth-100 run
(no extra model calls) and times each depth. `plot_tradeoff.py` draws every sweep
of the collection as nDCG vs latency curves, one per (variant, machine).

### Dense retrieval and hybrid fusion

```bat
python scripts/encode_corpus.py --collection msmarco-100k          :: resumable; GPU used if present
python scripts/build_dense_index.py --collection msmarco-100k --type flat
python scripts/build_dense_index.py --collection msmarco-100k --type hnsw
python scripts/build_dense_index.py --collection msmarco-100k --type ivfpq
python scripts/evaluate.py --collection msmarco-100k --retriever dense --dense-index flat
python scripts/evaluate.py --collection msmarco-100k --retriever hybrid-es
python scripts/compare_runs.py runs\msmarco-100k\dl19\bm25-es.json "runs\msmarco-100k\dl19\rrf(bm25-es+dense-flat:msmarco-MiniLM-L6-cos-v5).json" --metric recall@1000
```

### Search API

```bat
set SEARCH_BACKEND=hybrid-es
uvicorn searchengine.api:app --port 8000
:: http://127.0.0.1:8000/docs  (Swagger UI)
```

Backends: `es`, `local`, `dense`, `hybrid-es`, `hybrid-local`. Reranker via
`RERANK_MODEL` (HF id, local checkpoint, ONNX export dir, or `none`),
`RERANK_QUANTIZE=int8`, `RERANK_ONNX_VARIANT=int8|fp32`.

### Benchmarking across machines

```bat
python scripts/benchmark.py --collection msmarco-100k
python scripts/compare_benchmarks.py
```

`benchmark.py` times every first stage that exists (BM25 ES/local, each FAISS
index, hybrid) and every reranker variant the machine can run (PyTorch fp32,
fp16 on GPU, int8 on CPU, ONNX fp32/int8) on real queries and candidates. Results
go to `results/benchmarks/<hostname>_<time>.json` with the hardware fingerprint
(CPU, cores, GPU, battery state, library versions). Commit them; run the same
command on another machine and `compare_benchmarks.py` prints one table.

### GPU machines

Install the CUDA build of PyTorch *before* `pip install -e ".[all]"`. RTX 50-series
(Blackwell) GPUs need CUDA 12.8+ builds:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`--device auto` (the default everywhere) then uses the GPU in fp16 for both the
reranker and the bi-encoder. FAISS stays on CPU (`faiss-cpu`).

### Full corpus (8.8M passages)

```bat
python scripts/prepare_data.py --name msmarco-full --corpus msmarco-passage --eval-sets msmarco-dev dl19 dl20
python scripts/build_index.py --collection msmarco-full --backend es       :: ES heap 4g, Docker memory >= 8 GB
python scripts/encode_corpus.py --collection msmarco-full --batch-size 256 :: GPU: ~1 h; CPU: impractical (~1 day)
python scripts/build_dense_index.py --collection msmarco-full --type ivfpq :: flat would need ~13.5 GB RAM
```

## Layout

```text
searchengine/
  data/          registry (datasets + official metrics), collection format, subset builder, toy data
  text.py        analyzer mirroring ES `english` (BM25 only; neural models get raw text)
  index/es.py    index definition, bulk loading, k1/b updates without reindexing
  retrieval/     Retriever interface; LocalBM25, ElasticsearchBM25, DenseRetriever, RRFHybrid
  dense/         BiEncoder (prefix registry), resumable sharded encoding, FAISS flat/HNSW/IVF-PQ
  ranking/       CrossEncoderReranker (fp32 / fp16 / dynamic int8), OnnxCrossEncoderReranker + export
  pipeline.py    retrieve -> fetch text -> rerank, timed per stage; rerank_run for offline eval
  docstore.py    byte-offset random access to passages (ES backends use _mget)
  eval/          metrics (= trec_eval), randomization test, TREC IO, latency, depth sweep
  envinfo.py     hardware/software fingerprint stored in every report
  api.py         FastAPI /search, /health
  factory.py     builds retrievers / doc sources / rerankers / pipelines from settings
scripts/         prepare_data, build_index, evaluate, rerank, compare_runs,
                 encode_corpus, build_dense_index, export_onnx, depth_sweep, plot_tradeoff,
                 benchmark, compare_benchmarks, check_analyzer, check_bm25_parity, es_latency_probe
tests/           unit tests incl. tiny randomly initialized models (no downloads)
results/         committed benchmark results from each machine
```

## Design decisions and findings

- **Two BM25 implementations, verified against each other.** Matching Lucene took
  four fixes, each found by measurement (`check_analyzer.py`, `check_bm25_parity.py`):
  UAX#29-style tokenization, Lucene's Porter variant (Martin Porter's reference
  code, not the published algorithm), Lucene's 1-byte document-length
  quantization, and Elasticsearch's `(k1+1)` factor (kept by its
  LegacyBM25Similarity even though Lucene 8 dropped it).
- **`127.0.0.1`, not `localhost`.** On Windows + Docker Desktop, `localhost` added
  ~45 ms to every search request; ES itself needs ~3 ms.
- **Reranking cost is compute-bound and predictable**: MiniLM-L6 is ~21 MFLOP per
  token, ~2.1 GFLOP per (query, passage) pair, ~210 GFLOP for 100 candidates;
  a 6-core laptop CPU sustains ~100 GFLOPS, hence ~2 s per query. The levers are
  fewer candidates, int8, and smaller models; their effect is measured per machine.
- **"ONNX is faster" is not a law.** On one AVX-512 Xeon: PyTorch fp32 1.0x,
  ONNX fp32 0.47x, ONNX int8 1.0x, PyTorch dynamic int8 1.8x. Hence the benchmark
  script rather than a fixed recommendation.
- **ANN recall is measured tie-aware against exact search** (an approximate hit
  counts if its true score reaches the exact k-th score), separating index error
  from model error.
- **Hybrid fusion only helps if both retrievers are decent**: RRF gives rank-based
  votes, so a weak retriever's noise costs the strong one. Measured, not assumed.
- **Run files keep full float precision.** Rounding scores to 6 decimals turned
  close scores into ties and reordered rankings on re-read (caught by the depth sweep).
- **One ES shard** keeps BM25 statistics exact; **subsets keep every judged
  passage**; **official metric conventions per dataset**; **metrics match
  `trec_eval -c`**; **paired randomization tests** for every comparison.

## Author

Rishabh Singh · M.Tech, Geoinformatics and Natural Resources Engineering, IIT Bombay
