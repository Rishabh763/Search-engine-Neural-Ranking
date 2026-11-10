# Full MS MARCO (8.8M passages) on Kaggle

GPU work runs on Kaggle's free T4s; Elasticsearch and serving stay on the laptop.
Three notebooks, each run as a **saved version** so its output persists and can be
attached to the next one.

| Notebook | Accelerator | Inputs | Does | Output | Time (est.) |
|---|---|---|---|---|---|
| `01_data_bm25` | None (CPU) | none | 8.8M collection, 200K training queries, doc store, BM25 runs | ~4 GB | 1–1.5 h |
| `02_dense_retrieval` | GPU T4 x2 | 01 | encode 8.8M passages, exact dense runs, 2 IVF-PQ indexes + recall | ~9 GB | 1.5–2 h |
| `03_train_reranker` | GPU T4 x2 | 01 + 02 | mine hard negatives, train 2 rerankers, evaluate + significance | ~3 GB | 1.5–2.5 h |

Notebook 01 uses no GPU quota. 02 + 03 use roughly 3–5 of the 30 weekly GPU hours.
Times are estimates; every notebook prints elapsed time and the hardware it ran on.

## Before you start

1. **The notebooks clone this repo from GitHub**, so push your latest commits first.
   The repo must be **public** (or set a `REPO` URL with a token in the first cell).
2. Kaggle account with **phone verification** (needed for GPU and internet access).

## Run order

For each notebook:

1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → upload the
   `.ipynb` from this folder (or paste its GitHub URL).
2. Right panel **Settings**: Accelerator as in the table; **Internet: On**.
3. For 02 and 03: **Add Input** → *Your Work / Notebooks* → pick the earlier notebook(s).
   Their outputs appear under `/kaggle/input/...`; the notebooks find them automatically.
4. **Save Version → Save & Run All (Commit)**. It runs in the background; you can close
   the tab. Check progress under the notebook's *Versions*.
5. When it finishes, open the version → **Output** to see the files.

Run 01, wait for it to finish, then 02, then 03 (each needs the previous outputs).

## What to check in the logs

- **01:** BM25 on the full corpus should land near the published anchors
  (dev MRR@10 ≈ 0.184, DL19 nDCG@10 ≈ 0.506, DL20 ≈ 0.480).
- **02:** encoding throughput (passages/s) per shard; IVF-PQ `recall@10_vs_exact` for 48
  vs 96 bytes per vector. That number decides which index the laptop serves.
- **03:** `positive_recall_at_depth` in the mining stats (how often the judged positive is
  in the dense top 100); training loss falling from ≈ ln(8) = 2.08; then the significance
  tables: does the fine-tuned reranker beat the public one on dev (6,980 queries) and on
  DL19+DL20 pooled (97 queries)?

## After Kaggle: bring results to the laptop

Download each notebook's output with the Kaggle CLI (`pip install kaggle six`, an API token in `KAGGLE_API_TOKEN`), e.g. `kaggle kernels output <user>/02-dense-retrieval -p kaggle_out\nb02`, then move:

| From | Files | Put them in |
|---|---|---|
| 03 | `results_bundle.zip` (trained models + all JSON reports) | unzip in the project root |
| 02 | `indexes/msmarco-full/` (dense: `meta.json`, `ids.json`, `emb_*.npy`, `ivfpq96.faiss` + `.json`; `docstore/`) | same path in the project |

The 6.8 GB of embedding shards (`emb_*.npy`) are needed: serving keeps only the
IVF-PQ index in RAM, but refines its top candidates with exact scores read from the
shards (memory-mapped), which takes recall@10 vs exact from ~0.76 back towards 1.

Then on the laptop (Elasticsearch for BM25 over 8.8M, and the full collection for passage text):

```bat
python scripts/prepare_data.py --name msmarco-full --corpus msmarco-passage --eval-sets msmarco-dev dl19 dl20
python scripts/build_index.py --collection msmarco-full --backend es
python scripts/evaluate.py --collection msmarco-full --retriever bm25-es --eval-sets dl19 dl20

python scripts/ann_recall.py --collection msmarco-full --index ivfpq96 --nprobe 64 --refine 0 100 200 500 1000
python scripts/evaluate.py --collection msmarco-full --retriever dense --dense-index ivfpq96 --nprobe 64 --refine 1000 --eval-sets dl19 dl20

:: The fine-tuned reranker is statistically equivalent to the public one (see the main
:: README), so serve the public model's ONNX export (scripts/export_onnx.py).
set SEARCH_COLLECTION=msmarco-full
set SEARCH_BACKEND=dense
set DENSE_INDEX=ivfpq96
set DENSE_NPROBE=64
set DENSE_REFINE=1000
set RERANK_MODEL=models\ms-marco-MiniLM-L-6-v2-onnx
set RERANK_ONNX_VARIANT=fp16
set RERANK_DEVICE=dml
uvicorn searchengine.api:app --port 8000
```

Laptop resources: the full Elasticsearch index takes ~3–4 GB of disk and indexing
takes ~30 min; give Docker at least 6 GB of RAM if possible.

## If something fails

- `Could not resolve host: github.com`: Internet is off for the notebook. Turn it on in the right sidebar (Session options), restart, rerun. The toggle needs a phone-verified Kaggle account.
- `The repo on GitHub is older than this notebook`: you haven't pushed. Commit and push from the laptop, run `!rm -rf /tmp/repo`, rerun the first cell.
- Kaggle runs whatever is on `main` at clone time, so push before every "Save & Run All".
- **Training shows tens of pairs/s instead of hundreds:** passage text is being read from `/kaggle/input` (slow random reads). Notebook 03 copies the collection to `/tmp/data` first; make sure that cell ran and `DATA_DIR` points there.
- **Out of GPU memory in 02/03:** the corpus alone is 6.8 GB on the GPU. Make sure
  nothing else holds GPU memory (restart the session) and keep the defaults.
- **Session ended mid-run (12 h limit):** re-run. Encoding skips finished shards;
  training saves a checkpoint every 2,000 steps under `models/<name>/checkpoint`.
- **`find_input` error:** the earlier notebook's output isn't attached (Add Input).
