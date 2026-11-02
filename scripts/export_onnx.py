"""Export a cross-encoder to ONNX (fp32 with fused transformer ops, plus dynamic int8).

python scripts/export_onnx.py
python scripts/export_onnx.py --model cross-encoder/ms-marco-MiniLM-L-4-v2

Writes models/<model-name>-onnx/ with model_fp32.onnx, model_int8.onnx, the
tokenizer, and export_meta.json. Every variant is validated against a freshly
loaded PyTorch model on real (query, passage) pairs from the collection:
max |score diff| and Spearman rank correlation. Then use it anywhere a model is
accepted, e.g.:

python scripts/rerank.py --collection msmarco-100k --model models/ms-marco-MiniLM-L-6-v2-onnx --onnx-variant int8
"""

import argparse
import json
import random
from pathlib import Path, PurePath

import _bootstrap  # noqa: F401

from searchengine.config import PROJECT_ROOT, get_settings
from searchengine.ranking.cross_encoder import DEFAULT_MODEL


def sample_pairs(collection: str | None, n: int, seed: int = 0) -> list[tuple[str, str]] | None:
    """Real (query, top BM25 passage) pairs: realistic lengths and score spread."""
    if not collection:
        return None
    from searchengine.data.collection import Collection
    from searchengine.docstore import DocStore
    from searchengine.eval.metrics import rank_documents
    from searchengine.eval.trec import read_run

    s = get_settings()
    coll = Collection.from_name(collection, s.data_dir)
    es_name = "dl19" if "dl19" in coll.eval_set_names else coll.eval_set_names[0]
    queries = coll.queries(es_name)
    run_path = s.runs_dir / coll.name / es_name / "bm25-es.run"
    if not run_path.exists():
        run_path = s.runs_dir / coll.name / es_name / "bm25-local.run"
    if not run_path.exists():
        return None
    run = read_run(run_path)
    docs = DocStore.open(coll, s.index_dir)
    rng = random.Random(seed)
    pairs = []
    for q in rng.sample(sorted(run), min(len(run), max(1, n // 20))):
        top = rank_documents(run[q])[:20]  # 20 candidates per query: a realistic reranking slate
        texts = docs.get_documents(top)
        pairs.extend((queries[q], texts[d].text) for d in top if d in texts)
    return pairs[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=None, help="default: models/<model-name>-onnx")
    p.add_argument("--collection", default="msmarco-100k", help="source of validation pairs ('' for synthetic)")
    p.add_argument("--n-validation", type=int, default=200)
    p.add_argument("--no-optimize", action="store_true")
    p.add_argument("--no-quantize", action="store_true")
    args = p.parse_args()

    from searchengine.ranking.onnx_ce import export_cross_encoder

    out = Path(args.out) if args.out else PROJECT_ROOT / "models" / f"{PurePath(args.model).name}-onnx"
    pairs = None
    try:
        pairs = sample_pairs(args.collection, args.n_validation)
    except Exception as e:
        print(f"(no collection pairs: {e}; using built-in examples)")
    print(f"exporting {args.model} -> {out}  (validation pairs: {len(pairs) if pairs else 'built-in'})")
    meta = export_cross_encoder(args.model, out, optimize=not args.no_optimize, quantize=not args.no_quantize, validation_pairs=pairs)
    print(json.dumps({k: meta[k] for k in ("exporter", "graph_fused", "files", "validation")}, indent=2))


if __name__ == "__main__":
    main()
