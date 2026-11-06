"""Fine-tune a cross-encoder on mined hard negatives (LCE loss).

# continue training the public MS MARCO reranker on our dense retriever's negatives
python scripts/train_reranker.py --collection msmarco-full --negatives runs/msmarco-full/training/negatives-dense-exact.jsonl \
    --init cross-encoder/ms-marco-MiniLM-L-6-v2 --lr 7e-6 --name ce-minilm-l6-dense-ft

# train from the base language model (no reranking head yet): a from-scratch comparison
python scripts/train_reranker.py --collection msmarco-full --negatives runs/msmarco-full/training/negatives-dense-exact.jsonl \
    --init nreimers/MiniLM-L6-H384-uncased --lr 3e-5 --name ce-minilm-l6-from-base

The result is a normal Hugging Face model directory: use it anywhere a reranker
model is accepted (rerank.py --model models/<name>, export_onnx.py --model ...).
"""

import argparse

import _bootstrap  # noqa: F401

from searchengine.config import PROJECT_ROOT, get_settings
from searchengine.data.collection import Collection
from searchengine.docstore import DocStore
from searchengine.training.negatives import read_negatives
from searchengine.training.train_ce import GroupDataset, TrainConfig, train_cross_encoder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--negatives", required=True)
    p.add_argument("--init", required=True, help="HF id or path: a cross-encoder, or a base encoder")
    p.add_argument("--name", required=True, help="output dir name under --models-dir")
    p.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=7e-6)
    p.add_argument("--batch-groups", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--loss", choices=["lce", "bce"], default="lce")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    docs = DocStore.open(coll, s.index_dir)
    records = read_negatives(args.negatives)
    data = GroupDataset(records, coll.train_queries(), docs, args.group_size, args.seed, all_ids=docs.ids)
    print(f"{len(data)} training groups from {args.negatives}")
    cfg = TrainConfig(
        init_model=args.init, out_dir=str(args.models_dir + "/" + args.name), epochs=args.epochs, lr=args.lr,
        batch_groups=args.batch_groups, group_size=args.group_size, max_length=args.max_length,
        max_steps=args.max_steps, loss=args.loss, save_every=args.save_every, seed=args.seed, device=args.device,
        extra={"negatives": str(args.negatives), "collection": coll.name},
    )
    train_cross_encoder(cfg, data)


if __name__ == "__main__":
    main()
