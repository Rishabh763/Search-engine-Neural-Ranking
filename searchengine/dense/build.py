"""Resumable, sharded corpus encoding (library side of scripts/encode_corpus.py)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from searchengine.data.collection import Collection
from searchengine.pipeline import passage_text


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)  # a crash never leaves a half-written file


def encode_collection(
    coll: Collection,
    encoder,
    out: Path,
    shard_size: int = 100_000,
    environment: dict | None = None,
    log: Callable[[str], None] = print,
    progress: bool = True,
) -> dict:
    """Encode every passage into float16 shards under `out`; skip shards already done."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    cfg = encoder.config()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta["encoder"]["model"] != cfg["model"] or meta["shard_size"] != shard_size:
            raise ValueError(f"{out} holds a different model or shard size; delete it or use another model")
    else:
        meta = {"collection": coll.name, "encoder": cfg, "shard_size": shard_size, "n_docs": len(coll), "shards": []}
    if environment is not None:
        meta["environment"] = environment
    done = {sh["file"]: sh for sh in meta["shards"]}

    ids: list[str] = []
    batch: list[str] = []
    state = {"shard": 0, "secs": sum(sh["seconds"] for sh in meta["shards"])}

    def flush() -> None:
        name = f"emb_{state['shard']:05d}.npy"
        if name in done and (out / name).exists() and done[name]["n"] == len(batch):
            log(f"  {name}: already done, skipping")
        else:
            t0 = time.perf_counter()
            emb = encoder.encode_passages(batch, progress=progress)
            secs = max(time.perf_counter() - t0, 1e-9)
            np.save(out / name, emb.astype(np.float16))
            done[name] = {"file": name, "n": len(batch), "seconds": round(secs, 2), "docs_per_second": round(len(batch) / secs, 1)}
            meta["shards"] = [done[k] for k in sorted(done)]
            _write_json_atomic(meta_path, meta)
            state["secs"] += secs
            log(f"  {name}: {len(batch)} passages in {secs:.1f}s ({len(batch) / secs:.0f}/s)")
        state["shard"] += 1
        batch.clear()

    for d in coll.iter_docs():
        ids.append(d.id)
        batch.append(passage_text(d))
        if len(batch) == shard_size:
            flush()
    if batch:
        flush()

    with open(out / "ids.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(ids, f)
    n = sum(sh["n"] for sh in meta["shards"])
    meta["docs_per_second"] = round(n / state["secs"], 1) if state["secs"] else None
    _write_json_atomic(meta_path, meta)
    return meta
