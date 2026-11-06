r"""Cross-encoder fine-tuning on (query, positive, hard negatives) groups.

Loss: Localized Contrastive Estimation (Gao, Dai & Callan, ECIR 2021). Each training
example is a group of G passages for one query: 1 positive + (G-1) hard negatives.
The model scores all G pairs, s_1 (positive) ... s_G, and we minimize

    L = -log( exp(s_1) / sum_{j=1..G} exp(s_j) )        (softmax cross-entropy, target 0)

i.e. "rank the positive first among its own hard negatives". Compared with
pointwise BCE on each pair independently, the softmax couples the scores of one
query's passages, which matches what reranking actually needs (a good *order* per
query, not calibrated absolute scores), and every negative in the group gets a
gradient proportional to how much probability mass it steals from the positive:

    dL/ds_j = p_j - 1[j = 1],   p = softmax(s)

so easy negatives (p_j ~ 0) contribute nothing and hard ones dominate. `loss="bce"`
(pos = 1, negs = 0) is kept for comparison.

Engineering: mixed precision (fp16 autocast + gradient scaling) on CUDA, AdamW with
linear warmup/decay, gradient clipping, passage text fetched lazily per batch from
the DocStore, periodic checkpoints (Kaggle sessions end hard at 12 h).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from searchengine.pipeline import passage_text


@dataclass
class TrainConfig:
    init_model: str
    out_dir: str
    epochs: int = 1
    lr: float = 7e-6
    batch_groups: int = 16  # queries per step; pairs per step = batch_groups * group_size
    group_size: int = 8  # 1 positive + (group_size - 1) negatives
    max_length: int = 256
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    loss: str = "lce"  # "lce" | "bce"
    fp16: bool | None = None  # default: on for CUDA
    max_steps: int | None = None
    log_every: int = 50
    save_every: int = 2000
    seed: int = 13
    device: str = "auto"
    extra: dict = field(default_factory=dict)


class GroupDataset:
    """One item = (query text, [positive text, negative texts...]) of length group_size."""

    def __init__(self, records: list[dict], queries: dict[str, str], docs, group_size: int, seed: int, all_ids: list[str]):
        self.items = [r for r in records if r["qid"] in queries and r["pos"]]
        self.queries = queries
        self.docs = docs
        self.group_size = group_size
        self.seed = seed
        self.all_ids = all_ids  # for padding groups that have too few mined negatives
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def group(self, idx: int) -> tuple[str, list[str]]:
        r = self.items[idx]
        rng = random.Random(f"{self.seed}:{self.epoch}:{r['qid']}")
        pos = rng.choice(r["pos"])
        need = self.group_size - 1
        negs = rng.sample(r["neg"], need) if len(r["neg"]) >= need else list(r["neg"])
        while len(negs) < need:  # pad with random corpus passages (easy negatives)
            d = rng.choice(self.all_ids)
            if d != pos and d not in r["pos"]:
                negs.append(d)
        ids = [pos, *negs]
        texts = self.docs.get_documents(ids)
        missing = [d for d in ids if d not in texts]
        if missing:
            raise KeyError(f"passages missing from the doc store: {missing[:3]}")
        return self.queries[r["qid"]], [passage_text(texts[d]) for d in ids]


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_cross_encoder(cfg: TrainConfig, dataset: GroupDataset, log: Callable[[str], None] = print) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from searchengine.envinfo import environment_info
    from searchengine.ranking.cross_encoder import resolve_device

    if cfg.loss not in ("lce", "bce"):
        raise ValueError("loss must be 'lce' or 'bce'")
    _set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    use_amp = (device == "cuda") if cfg.fp16 is None else (cfg.fp16 and device == "cuda")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.init_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.init_model, num_labels=1).to(device)
    model.train()

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    steps_per_epoch = len(dataset) // cfg.batch_groups
    total = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    if total <= 0:
        raise ValueError(f"not enough training groups ({len(dataset)}) for batch_groups={cfg.batch_groups}")
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=use_amp)

    G = cfg.group_size
    target_lce = None
    history: list[dict] = []
    step, t_start = 0, time.perf_counter()
    win_loss, win_pairs, win_t = 0.0, 0, time.perf_counter()
    log(f"training {cfg.init_model} -> {out}  steps={total}  groups/step={cfg.batch_groups}  G={G}  "
        f"device={device} amp={use_amp} loss={cfg.loss}")

    def save(path: Path) -> None:
        model.save_pretrained(path)
        tok.save_pretrained(path)

    done = False
    for epoch in range(cfg.epochs):
        dataset.epoch = epoch
        order = list(range(len(dataset)))
        random.Random(f"{cfg.seed}:order:{epoch}").shuffle(order)
        for b in range(steps_per_epoch):
            idx = order[b * cfg.batch_groups : (b + 1) * cfg.batch_groups]
            groups = [dataset.group(i) for i in idx]
            qs = [q for q, ps in groups for _ in ps]
            ps = [p for _, pl in groups for p in pl]
            enc = tok(qs, ps, padding=True, truncation="longest_first", max_length=cfg.max_length, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.view(len(groups), G).float()
            if cfg.loss == "lce":
                if target_lce is None or target_lce.shape[0] != len(groups):
                    target_lce = torch.zeros(len(groups), dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, target_lce)
            else:
                y = torch.zeros_like(logits)
                y[:, 0] = 1.0
                loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            win_loss += float(loss.detach())
            win_pairs += len(ps)
            if step % cfg.log_every == 0 or step == total:
                n = cfg.log_every if step % cfg.log_every == 0 else step % cfg.log_every
                dt = time.perf_counter() - win_t
                rec = {"step": step, "loss": round(win_loss / n, 4), "lr": sched.get_last_lr()[0],
                       "pairs_per_s": round(win_pairs / dt, 1), "elapsed_min": round((time.perf_counter() - t_start) / 60, 1)}
                history.append(rec)
                log(f"  step {step}/{total}  loss {rec['loss']:.4f}  lr {rec['lr']:.2e}  {rec['pairs_per_s']:.0f} pairs/s  {rec['elapsed_min']} min")
                win_loss, win_pairs, win_t = 0.0, 0, time.perf_counter()
            if cfg.save_every and step % cfg.save_every == 0:
                save(out / "checkpoint")
                (out / "checkpoint" / "step.txt").write_text(str(step), encoding="utf-8")
            if step >= total:
                done = True
                break
        if done:
            break

    model.eval()
    save(out)
    secs = time.perf_counter() - t_start
    meta = {
        "config": asdict(cfg),
        "train_groups": len(dataset),
        "steps": step,
        "seconds": round(secs, 1),
        "pairs_per_second": round(step * cfg.batch_groups * G / secs, 1),
        "history": history,
        "environment": environment_info(),
    }
    with open(out / "train_meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    log(f"saved {out}  ({secs / 60:.1f} min, {meta['pairs_per_second']:.0f} pairs/s)")
    return meta
