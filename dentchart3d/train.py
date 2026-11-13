"""Train a segmentation model.

Usage:  python -m dentchart3d.train --model dgcnn --epochs 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ScanDataset
from .inference import predict_points
from .labels import NUM_CLASSES
from .metrics import per_class_iou
from .models import build_model


def load_split(cache: str, splits: str):
    items = torch.load(cache, weights_only=False)
    by_key = {f"{i['id']}_{i['jaw']}": i for i in items}
    sp = json.loads(Path(splits).read_text())
    return {k: [by_key[x] for x in v] for k, v in sp.items()}


def class_weights(items) -> torch.Tensor:
    counts = np.bincount(np.concatenate([i["cls"] for i in items]), minlength=NUM_CLASSES).astype(np.float64)
    w = 1.0 / np.sqrt(counts / counts.sum() + 1e-6)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def evaluate_miou(model, items, eval_points: int = 8192, chunk: int = 2048) -> float:
    scores = []
    for it in items:
        sub = slice(0, eval_points)
        pred = predict_points(model, it["xyz"][sub], it["normals"][sub], it["jaw"], chunk=chunk)
        scores.append(np.nanmean(per_class_iou(pred, it["cls"][sub])))
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dgcnn", choices=["pointnet", "dgcnn"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="data/cache.pt")
    ap.add_argument("--splits", default="results/splits.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    data = load_split(a.cache, a.splits)
    ds = ScanDataset(data["train"], n_points=a.points, augment=True, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=0)
    model = build_model(a.model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    w = class_weights(data["train"])
    Path(a.ckpt_dir).mkdir(exist_ok=True)
    Path(a.results).mkdir(exist_ok=True)
    log, best = [], -1.0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        losses = []
        for x, y in dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=w, label_smoothing=0.05)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "minutes": round((time.time() - t0) / 60, 2)}
        if ep % a.val_every == 0 or ep == a.epochs:
            row["val_miou"] = evaluate_miou(model, data["val"], chunk=a.points)
            if row["val_miou"] > best:
                best = row["val_miou"]
                torch.save({"model": a.model, "state_dict": model.state_dict(), "epoch": ep, "val_miou": best},
                           Path(a.ckpt_dir) / f"{a.model}.pt")
        log.append(row)
        print(json.dumps(row), flush=True)
        Path(a.results, f"{a.model}_train_log.json").write_text(json.dumps(log, indent=1))
    print(f"best val mIoU {best:.4f}")


if __name__ == "__main__":
    main()
