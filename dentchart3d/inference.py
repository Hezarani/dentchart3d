"""Whole-scan inference by partitioning a dense point set into random chunks."""
from __future__ import annotations

import numpy as np
import torch

from .data import ScanDataset


@torch.no_grad()
def predict_points(model, xyz: np.ndarray, normals: np.ndarray, jaw: str, chunk: int = 2048,
                   votes: int = 1, seed: int = 0, return_logits: bool = False):
    """Predict a class for every point. Points are shuffled and split into chunks of `chunk` points,
    each chunk being a random, roughly uniform subsample of the whole jaw (so global context is kept).
    With votes > 1 the partition is repeated with different shuffles and logits are averaged."""
    model.eval()
    n = len(xyz)
    feats = ScanDataset.features(xyz, normals, jaw)
    acc = np.zeros((n, model.head[-1].out_channels), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for _ in range(votes):
        perm = rng.permutation(n)
        pad = (-n) % chunk
        perm = np.concatenate([perm, perm[:pad]])
        batch = torch.from_numpy(feats[perm].reshape(-1, chunk, feats.shape[1]).transpose(0, 2, 1).copy())
        logits = torch.cat([model(b) for b in torch.split(batch, 8)], dim=0)
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[1]).numpy()
        np.add.at(acc, perm, logits)
    pred = acc.argmax(1)
    return (pred, acc) if return_logits else pred
