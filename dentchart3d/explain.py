"""Point-level explanations with input-gradient saliency.

For a chosen tooth class, the saliency of each input point is the norm of the gradient of
that tooth's summed logit with respect to the point's coordinates. High values show which
surface regions drive the decision (e.g. cusp tips, contact areas, gingival margin).
"""
from __future__ import annotations

import numpy as np
import torch

from .data import ScanDataset


def saliency(model, xyz: np.ndarray, normals: np.ndarray, jaw: str, target_class: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(ScanDataset.features(xyz, normals, jaw).T.copy()).unsqueeze(0)
    x.requires_grad_(True)
    logits = model(x)[0]  # (classes, N)
    mask = logits.argmax(0) == target_class
    if mask.sum() == 0:
        mask = torch.ones_like(mask)
    score = logits[target_class][mask].sum()
    score.backward()
    g = x.grad[0, :3].norm(dim=0).numpy()
    # robust scaling: a handful of extreme points should not wash out the map
    return np.clip(g / (np.percentile(g, 98) + 1e-12), 0, 1)
