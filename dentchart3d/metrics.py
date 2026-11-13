"""Segmentation and charting metrics."""
from __future__ import annotations

import numpy as np

from .labels import NUM_CLASSES


def per_class_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """IoU per class; NaN for classes absent from both prediction and ground truth."""
    iou = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p, g = pred == c, gt == c
        union = np.logical_or(p, g).sum()
        if union:
            iou[c] = np.logical_and(p, g).sum() / union
    return iou


def tooth_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Dice over the teeth present in the ground truth (gingiva excluded)."""
    vals = []
    for c in np.unique(gt):
        if c == 0:
            continue
        p, g = pred == c, gt == c
        vals.append(2 * np.logical_and(p, g).sum() / (p.sum() + g.sum()))
    return float(np.mean(vals)) if vals else float("nan")


def tooth_identification_rate(pred: np.ndarray, gt: np.ndarray) -> float:
    """Share of ground-truth teeth whose points are mostly given the correct tooth label."""
    hits, total = 0, 0
    for c in np.unique(gt):
        if c == 0:
            continue
        total += 1
        lab, cnt = np.unique(pred[gt == c], return_counts=True)
        hits += int(lab[np.argmax(cnt)] == c)
    return hits / total if total else float("nan")


def presence(labels: np.ndarray, min_points: int) -> np.ndarray:
    """Boolean vector (16,) telling which tooth classes 1..16 are present."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)[1:]
    return counts >= min_points


def charting_scores(pred: np.ndarray, gt: np.ndarray, min_points: int) -> dict:
    """Agreement between the predicted dental chart (present/missing per position) and the ground truth."""
    p, g = presence(pred, min_points), presence(gt, 1)
    return {
        "chart_agreement": float((p == g).mean()),
        "missing_true": int((~g).sum()),
        "missing_detected": int((~g & ~p).sum()),
        "present_true": int(g.sum()),
        "present_detected": int((g & p).sum()),
        "chart_exact": bool((p == g).all()),
    }
