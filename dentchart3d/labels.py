"""Label mapping between FDI tooth numbers and model classes.

The model predicts 17 classes for one jaw at a time:
    0      gingiva / background
    1..8   teeth on the patient's RIGHT side, central incisor (1) to third molar (8)
    9..16  teeth on the patient's LEFT side, central incisor (9) to third molar (16)

Upper jaw: right = FDI quadrant 1 (11-18), left = quadrant 2 (21-28).
Lower jaw: right = FDI quadrant 4 (41-48), left = quadrant 3 (31-38).
Using "patient side" for both jaws keeps the classes anatomically aligned.
"""
from __future__ import annotations

import numpy as np

NUM_CLASSES = 17
JAWS = ("upper", "lower")
_RIGHT_Q = {"upper": 1, "lower": 4}
_LEFT_Q = {"upper": 2, "lower": 3}


def fdi_to_class(fdi: int, jaw: str) -> int:
    """Map one FDI number (or 0 for gingiva) to a model class."""
    if fdi == 0:
        return 0
    q, pos = divmod(int(fdi), 10)
    if not 1 <= pos <= 8:
        raise ValueError(f"invalid FDI tooth position: {fdi}")
    if q == _RIGHT_Q[jaw]:
        return pos
    if q == _LEFT_Q[jaw]:
        return 8 + pos
    raise ValueError(f"FDI {fdi} does not belong to the {jaw} jaw")


def class_to_fdi(cls: int, jaw: str) -> int:
    """Map a model class back to an FDI number (0 = gingiva)."""
    cls = int(cls)
    if cls == 0:
        return 0
    if 1 <= cls <= 8:
        return _RIGHT_Q[jaw] * 10 + cls
    if 9 <= cls <= 16:
        return _LEFT_Q[jaw] * 10 + (cls - 8)
    raise ValueError(f"invalid class: {cls}")


def map_labels(fdi_labels: np.ndarray, jaw: str) -> np.ndarray:
    """Vectorised FDI -> class mapping for per-vertex labels."""
    lut = {f: fdi_to_class(f, jaw) for f in np.unique(fdi_labels)}
    out = np.zeros_like(fdi_labels, dtype=np.int64)
    for f, c in lut.items():
        out[fdi_labels == f] = c
    return out


def all_fdi(jaw: str) -> list[int]:
    """All 16 FDI tooth numbers of a jaw in class order 1..16."""
    return [class_to_fdi(c, jaw) for c in range(1, NUM_CLASSES)]
