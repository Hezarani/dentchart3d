"""Automated dental charting: turn a per-point segmentation into a tooth-level chart.

Each of the 16 tooth positions of a jaw is marked present when the model assigns at least
`min_points` points to it. Tooth numbers are given in FDI and, through the ToothNotation
library (https://github.com/Hezarani/toothnotation), also in the Universal and Palmer systems.
"""
from __future__ import annotations

import numpy as np

from .labels import NUM_CLASSES, class_to_fdi

try:  # optional dependency, same author
    from toothnotation import convert as _convert
except Exception:  # pragma: no cover
    _convert = None


def notation(fdi: int) -> dict:
    out = {"fdi": str(fdi)}
    if _convert is not None:
        out["universal"] = _convert(str(fdi), "fdi", "universal")
        out["palmer"] = _convert(str(fdi), "fdi", "palmer")
    return out


def dental_chart(labels: np.ndarray, jaw: str, min_points: int = 40) -> list[dict]:
    """Return one row per tooth position with presence, point share and notations."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = max(int(counts.sum()), 1)
    rows = []
    for cls in range(1, NUM_CLASSES):
        fdi = class_to_fdi(cls, jaw)
        rows.append({**notation(fdi), "class": cls, "present": bool(counts[cls] >= min_points),
                     "points": int(counts[cls]), "share": round(counts[cls] / total, 4)})
    return rows


def chart_text(rows: list[dict]) -> str:
    """Compact text chart, e.g. for a report: FDI(Universal) present/missing."""
    cells = []
    for r in rows:
        tag = r["fdi"] + (f"({r['universal']})" if "universal" in r else "")
        cells.append(f"{tag}:{'+' if r['present'] else '-'}")
    return "  ".join(cells)
