"""Evaluate trained models on the held-out test scans, run robustness tests and make figures.

Usage:  python -m dentchart3d.evaluate --models pointnet dgcnn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .charting import chart_text, dental_chart
from .data import rotation_z
from .inference import predict_points
from .labels import NUM_CLASSES, class_to_fdi
from .metrics import charting_scores, per_class_iou, tooth_dice, tooth_identification_rate
from .models import build_model
from .train import load_split

CHUNK = {"pointnet": 2048, "dgcnn": 1024}


def load_model(name: str, ckpt_dir: str):
    ck = torch.load(Path(ckpt_dir) / f"{name}.pt", weights_only=False)
    m = build_model(name)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def scan_metrics(pred, gt, min_points):
    iou = per_class_iou(pred, gt)
    out = {"miou": float(np.nanmean(iou)), "acc": float((pred == gt).mean()),
           "tooth_dice": tooth_dice(pred, gt), "tooth_id_rate": tooth_identification_rate(pred, gt)}
    out.update(charting_scores(pred, gt, min_points))
    return out, iou


def perturb(xyz, nrm, cond, rng):
    if cond == "clean":
        return xyz, nrm
    if cond == "density_25pct":
        return xyz, nrm  # handled by a smaller chunk size at inference
    if cond == "noise_0.01":
        return (xyz + rng.normal(0, 0.01, xyz.shape)).astype(np.float32), nrm
    if cond.startswith("rotate_"):
        r = rotation_z(float(cond.split("_")[1]))
        return (xyz @ r.T).astype(np.float32), (nrm @ r.T).astype(np.float32)
    raise ValueError(cond)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pointnet", "dgcnn"])
    ap.add_argument("--cache", default="data/cache.pt")
    ap.add_argument("--splits", default="results/splits.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--points", type=int, default=16384)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[20, 40, 80, 120, 160, 240, 320],
                    help="candidate minimum point counts (of 16,384) for calling a tooth present; chosen on validation")
    ap.add_argument("--votes", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(2)
    data = load_split(a.cache, a.splits)
    test = data["test"]
    summary = {}
    for name in a.models:
        model, ck = load_model(name, a.ckpt_dir)
        # choose the charting threshold on the validation set only
        val_preds = [(predict_points(model, it["xyz"][: a.points], it["normals"][: a.points], it["jaw"],
                                     chunk=CHUNK[name], votes=1), it["cls"][: a.points]) for it in data["val"]]
        thr_scores = {t: float(np.mean([charting_scores(p, g, t)["chart_agreement"] for p, g in val_preds]))
                      for t in a.thresholds}
        min_points = max(thr_scores, key=thr_scores.get)
        rows, ious = [], []
        for it in test:
            xyz, nrm, gt = it["xyz"][: a.points], it["normals"][: a.points], it["cls"][: a.points]
            pred = predict_points(model, xyz, nrm, it["jaw"], chunk=CHUNK[name], votes=a.votes)
            m, iou = scan_metrics(pred, gt, min_points)
            m.update({"scan": f"{it['id']}_{it['jaw']}", "jaw": it["jaw"],
                      "chart": chart_text(dental_chart(pred, it["jaw"], min_points))})
            rows.append(m)
            ious.append(iou)
        ious = np.array(ious)
        per_class = {str(c): float(np.nanmean(ious[:, c])) for c in range(NUM_CLASSES)}
        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in ["miou", "acc", "tooth_dice", "tooth_id_rate", "chart_agreement"]}
        agg["chart_exact_share"] = float(np.mean([r["chart_exact"] for r in rows]))
        mt, md = sum(r["missing_true"] for r in rows), sum(r["missing_detected"] for r in rows)
        pt, pdx = sum(r["present_true"] for r in rows), sum(r["present_detected"] for r in rows)
        agg["missing_tooth_sensitivity"] = md / mt if mt else float("nan")
        agg["present_tooth_sensitivity"] = pdx / pt if pt else float("nan")
        for jaw in ("upper", "lower"):
            agg[f"miou_{jaw}"] = float(np.mean([r["miou"] for r in rows if r["jaw"] == jaw]))
        # robustness (fewer points for speed)
        rng = np.random.default_rng(0)
        rob = {}
        for cond in ["clean", "density_25pct", "noise_0.01", "rotate_30", "rotate_90"]:
            sc = []
            for it in test:
                xyz, nrm, gt = it["xyz"][:8192], it["normals"][:8192], it["cls"][:8192]
                xyz, nrm = perturb(xyz, nrm, cond, rng)
                # density test: each forward pass sees 4x fewer points spread over the whole jaw
                chunk = CHUNK[name] // 4 if cond == "density_25pct" else CHUNK[name]
                pred = predict_points(model, xyz, nrm, it["jaw"], chunk=chunk)
                sc.append(np.nanmean(per_class_iou(pred, gt)))
            rob[cond] = float(np.mean(sc))
        summary[name] = {"checkpoint_epoch": ck["epoch"], "val_miou": ck["val_miou"], "test": agg,
                         "chart_min_points": min_points, "val_chart_agreement_by_threshold": thr_scores,
                         "per_class_iou": per_class, "robustness_miou": rob, "n_test_scans": len(test)}
        Path(a.results, f"{name}_test_per_scan.json").write_text(json.dumps(rows, indent=1))
        print(name, json.dumps(summary[name]["test"], indent=1), json.dumps(rob))
    Path(a.results, "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
