"""Create the README figures from the trained checkpoints and result files."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentchart3d.charting import dental_chart  # noqa: E402
from dentchart3d.evaluate import CHUNK, load_model  # noqa: E402
from dentchart3d.explain import saliency  # noqa: E402
from dentchart3d.inference import predict_points  # noqa: E402
from dentchart3d.labels import class_to_fdi  # noqa: E402
from dentchart3d.train import load_split  # noqa: E402

FIG = Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)
RES = Path("results")
TOOTH = ["gingiva", "CI", "LI", "C", "PM1", "PM2", "M1", "M2", "M3"]
NAMES = [TOOTH[0]] + [f"R-{t}" for t in TOOTH[1:]] + [f"L-{t}" for t in TOOTH[1:]]
CMAP = plt.get_cmap("tab20")
COLORS = np.array([(0.82, 0.82, 0.82, 1.0)] + [CMAP(i % 20) for i in range(16)])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def curves():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, col in [("pointnet", "#8a8a8a"), ("dgcnn", "#1f4e79")]:
        p = RES / f"{name}_train_log.json"
        if not p.exists():
            continue
        log = json.loads(p.read_text())
        ax[0].plot([r["epoch"] for r in log], [r["train_loss"] for r in log], color=col, label=name.upper())
        v = [r for r in log if "val_miou" in r]
        ax[1].plot([r["epoch"] for r in v], [r["val_miou"] for r in v], color=col, marker="o", ms=3, label=name.upper())
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].set(title="Validation mIoU", xlabel="epoch", ylim=(0, 1))
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=160)


def per_tooth(summary):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.4
    x = np.arange(17)
    for k, (name, col) in enumerate([("pointnet", "#9a9a9a"), ("dgcnn", "#1f4e79")]):
        if name in summary:
            vals = [summary[name]["per_class_iou"][str(c)] for c in range(17)]
            ax.bar(x + (k - 0.5) * w, vals, w, color=col, label=name.upper())
    ax.set_xticks(x, NAMES, rotation=60, ha="right")
    ax.set(ylabel="IoU (test)", ylim=(0, 1), title="Per-class IoU on held-out test scans (R/L = patient's right/left)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "per_tooth_iou.png", dpi=160)


def robustness(summary):
    conds = ["clean", "density_25pct", "noise_0.01", "rotate_30", "rotate_90"]
    labels = ["clean", "25% point\ndensity", "noise\n(σ=0.01)", "rotated\n30°", "rotated\n90°"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.38
    x = np.arange(len(conds))
    for k, (name, col) in enumerate([("pointnet", "#9a9a9a"), ("dgcnn", "#1f4e79")]):
        if name in summary:
            ax.bar(x + (k - 0.5) * w, [summary[name]["robustness_miou"][c] for c in conds], w, color=col, label=name.upper())
    ax.set_xticks(x, labels)
    ax.set(ylabel="mIoU", ylim=(0, 1), title="Robustness to scan variations (test set)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", dpi=160)


def examples(model_name, data, min_points=40):
    model, _ = load_model(model_name, "checkpoints")
    test = data["test"]
    picks = [next(i for i in test if i["jaw"] == "upper"), next(i for i in test if i["jaw"] == "lower")]
    fig, ax = plt.subplots(2, 2, figsize=(9, 8.5))
    for r, it in enumerate(picks):
        xyz, nrm, gt = it["xyz"][:16384], it["normals"][:16384], it["cls"][:16384]
        pred = predict_points(model, xyz, nrm, it["jaw"], chunk=CHUNK[model_name], votes=2)
        for c, (lab, title) in enumerate([(gt, "Ground truth"), (pred, f"{model_name.upper()} prediction")]):
            a = ax[r, c]
            order = np.argsort(xyz[:, 2])  # both jaws face +z: draw from below to the occlusal surface
            a.scatter(xyz[order, 0], xyz[order, 1], c=COLORS[lab[order]], s=0.6, linewidths=0)
            a.set_aspect("equal")
            a.axis("off")
            a.set_title(f"{title} - {it['jaw']} jaw")
    fig.suptitle("Occlusal view of held-out test scans (Teeth3DS+, CC BY-NC-ND 4.0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "prediction_example.png", dpi=150)
    # dental chart for the lower example
    it = picks[1]
    pred = predict_points(model, it["xyz"], it["normals"], it["jaw"], chunk=CHUNK[model_name], votes=2)
    rows_p = dental_chart(pred, it["jaw"], min_points)
    rows_g = dental_chart(it["cls"], it["jaw"], 1)
    fig, ax = plt.subplots(figsize=(10, 2.4))
    order = list(range(8, 0, -1)) + list(range(9, 17))  # patient right molars ... left molars (as seen on a chart)
    for k, cls in enumerate(order):
        rp, rg = rows_p[cls - 1], rows_g[cls - 1]
        for y, row, lab in [(1, rg, "truth"), (0, rp, "model")]:
            col = "#2e7d32" if row["present"] else "#d0d0d0"
            ax.add_patch(plt.Rectangle((k, y), 0.92, 0.85, color=col))
            ax.text(k + 0.46, y + 0.55, row["fdi"], ha="center", va="center", fontsize=9, color="white" if row["present"] else "#555")
            ax.text(k + 0.46, y + 0.2, row.get("universal", ""), ha="center", va="center", fontsize=7, color="white" if row["present"] else "#555")
    ax.text(-0.3, 1.42, "Ground truth", ha="right", va="center")
    ax.text(-0.3, 0.42, "Model chart", ha="right", va="center")
    ax.set_xlim(-3.2, 16)
    ax.set_ylim(-0.1, 1.95)
    ax.axis("off")
    ax.set_title("Automated dental chart, lower jaw (green = present; FDI and Universal numbers)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "dental_chart_example.png", dpi=160)
    # saliency for the first molar on the patient's right
    sub = slice(0, CHUNK[model_name])
    xyz, nrm = it["xyz"][sub], it["normals"][sub]
    s = saliency(model, xyz, nrm, it["jaw"], target_class=6)
    p = predict_points(model, xyz, nrm, it["jaw"], chunk=CHUNK[model_name])
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    ax[0].scatter(xyz[:, 0], xyz[:, 1], c=COLORS[p], s=5, linewidths=0)
    ax[0].set_title("Prediction")
    sc = ax[1].scatter(xyz[:, 0], xyz[:, 1], c=s, cmap="magma", s=5, linewidths=0)
    ax[1].set_title(f"Saliency for tooth {class_to_fdi(6, it['jaw'])} (first molar)")
    for a in ax:
        a.set_aspect("equal")
        a.axis("off")
    fig.colorbar(sc, ax=ax[1], fraction=0.04, label="relative influence")
    fig.tight_layout()
    fig.savefig(FIG / "saliency_example.png", dpi=160)


if __name__ == "__main__":
    torch.set_num_threads(2)
    summary = json.loads((RES / "summary.json").read_text()) if (RES / "summary.json").exists() else {}
    curves()
    if summary:
        per_tooth(summary)
        robustness(summary)
    data = load_split("data/cache.pt", "results/splits.json")
    best = max(summary, key=lambda k: summary[k]["test"]["miou"]) if summary else "pointnet"
    examples(best, data, summary.get(best, {}).get("chart_min_points", 40) if summary else 40)
    print("figures written to", FIG)
