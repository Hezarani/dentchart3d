"""Fill README.md and docs/MODEL_CARD.md from results/summary.json."""
import json
from pathlib import Path

S = json.loads(Path("results/summary.json").read_text())
f = lambda v: f"{v:.3f}"  # noqa: E731
pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
best = max(S, key=lambda k: S[k]["test"]["miou"])
rows, jaws, rob = [], [], []
for name in ["pointnet", "dgcnn"]:
    if name not in S:
        continue
    t, r = S[name]["test"], S[name]["robustness_miou"]
    rows.append(f"| {name.upper()} | {f(t['miou'])} | {pct(t['acc'])} | {f(t['tooth_dice'])} | {pct(t['tooth_id_rate'])} | "
                f"{pct(t['chart_agreement'])} | {pct(t['chart_exact_share'])} | {pct(t['missing_tooth_sensitivity'])} |")
    jaws.append(f"{name.upper()} {f(t['miou_upper'])} / {f(t['miou_lower'])}")
    rob.append(f"| {name.upper()} | {f(r['clean'])} | {f(r['density_25pct'])} | {f(r['noise_0.01'])} | {f(r['rotate_30'])} | {f(r['rotate_90'])} |")
t = S[best]["test"]
rep = {"{BEST}": best.upper(), "{BEST_MIOU}": f(t["miou"]), "{BEST_TID}": pct(t["tooth_id_rate"]),
       "{BEST_CHART}": pct(t["chart_agreement"]), "{N_TEST}": str(S[best]["n_test_scans"]),
       "{RESULT_ROWS}": "\n".join(rows), "{JAW_ROWS}": "; ".join(jaws) + " (upper / lower)", "{ROB_ROWS}": "\n".join(rob)}
for src, dst in [("scripts/README.template.md", "README.md"), ("scripts/MODEL_CARD.template.md", "docs/MODEL_CARD.md")]:
    txt = Path(src).read_text()
    for k, v in rep.items():
        txt = txt.replace(k, v)
    Path(dst).write_text(txt)
print("README.md and docs/MODEL_CARD.md written; best =", best)
