"""Convert raw Teeth3DS+ scans into a compact cache of fixed-size point sets and a patient-level split.

Usage:  python scripts/prepare_data.py --raw data/raw --out data/cache.pt --points 16384
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentchart3d.data import preprocess_scan  # noqa: E402


def job(args):
    obj, js, n, seed = args
    return preprocess_scan(Path(obj), Path(js), n, np.random.default_rng(seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/cache.pt")
    ap.add_argument("--points", type=int, default=16384)
    ap.add_argument("--split-out", default="results/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    raw = Path(a.raw)
    tasks = []
    for jaw in ("upper", "lower"):
        for d in sorted((raw / jaw).iterdir()):
            obj, js = d / f"{d.name}_{jaw}.obj", d / f"{d.name}_{jaw}.json"
            if obj.exists() and js.exists():
                tasks.append((str(obj), str(js), a.points, a.seed + len(tasks)))
    print(f"{len(tasks)} scans found")
    with Pool(a.workers) as pool:
        items = pool.map(job, tasks, chunksize=4)
    # split per jaw at scan level (each scan comes from a different patient within a jaw): 70/15/15
    rng = np.random.default_rng(a.seed)
    split = {"train": [], "val": [], "test": []}
    for jaw in ("upper", "lower"):
        ids = sorted(i["id"] + "_" + jaw for i in items if i["jaw"] == jaw)
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(round(0.70 * n)), int(round(0.15 * n))
        split["train"] += ids[:n_tr]
        split["val"] += ids[n_tr:n_tr + n_va]
        split["test"] += ids[n_tr + n_va:]
    Path(a.split_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.split_out).write_text(json.dumps({k: sorted(v) for k, v in split.items()}, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, a.out)
    print({k: len(v) for k, v in split.items()}, "saved", a.out)


if __name__ == "__main__":
    main()
