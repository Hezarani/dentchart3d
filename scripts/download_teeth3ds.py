"""Download Teeth3DS+ data parts from the official OSF repository (https://osf.io/xctdy/).

The dataset is licensed CC BY-NC-ND 4.0 (non-commercial research only). It is NOT included in
this repository and must be downloaded by each user from the official source.

Usage:  python scripts/download_teeth3ds.py --parts 1 --out data/
"""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

PART_NODES = {1: "5cmg3", 2: "xfcn9", 3: "hw7bj", 4: "n9bd7", 5: "7az58", 6: "2ybr4", 7: "gvjah"}


def part_url(node: str) -> str:
    listing = json.load(urllib.request.urlopen(f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/"))
    return listing["data"][0]["links"]["download"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    for p in a.parts:
        zpath = out / f"data_part_{p}.zip"
        if not zpath.exists():
            url = part_url(PART_NODES[p])
            print(f"downloading part {p} from {url}")
            urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out / "raw")
        print(f"part {p} extracted to {out / 'raw'}")


if __name__ == "__main__":
    main()
