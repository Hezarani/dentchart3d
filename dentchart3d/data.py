"""Reading Teeth3DS+ scans, preprocessing to fixed-size point sets, and a PyTorch dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import map_labels


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Fast OBJ reader returning vertices (N,3) and triangle faces (M,3, 0-based)."""
    verts, faces = [], []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                p = line.split()[1:4]
                faces.append(tuple(int(x.split("/")[0]) - 1 for x in p))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from a triangle mesh."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.clip(norm, 1e-12, None)).astype(np.float32)


def normalise(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre on the centroid and scale to the unit sphere."""
    c = xyz.mean(axis=0)
    x = xyz - c
    s = float(np.linalg.norm(x, axis=1).max())
    return (x / s).astype(np.float32), c, s


def preprocess_scan(obj_path: Path, json_path: Path, n_points: int, rng: np.random.Generator) -> dict:
    """Load one scan and keep a random subset of n_points vertices with xyz, normals and labels."""
    v, f = read_obj(obj_path)
    meta = json.loads(Path(json_path).read_text())
    jaw = meta["jaw"]
    fdi = np.asarray(meta["labels"], dtype=np.int64)
    assert len(fdi) == len(v), f"label/vertex mismatch in {obj_path}"
    n = vertex_normals(v, f)
    xyz, centre, scale = normalise(v)
    idx = rng.choice(len(v), size=min(n_points, len(v)), replace=len(v) < n_points)
    return {
        "id": meta["id_patient"],
        "jaw": jaw,
        "xyz": xyz[idx],
        "normals": n[idx],
        "fdi": fdi[idx],
        "cls": map_labels(fdi[idx], jaw),
        "n_vertices": len(v),
    }


class ScanDataset(Dataset):
    """Point-set dataset. Each item is a random subset of `n_points` from a cached scan.

    Input features per point: x, y, z, nx, ny, nz, jaw_upper, jaw_lower (8 channels).
    """

    def __init__(self, items: list[dict], n_points: int = 2048, augment: bool = False, seed: int = 0):
        self.items = items
        self.n_points = n_points
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def features(xyz: np.ndarray, normals: np.ndarray, jaw: str) -> np.ndarray:
        onehot = np.zeros((len(xyz), 2), dtype=np.float32)
        onehot[:, 0 if jaw == "upper" else 1] = 1.0
        return np.concatenate([xyz, normals, onehot], axis=1)

    def __getitem__(self, i: int):
        it = self.items[i]
        idx = self.rng.choice(len(it["xyz"]), size=self.n_points, replace=False)
        xyz, nrm = it["xyz"][idx].copy(), it["normals"][idx].copy()
        if self.augment:
            xyz, nrm = augment(xyz, nrm, self.rng)
        x = self.features(xyz, nrm, it["jaw"])
        return torch.from_numpy(x.T.copy()), torch.from_numpy(it["cls"][idx].copy())


def rotation_z(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]], dtype=np.float32)


def augment(xyz: np.ndarray, nrm: np.ndarray, rng: np.random.Generator):
    """Small, label-preserving augmentations (no mirroring: left/right labels must not swap)."""
    r = rotation_z(rng.uniform(-15, 15))
    s = rng.uniform(0.9, 1.1)
    xyz = (xyz @ r.T) * s + rng.normal(0, 0.005, size=xyz.shape).astype(np.float32)
    nrm = nrm @ r.T
    return xyz.astype(np.float32), nrm.astype(np.float32)
