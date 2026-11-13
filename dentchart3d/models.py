"""Two point-cloud segmentation baselines: PointNet and DGCNN (EdgeConv)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSeg(nn.Module):
    """PointNet segmentation network (Qi et al., CVPR 2017), without input transforms."""

    def __init__(self, in_ch: int = 8, num_classes: int = 17):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(in_ch, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.glob = nn.Sequential(nn.Conv1d(128, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv1d(128 + 512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, N)
        f = self.local(x)
        g = self.glob(f).max(dim=2, keepdim=True)[0].expand(-1, -1, x.shape[2])
        return self.head(torch.cat([f, g], dim=1))  # (B, classes, N)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k nearest neighbours in feature space. x: (B, C, N) -> (B, N, k)."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    dist = -xx - inner - xx.transpose(2, 1)
    return dist.topk(k=k, dim=-1)[1]


def edge_features(x: torch.Tensor, k: int, idx: torch.Tensor | None = None) -> torch.Tensor:
    """EdgeConv input: concat(x_j - x_i, x_i) for the k neighbours j of each point i."""
    b, c, n = x.shape
    if idx is None:
        idx = knn(x, k)
    base = torch.arange(0, b, device=x.device).view(-1, 1, 1) * n
    flat = (idx + base).view(-1)
    xt = x.transpose(2, 1).contiguous()
    nb = xt.view(b * n, -1)[flat].view(b, n, k, c)
    xi = xt.view(b, n, 1, c).expand(-1, -1, k, -1)
    return torch.cat([nb - xi, xi], dim=3).permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)


class DGCNNSeg(nn.Module):
    """Dynamic Graph CNN segmentation (Wang et al., ACM TOG 2019), compact CPU-friendly version.

    The first EdgeConv builds the graph on xyz coordinates only; later layers use dynamic
    feature-space graphs, as in the original paper.
    """

    def __init__(self, in_ch: int = 8, num_classes: int = 17, k: int = 16, emb: int = 512):
        super().__init__()
        self.k = k
        self.ec1 = nn.Sequential(nn.Conv2d(2 * in_ch, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.ec2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.ec3 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.emb = nn.Sequential(nn.Conv1d(192, emb, 1, bias=False), nn.BatchNorm1d(emb), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(
            nn.Conv1d(192 + emb, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, N)
        n = x.shape[2]
        idx = knn(x[:, :3], self.k)  # spatial graph for the first layer
        x1 = self.ec1(edge_features(x, self.k, idx)).max(dim=-1)[0]
        x2 = self.ec2(edge_features(x1, self.k)).max(dim=-1)[0]
        x3 = self.ec3(edge_features(x2, self.k)).max(dim=-1)[0]
        local = torch.cat([x1, x2, x3], dim=1)
        g = self.emb(local).max(dim=2, keepdim=True)[0].expand(-1, -1, n)
        return self.head(torch.cat([local, g], dim=1))


def build_model(name: str, num_classes: int = 17) -> nn.Module:
    name = name.lower()
    if name == "pointnet":
        return PointNetSeg(num_classes=num_classes)
    if name == "dgcnn":
        return DGCNNSeg(num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")
