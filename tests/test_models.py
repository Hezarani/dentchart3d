import torch

from dentchart3d.models import build_model


def test_output_shapes():
    for name in ["pointnet", "dgcnn"]:
        m = build_model(name).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 8, 256))
        assert out.shape == (2, 17, 256)
