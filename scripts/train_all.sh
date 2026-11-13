#!/usr/bin/env bash
# Reproduce the two baselines reported in the README (CPU, about 1 hour in total).
set -e
python -m dentchart3d.train --model pointnet --epochs 50 --points 2048 --batch 8
python -m dentchart3d.train --model dgcnn --epochs 50 --points 1024 --batch 8
python -m dentchart3d.evaluate --models pointnet dgcnn
