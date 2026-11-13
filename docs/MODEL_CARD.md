# Model card: DentChart3D baselines (v0.1.0)

## Model details
- **Developer:** Hossein Boustani Hezarani, DDS (ORCID 0009-0007-8481-0498), 2026.
- **Models:** PointNet and DGCNN point-cloud segmentation networks (`dentchart3d/models.py`), trained from scratch.
- **Input:** one intraoral scan (upper or lower jaw) as points with coordinates, normals and a jaw indicator.
- **Output:** one of 17 classes per point (gingiva or tooth position), then a tooth-level dental chart.
- **Licence:** code MIT. Weights are not distributed because they derive from CC BY-NC-ND 4.0 data.

## Intended use
- Research and teaching on 3D dental image analysis, and a starting baseline for automated dental charting.
- **Not** for clinical diagnosis or treatment decisions. Not a medical device.

## Training and evaluation data
- Teeth3DS+ data part 1: 300 scans (150 upper, 150 lower). Split 210 / 44 / 46 (train / validation / test), fixed seed, per jaw. See `docs/DATA.md`.

## Performance (test set, 46 scans)

| Model | mIoU | Accuracy | Tooth Dice | Tooth identification | Chart agreement | Exact charts | Missing-tooth sensitivity |
|---|---|---|---|---|---|---|---|
| POINTNET | 0.687 | 84.7% | 0.814 | 90.7% | 95.8% | 56.5% | 79.8% |
| DGCNN | 0.705 | 86.2% | 0.830 | 92.2% | 96.1% | 63.0% | 81.5% |

Upper vs lower jaw mIoU: POINTNET 0.741 / 0.632; DGCNN 0.742 / 0.667 (upper / lower)

## Robustness (mIoU, no retraining)

| Model | Clean | 25% density | Noise σ=0.01 | Rotated 30° | Rotated 90° |
|---|---|---|---|---|---|
| POINTNET | 0.682 | 0.651 | 0.665 | 0.383 | 0.035 |
| DGCNN | 0.678 | 0.425 | 0.657 | 0.415 | 0.044 |

## Known limitations and risks
- Single dataset; no external validation, no subgroup analysis (age, sex and scanner are not in the public labels).
- Sensitive to large rotations: scans must be in the usual orientation (occlusal plane horizontal, front teeth forward).
- Third molars are rare in the data and are the least reliable classes.
- Charting depends on a minimum-points threshold chosen on the validation set (reported in results/summary.json); very small or partially scanned teeth can be missed, and fragments can be over-called.

## Ethical considerations
- The public data is de-identified by the dataset authors. No new patient data was collected.
- Any clinical use would require prospective validation against chair-side measurements, regulatory assessment and monitoring.
