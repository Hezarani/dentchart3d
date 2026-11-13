import numpy as np

from dentchart3d.charting import dental_chart
from dentchart3d.metrics import charting_scores, per_class_iou, tooth_dice, tooth_identification_rate


def test_perfect_prediction():
    gt = np.repeat(np.arange(17), 100)
    assert np.nanmean(per_class_iou(gt, gt)) == 1.0
    assert tooth_dice(gt, gt) == 1.0
    assert tooth_identification_rate(gt, gt) == 1.0
    assert charting_scores(gt, gt, min_points=10)["chart_exact"]


def test_missing_tooth_is_charted_missing():
    gt = np.repeat(np.arange(17), 100)
    gt[gt == 8] = 0  # third molar missing
    rows = dental_chart(gt, "upper", min_points=10)
    by_fdi = {r["fdi"]: r for r in rows}
    assert by_fdi["18"]["present"] is False
    assert by_fdi["11"]["present"] is True
