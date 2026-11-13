import numpy as np
import pytest

from dentchart3d.labels import all_fdi, class_to_fdi, fdi_to_class, map_labels


@pytest.mark.parametrize("jaw", ["upper", "lower"])
def test_round_trip(jaw):
    for c in range(17):
        assert fdi_to_class(class_to_fdi(c, jaw), jaw) == c


def test_patient_side_alignment():
    # first molars on the patient's right share one class in both jaws
    assert fdi_to_class(16, "upper") == fdi_to_class(46, "lower") == 6
    assert fdi_to_class(26, "upper") == fdi_to_class(36, "lower") == 14


def test_wrong_jaw_raises():
    with pytest.raises(ValueError):
        fdi_to_class(36, "upper")


def test_vectorised_mapping():
    lab = np.array([0, 11, 21, 18, 28])
    assert map_labels(lab, "upper").tolist() == [0, 1, 9, 8, 16]
    assert len(all_fdi("lower")) == 16
