import numpy as np

from motion import count_motion_area


def _blank_mask(h=240, w=320):
    return np.zeros((h, w), dtype=np.uint8)


def test_empty_mask_is_not_detected():
    detected, area = count_motion_area(_blank_mask(), min_area=1500)
    assert detected is False
    assert area == 0.0

def test_small_motion_area_is_not_detected():
    mask = _blank_mask()
    mask[10:20, 10:20] = 255
    detected, area = count_motion_area(mask, min_area=1500)
    assert detected is False
    assert area == 0.0

def test_large_motion_area_is_detected():
    mask = _blank_mask()
    mask[50:150, 50:150] = 255
    detected, area = count_motion_area(mask, min_area=1500)
    assert detected is True
    assert area >= 9000


