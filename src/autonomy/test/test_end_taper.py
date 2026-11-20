import math

import numpy as np
import pytest

from autonomy.utils.costmap_speed import end_taper_speed, path_progress


def test_taper_is_the_braking_curve():
    assert end_taper_speed(1.40625, 0.8, 0.15) == pytest.approx(1.5)   # 1.5^2 / (2 * 0.8)
    assert end_taper_speed(0.25, 0.8, 0.15) == pytest.approx(math.sqrt(0.4))
    # braking from v at decel over the remaining distance ends at the path end
    for d in (0.1, 0.5, 1.0, 3.0):
        v = end_taper_speed(d, 0.8, 0.0)
        assert v * v / (2 * 0.8) == pytest.approx(d)


def test_taper_leaves_long_paths_alone():
    # local_planner's intermediate paths end ~10-12 m ahead: far above any max_speed
    assert end_taper_speed(10.0, 0.8, 0.15) > 1.5


def test_taper_creeps_at_and_past_the_end():
    assert end_taper_speed(0.0, 0.8, 0.15) == 0.15
    assert end_taper_speed(-0.3, 0.8, 0.15) == 0.15
    assert end_taper_speed(0.01, 0.8, 0.15) == 0.15


def test_remaining_arc_length_along_a_bent_path():
    # 5 m east then 5 m north, points every 0.1 m
    east = np.column_stack((np.arange(0.0, 5.0, 0.1), np.zeros(50)))
    north = np.column_stack((np.full(51, 5.0), np.arange(0.0, 5.0 + 1e-9, 0.1)))
    xy = np.vstack((east, north))
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    assert s[-1] - path_progress((2.03, 0.2), xy, s, 0) == pytest.approx(7.97)
    assert s[-1] - path_progress((5.1, 4.0), xy, s, 60) == pytest.approx(1.0)
    assert s[-1] - path_progress((5.0, 5.4), xy, s, 90) == pytest.approx(0.0)   # past the end
