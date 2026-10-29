import math
from types import SimpleNamespace

import numpy as np
import pytest

from autonomy.controller import CONTROLLER_PARAMS, controller_param_problem
from autonomy.utils.goal_end import goal_end
from autonomy.utils.reference_trajectory import process_path


def path_tdm43(points):
    """The controller's (xy, unwrapped heading, arc length) for a path through `points`."""
    return process_path(SimpleNamespace(poses=[
        SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y))) for x, y in points]))


def straight_tdm43(length=6.0):
    """East from the origin, a point every 0.1 m."""
    return path_tdm43([(s, 0.0) for s in np.arange(0.0, length + 1e-9, 0.1)])


def end_tdm43(path, x, y, yaw, start_idx=None, yaw_tolerance=0.15):
    """goal_end for a robot at (x, y) heading yaw. The robot is looked for from start_idx on, by
    default a few points behind the nearest, like the controller's progress."""
    if start_idx is None:
        start_idx = max(0, int(np.argmin(np.hypot(*(path[0] - (x, y)).T))) - 5)
    return goal_end(np.array([x, y, yaw, 0.5, 0.0, 0.0]), *path, start_idx, 0.25, yaw_tolerance)


def test_inside_both_tolerances_the_goal_is_reached_short_of_the_end():
    end = end_tdm43(straight_tdm43(), 5.85, 0.05, 0.05)
    assert end.reached and not end.at_path_end
    assert end.distance == pytest.approx(math.hypot(0.15, 0.05))
    assert end.yaw_error == pytest.approx(0.05)


def test_at_the_wrong_heading_it_goes_on_to_the_end_of_the_path():
    path = straight_tdm43()
    assert end_tdm43(path, 5.85, 0.05, 0.3) is None
    end = end_tdm43(path, 6.02, 0.05, 0.3)
    assert end.reached and end.at_path_end
    assert end.yaw_error == pytest.approx(0.3)


def test_a_yaw_tolerance_of_pi_or_more_ignores_the_heading():
    path = straight_tdm43()
    for yaw_tolerance in (math.pi, 4.0):
        for yaw in (0.3, -2.0, math.pi):
            end = end_tdm43(path, 5.85, 0.05, yaw, yaw_tolerance=yaw_tolerance)
            assert end.reached and not end.at_path_end, (yaw_tolerance, yaw)
    assert controller_param_problem(**{**CONTROLLER_PARAMS, 'yaw_goal_tolerance': math.pi}) is None


def test_passing_beside_the_goal_ends_it_aborted():
    # The Pi's failure: 0.27 m to the side with a 12 deg heading error drove on for 8 m
    path = straight_tdm43()
    assert end_tdm43(path, 5.95, 0.27, 0.21) is None
    end = end_tdm43(path, 6.0, 0.27, 0.21)
    assert not end.reached and end.at_path_end
    assert end.distance == pytest.approx(0.27)


def test_far_past_the_end_it_is_aborted():
    end = end_tdm43(straight_tdm43(), 8.0, 0.0, 0.0)
    assert not end.reached and end.at_path_end and end.distance == pytest.approx(2.0)


def test_short_of_the_end_and_outside_the_tolerances_it_goes_on():
    path = straight_tdm43()
    for x, y in ((3.0, 0.5), (5.7, 0.0), (5.99, 0.6)):
        assert end_tdm43(path, x, y, 0.0) is None, (x, y)


def test_a_route_that_doubles_back_is_not_over_where_it_starts():
    # 10 m east, a U-turn of radius 1 m, 5 m back west: the goal is (5, 2) heading west, and
    # the whole first half of the route lies past the line through it
    east = [(s, 0.0) for s in np.arange(0.0, 10.0, 0.1)]
    turn = [(10.0 + math.sin(a), 1.0 - math.cos(a)) for a in np.linspace(0.0, math.pi, 32)]
    west = [(x, 2.0) for x in np.linspace(9.9, 5.0, 50)]
    path = path_tdm43(east + turn + west)
    assert end_tdm43(path, 0.0, 0.0, 0.0) is None
    assert end_tdm43(path, 4.0, 0.0, 0.0, start_idx=35) is None
    late = len(path[0]) - 20
    assert end_tdm43(path, 5.5, 2.1, math.pi, start_idx=late) is None
    end = end_tdm43(path, 4.6, 2.1, math.pi, start_idx=late)
    assert not end.reached and end.at_path_end and end.distance == pytest.approx(math.hypot(0.4, 0.1))


def test_a_single_point_path_ends_only_inside_the_tolerances():
    path = path_tdm43([(1.0, 1.0)])
    assert end_tdm43(path, 3.0, 3.0, 0.0) is None
    assert end_tdm43(path, 1.1, 1.0, 0.0).reached


def test_every_end_says_how_far_from_the_goal():
    path = straight_tdm43()
    reached = end_tdm43(path, 5.85, 0.05, 0.05).message()
    at_end = end_tdm43(path, 6.02, 0.05, 0.3).message()
    missed = end_tdm43(path, 6.0, 0.27, 0.21).message()
    assert reached == 'Goal reached 0.16 m from the goal, heading off by 2.9 deg: stopping'
    assert at_end == ('Reached the end of the path 0.05 m from the goal, heading off by 17.2 deg: '
                      'stopping, goal reached')
    assert missed == ('Passed the end of the path 0.27 m from the goal, heading off by 12.0 deg: '
                      'stopping, goal aborted')
