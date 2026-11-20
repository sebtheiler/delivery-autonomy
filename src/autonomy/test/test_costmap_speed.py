import math

import numpy as np
import pytest

from autonomy.utils.costmap_speed import (
    GOVERNOR_PARAMS, Costmap, governor_param_problem, obstacle_clearance, path_ahead,
    remembered_costmap, speed_for_clearance, speed_limit, to_frame)

RES = 0.1
LIMITS = dict(max_speed=1.5, slow_distance=2.0, stop_distance=0.35, creep_speed=0.15,
              costmap_timeout=1.0, crawl_speed=0.3)
# delivery-robo's hardware.yaml (TDM-34)
ROBOT = {**LIMITS, 'max_speed': 1.0, 'stop_distance': 0.2}


def corridor_tdm32(half_width=1.5, inscribed=0.32, length=20.0):
    """A straight corridor along +x through y = 0, in odom with its corner at (-5, -5): walls
    lethal, the inscribed band inside them 99, free in between."""
    h, w = int(10 / RES), int(length / RES)
    y = -5 + (np.arange(h) + 0.5) * RES
    cells = np.zeros((h, w), dtype=np.int8)
    cells[np.abs(y) >= half_width - inscribed, :] = 99
    cells[np.abs(y) >= half_width, :] = 100
    return Costmap(cells, RES, (-5.0, -5.0), 'odom')


def add_box_tdm32(costmap, cx, cy, size=0.6, inscribed=0.32):
    """Mark a square box lethal and the band round it inscribed (as a square, close enough)."""
    ox, oy = costmap.origin
    h, w = costmap.cells.shape
    x = ox + (np.arange(w) + 0.5) * RES
    y = oy + (np.arange(h) + 0.5) * RES
    dx, dy = np.abs(x[None, :] - cx), np.abs(y[:, None] - cy)
    cells = costmap.cells.copy()
    cells[(dx <= size / 2 + inscribed) & (dy <= size / 2 + inscribed)] = 99
    cells[(dx <= size / 2) & (dy <= size / 2)] = 100
    return costmap._replace(cells=cells)


def straight_path_tdm32(x0=0.0, x1=12.0):
    xy = np.column_stack((np.arange(x0, x1 + 1e-9, 0.1), np.zeros(int(round((x1 - x0) / 0.1)) + 1)))
    return xy, np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))


def ahead_tdm32(x, path=None, y=0.0):
    """The robot at (x, y) and the next 3 m of a path sampled every 0.1 m from x = 0."""
    xy, s = path if path is not None else straight_path_tdm32()
    return path_ahead((x, y), xy, s, max(int(x / 0.1) - 5, 0), 3.0)


def test_ramp():
    assert speed_for_clearance(math.inf, 1.5, 2.0, 0.35, 0.15) == 1.5
    assert speed_for_clearance(2.0, 1.5, 2.0, 0.35, 0.15) == 1.5
    assert speed_for_clearance(0.35, 1.5, 2.0, 0.35, 0.15) == 0.0
    assert speed_for_clearance(0.1, 1.5, 2.0, 0.35, 0.15) == 0.0
    assert speed_for_clearance(1.175, 1.5, 2.0, 0.35, 0.15) == pytest.approx(0.75)
    # the ramp gives 0.15 m/s at 0.515 m; from there to stop_distance it is the floor
    assert speed_for_clearance(0.6, 1.5, 2.0, 0.35, 0.15) == pytest.approx(1.5 * 0.25 / 1.65)
    assert speed_for_clearance(0.5, 1.5, 2.0, 0.35, 0.15) == 0.15
    assert speed_for_clearance(0.36, 1.5, 2.0, 0.35, 0.15) == 0.15
    # never above max_speed, even with a floor above it
    assert speed_for_clearance(0.36, 0.1, 2.0, 0.35, 0.15) == 0.1


def test_robot_values_stop_at_stop_distance_not_before():
    """TDM-40: with hardware.yaml's values the old 0.05 m/s cut-off stopped the robot at 0.29 m."""
    ramp = lambda c: speed_for_clearance(c, 1.0, 2.0, 0.2, 0.15)  # noqa: E731
    assert ramp(0.2) == 0.0
    assert ramp(0.19) == 0.0
    assert ramp(0.2 + 1e-9) == 0.15
    for clearance in (0.21, 0.25, 0.29, 0.4, 0.47):
        assert ramp(clearance) == 0.15, clearance
    assert ramp(1.1) == pytest.approx(0.5)
    # the same through speed_limit, robot in front of a box on the path (band edge at x 5.40)
    costmap = add_box_tdm32(corridor_tdm32(), 6.0, 0.0)
    for x, clearance in ((5.11, 0.29), (5.19, 0.21)):
        limit, c, status = speed_limit(costmap, 0.1, ahead_tdm32(x), (0.0, 0.0, 0.0), **ROBOT)
        assert c == pytest.approx(clearance, abs=1e-6)
        assert (limit, status) == (0.15, 'slow'), x
    limit, c, status = speed_limit(costmap, 0.1, ahead_tdm32(5.21), (0.0, 0.0, 0.0), **ROBOT)
    assert c == pytest.approx(0.19, abs=1e-6)
    assert (limit, status) == (0.0, 'stop')


def test_path_ahead_starts_at_the_robot_and_runs_lookahead_along_the_path():
    pts = ahead_tdm32(2.04, y=0.3)
    assert pts[0] == pytest.approx([2.04, 0.3])
    assert pts[1] == pytest.approx([2.04, 0.0])     # its projection, between two path points
    assert pts[-1] == pytest.approx([5.04, 0.0])
    assert np.diff(pts[1:, 0]).max() <= 0.1 + 1e-9


def test_path_ahead_stops_at_the_end_of_the_path():
    assert ahead_tdm32(11.0)[-1] == pytest.approx([12.0, 0.0])


def test_corridor_walls_do_not_slow_the_robot():
    assert obstacle_clearance(corridor_tdm32(), ahead_tdm32(2.0), 0.35) == math.inf
    limit, _, status = speed_limit(corridor_tdm32(), 0.1, ahead_tdm32(2.0), (0.0, 0.0, 0.0), **LIMITS)
    assert (limit, status) == (1.5, 'clear')


def test_box_on_the_path_slows_then_stops_the_robot():
    costmap = add_box_tdm32(corridor_tdm32(), 6.0, 0.0)
    # the inscribed band's first cells are centred on x 5.45, so their near edge is at 5.40
    far = speed_limit(costmap, 0.1, ahead_tdm32(2.0), (0.0, 0.0, 0.0), **LIMITS)
    assert far[0] == 1.5 and far[2] == 'clear'      # path ends at 5.0, 0.40 m short of the band
    mid = speed_limit(costmap, 0.1, ahead_tdm32(4.4), (0.0, 0.0, 0.0), **LIMITS)
    assert mid[1] == pytest.approx(1.0, abs=1e-6)
    assert mid[0] == pytest.approx(1.5 * (1.0 - 0.35) / 1.65) and mid[2] == 'slow'
    near = speed_limit(costmap, 0.1, ahead_tdm32(5.1), (0.0, 0.0, 0.0), **LIMITS)
    assert near[1] == pytest.approx(0.3, abs=1e-6)
    assert near[0] == 0.0 and near[2] == 'stop'
    # at 0.38 m the ramp would allow 0.027 m/s; the floor keeps it moving to stop_distance
    creep = speed_limit(costmap, 0.1, ahead_tdm32(5.02), (0.0, 0.0, 0.0), **LIMITS)
    assert creep[1] == pytest.approx(0.38, abs=1e-6)
    assert creep[0] == 0.15 and creep[2] == 'slow'


def test_box_in_the_middle_of_a_3m_corridor_blocks_the_gap():
    """A centred 0.6 m box leaves 1.2 m gaps; a path through one keeps < 0.35 m on both sides."""
    costmap = add_box_tdm32(corridor_tdm32(), 6.0, 0.0)
    # the path swerves round the box to y 0.9, the middle of the gap
    x = np.arange(0.0, 12.0 + 1e-9, 0.1)
    xy = np.column_stack((x, 0.9 * np.exp(-((x - 6.0) / 1.5) ** 2)))
    path = (xy, np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))))
    on_path = lambda x0: ahead_tdm32(x0, path, np.interp(x0, xy[:, 0], xy[:, 1]))  # noqa: E731
    assert 0.35 < obstacle_clearance(costmap, on_path(3.8), 0.35) < 2.0   # slowing already
    assert obstacle_clearance(costmap, on_path(5.2), 0.35) <= 0.35        # stopped at the gap


def test_box_beside_the_path_with_room_to_spare_is_ignored():
    costmap = add_box_tdm32(corridor_tdm32(), 6.0, 1.1)   # box from y 0.8, its band from 0.5
    assert obstacle_clearance(costmap, ahead_tdm32(4.0), 0.35) == math.inf


def test_obstacle_the_path_leads_away_from_does_not_lock_the_robot():
    costmap = add_box_tdm32(corridor_tdm32(), 4.6, 0.95)  # band x 4.0-5.2, down to y 0.30
    # just past the band, still within 0.35 m of its last cells: free to drive on
    assert obstacle_clearance(costmap, ahead_tdm32(5.3), 0.35) == math.inf
    # level with its start, the same box stops the robot
    assert obstacle_clearance(costmap, ahead_tdm32(4.0), 0.35) <= 0.35


def path_tdm42(xy):
    xy = np.asarray(xy, dtype=float)
    return xy, np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))


def beside_box_tdm42():
    """TDM-40's creep: the robot stopped beside a box, 0.17 m from its inscribed band (box at
    (6.0, 0.7), band from x 5.40 and y 0.10, robot at (5.8, -0.07) heading +x)."""
    return add_box_tdm32(corridor_tdm32(), 6.0, 0.7), (5.8, -0.07)


def test_parallel_path_inside_stop_distance_stops_the_robot():
    """TDM-40: a path along the box brought the robot no closer to the cells beside it, and the
    cells further along were beyond stop_distance, so the robot crept on at 0.15 m/s."""
    costmap, (x, y) = beside_box_tdm42()
    path = path_tdm42([(x0, y) for x0 in np.arange(5.0, 12.0, 0.1)])
    pts = path_ahead((x, y), *path, 0, 3.0)
    assert obstacle_clearance(costmap, pts, 0.2) == pytest.approx(0.17, abs=1e-6)
    limit, clearance, status = speed_limit(costmap, 0.1, pts, (0.0, 0.0, 0.0), **ROBOT)
    assert (limit, status) == (0.0, 'stop')
    # the same path with the box gone: full speed again
    limit, clearance, status = speed_limit(corridor_tdm32(), 0.1, pts, (0.0, 0.0, 0.0), **ROBOT)
    assert (limit, clearance, status) == (1.0, math.inf, 'clear')


def test_path_leading_away_inside_stop_distance_creeps_out():
    costmap, (x, y) = beside_box_tdm42()
    # a 1.0 m radius turn away from the box: 0.045 m further from it after 0.3 m
    theta = np.arange(0.0, 1.5, 0.05)
    path = path_tdm42(np.column_stack((x + np.sin(theta), y - (1.0 - np.cos(theta)))))
    pts = path_ahead((x, y), *path, 0, 3.0)
    assert obstacle_clearance(costmap, pts, 0.2) == math.inf
    limit, clearance, status = speed_limit(costmap, 0.1, pts, (0.0, 0.0, 0.0), **ROBOT)
    assert (limit, status) == (0.15, 'slow')
    # turning towards the box instead stops it
    path = path_tdm42(np.column_stack((x + np.sin(theta), y + (1.0 - np.cos(theta)))))
    pts = path_ahead((x, y), *path, 0, 3.0)
    assert speed_limit(costmap, 0.1, pts, (0.0, 0.0, 0.0), **ROBOT)[0] == 0.0


def test_remembered_costmap_keeps_the_obstacles_of_recent_updates():
    """TDM-42: in the sim a box corner's cells came and went between costmap updates, and every
    update without them let the robot creep a few centimetres closer."""
    seen = add_box_tdm32(corridor_tdm32(), 6.0, 0.7)
    # the next update has rolled 1.0 m on and 0.2 m up, and lost the box
    gone = Costmap(np.zeros((100, 200), dtype=np.int8), RES, (-4.0, -4.8), 'odom')
    merged = remembered_costmap([seen, gone])
    assert merged.origin == gone.origin and merged.cells.shape == gone.cells.shape
    row, col = round((0.7 + 4.8) / RES), round((6.0 + 4.0) / RES)
    assert merged.cells[row, col] == 100 and gone.cells[row, col] == 0
    # a grid in another frame is left out
    assert remembered_costmap([seen._replace(frame='map'), gone]).cells.max() == 0


def test_map_to_odom_is_applied():
    # costmap in odom; odom sits at (100, 50) in map, rotated 90 deg
    costmap = add_box_tdm32(corridor_tdm32(), 6.0, 0.0)
    pose = (100.0, 50.0, math.pi / 2)
    odom_pts = ahead_tdm32(4.4)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    map_pts = np.column_stack((pose[0] + c * odom_pts[:, 0] - s * odom_pts[:, 1],
                               pose[1] + s * odom_pts[:, 0] + c * odom_pts[:, 1]))
    assert to_frame(map_pts, pose) == pytest.approx(odom_pts)
    limit, clearance, _ = speed_limit(costmap, 0.1, map_pts, pose, **LIMITS)
    assert clearance == pytest.approx(1.0, abs=1e-6)


def test_missing_unplaced_and_stale_costmaps_crawl():
    pts = ahead_tdm32(2.0)
    assert speed_limit(None, None, pts, None, **LIMITS) == (0.3, None, 'missing')
    assert speed_limit(corridor_tdm32(), 0.1, pts, None, **LIMITS) == (0.3, None, 'unplaced')
    limit, _, status = speed_limit(corridor_tdm32(), 1.5, pts, (0.0, 0.0, 0.0), **LIMITS)
    assert (limit, status) == (0.3, 'stale')
    # the crawl never exceeds max_speed
    assert speed_limit(None, None, pts, None, **{**LIMITS, 'max_speed': 0.2})[0] == 0.2


def test_stale_costmap_still_stops_for_what_it_shows():
    costmap = add_box_tdm32(corridor_tdm32(), 6.0, 0.0)
    limit, _, status = speed_limit(costmap, 5.0, ahead_tdm32(5.1), (0.0, 0.0, 0.0), **LIMITS)
    assert (limit, status) == (0.0, 'stale')


def test_costmap_from_occupancy_grid_message():
    class Msg:   # the fields of a nav_msgs/OccupancyGrid that are read
        class header:
            frame_id = 'odom'

        class info:
            width, height, resolution = 3, 2, 0.1

            class origin:
                class position:
                    x, y = -1.0, 2.0
        data = [0, 0, 100, 99, -1, 0]
    costmap = Costmap.from_msg(Msg)
    assert costmap.cells.shape == (2, 3) and costmap.cells[0, 2] == 100 and costmap.cells[1, 0] == 99
    assert costmap.origin == (-1.0, 2.0) and costmap.frame == 'odom'


def test_param_checks():
    assert governor_param_problem(**GOVERNOR_PARAMS) is None
    for name, value in [('slow_distance', 0.35), ('stop_distance', -0.1), ('clearance_lookahead', 0.0),
                        ('costmap_timeout', 0.0), ('crawl_speed', -0.1), ('obstacle_creep_speed', 0.0),
                        ('obstacle_creep_speed', -0.1)]:
        assert governor_param_problem(**{**GOVERNOR_PARAMS, name: value}) is not None, (name, value)
    assert governor_param_problem(**{**GOVERNOR_PARAMS, 'crawl_speed': 0.0}) is None
