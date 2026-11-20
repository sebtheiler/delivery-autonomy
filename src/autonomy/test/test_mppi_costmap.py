import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from autonomy.utils.costmap_speed import Costmap, end_taper_speed, path_progress, to_frame
from autonomy.utils.dynamics import INPUT_DIMENSION, dynamics_model
from autonomy.utils.mppi_step import (
    CRUISE_SPEED, LETHAL_PENALTY, W_OBSTACLE, WINDOW_CELLS, cell_cost, charged_steps, crop_window,
    free_window, mppi_step, obstacle_cost)
from autonomy.utils.reference_trajectory import get_local_reference

RES = 0.1
# odom's pose in map, turned so that a missed transform shows
MAP_TO_ODOM_TDM33 = (-94.63, 25.504, 0.3)


def to_map_tdm33(x, y):
    """An odom point in map."""
    px, py, yaw = MAP_TO_ODOM_TDM33
    return px + math.cos(yaw) * x - math.sin(yaw) * y, py + math.sin(yaw) * x + math.cos(yaw) * y


def costmap_tdm33(cells):
    """A 400 x 400 odom costmap at 0.1 m, outer corner at (-20, -20), like Nav2's."""
    return Costmap(np.asarray(cells, dtype=np.int8), RES, (-20.0, -20.0), 'odom')


def cell_centre_tdm33(row, col):
    return -20.0 + (col + 0.5) * RES, -20.0 + (row + 0.5) * RES


def box_costmap_tdm33(bx, by, size=0.6, half_width=None, end_wall=None, inscribed=0.32, radius=1.8,
                      scaling=2.0):
    """One square box in odom, optionally in a corridor along x (walls beyond |y| = half_width)
    that may end in a wall at x = end_wall, inflated like Nav2: 100 inside, 99 within the
    inscribed radius, then 252 exp(-scaling (d - inscribed)) out to the radius, translated to
    1-98 as in Nav2's OccupancyGrid."""
    x = -20.0 + (np.arange(400) + 0.5) * RES
    dx = np.maximum(np.abs(x[None, :] - bx) - size / 2, 0.0)
    dy = np.maximum(np.abs(x[:, None] - by) - size / 2, 0.0)
    d = np.hypot(dx, dy)
    if half_width is not None:
        d = np.minimum(d, np.maximum(half_width - np.abs(x[:, None]), 0.0))
    if end_wall is not None:
        d = np.minimum(d, np.maximum(end_wall - x[None, :], 0.0))
    raw = np.floor(252.0 * np.exp(-scaling * (d - inscribed)))
    cells = np.where(d <= radius, 1 + (97 * (raw - 1)) // 251, 0)
    cells = np.where(d <= inscribed, 99, cells)
    return costmap_tdm33(np.where(d == 0.0, 100, cells))


def state_tdm33(x, y, v=CRUISE_SPEED):
    """A state (map frame) with the robot at odom (x, y), driving at v."""
    return jnp.array([*to_map_tdm33(x, y), 0.0, v, 0.0, 0.0])


def cell_at_tdm33(window, x, y):
    """cell_cost of a robot at odom (x, y)."""
    return float(cell_cost(state_tdm33(x, y), window))


def obstacle_cost_tdm33(window, x, y, ref_odom, v=CRUISE_SPEED):
    """obstacle_cost of a robot at odom (x, y), driving at v, against a reference through the
    odom points ref_odom (heading along odom +x)."""
    ref = jnp.array([[*to_map_tdm33(rx, ry), MAP_TO_ODOM_TDM33[2]] for rx, ry in ref_odom])
    ref_costs = jax.vmap(cell_cost, in_axes=(0, None))(ref, window)
    return float(obstacle_cost(state_tdm33(x, y, v), window, ref, ref_costs))


def test_window_is_centred_on_the_robot_and_copies_the_cells():
    cells = (np.arange(400 * 400).reshape(400, 400) % 101).astype(np.int8)
    window = crop_window(costmap_tdm33(cells), to_map_tdm33(3.04, -1.97), MAP_TO_ODOM_TDM33)
    assert window.cells.shape == (WINDOW_CELLS, WINDOW_CELLS) and window.cells.dtype == np.float32
    # The robot is in cell row (-1.97 + 20) / 0.1 = 180, column (3.04 + 20) / 0.1 = 230
    np.testing.assert_array_equal(window.cells, cells[80:280, 130:330])
    assert window.origin == pytest.approx([-20.0 + 130 * RES, -20.0 + 80 * RES])
    assert window.frame_pose == pytest.approx(MAP_TO_ODOM_TDM33)


def test_unknown_and_off_map_cells_are_free():
    cells = np.full((400, 400), -1)
    cells[:10, :10] = 100
    # The robot in the costmap's corner cell: only a quarter of the window is on the costmap
    window = crop_window(costmap_tdm33(cells), to_map_tdm33(*cell_centre_tdm33(0, 0)), MAP_TO_ODOM_TDM33)
    assert (window.cells[100:110, 100:110] == 100).all()
    assert window.cells.sum() == 100 * 100


def test_cell_lookup():
    cells = np.zeros((400, 400))
    probes = {(210, 205): 50, (210, 206): 98, (215, 190): 99, (170, 202): 100, (230, 230): -1,
              (200, 325): 100}
    for cell, value in probes.items():
        cells[cell] = value
    window = crop_window(costmap_tdm33(cells), to_map_tdm33(0.0, 0.0), MAP_TO_ODOM_TDM33)
    cost = {cell: cell_at_tdm33(window, *cell_centre_tdm33(*cell)) for cell in probes}
    assert cost == {(210, 205): 50, (210, 206): 98, (215, 190): 99, (170, 202): 100,
                    (230, 230): 0,          # unknown
                    (200, 325): 0}          # lethal, but 12.5 m away: outside the window
    assert cell_at_tdm33(window, 0.55, 0.55) == 0.0
    # All of a cell maps to it: its four corners
    x, y = cell_centre_tdm33(170, 202)
    for ex, ey in ((0.01, 0.01), (0.09, 0.01), (0.01, 0.09), (0.09, 0.09)):
        assert cell_at_tdm33(window, x - 0.05 + ex, y - 0.05 + ey) == 100


def test_cost_is_the_change_from_the_nearest_reference_point_plus_the_penalty():
    cells = np.zeros((400, 400))
    cells[200, 210:213] = (40, 60, 99)      # odom (1.05, 0.05), (1.15, 0.05), (1.25, 0.05)
    window = crop_window(costmap_tdm33(cells), to_map_tdm33(0.0, 0.0), MAP_TO_ODOM_TDM33)
    ref = [(0.05, 0.05), (1.05, 0.05), (2.05, 0.05)]    # through the 40 cell
    assert obstacle_cost_tdm33(window, 1.05, 0.05, ref) == pytest.approx(0.0)  # on the path: nothing
    assert obstacle_cost_tdm33(window, 1.15, 0.05, ref) == pytest.approx(W_OBSTACLE * (0.6**2 - 0.4**2))
    assert obstacle_cost_tdm33(window, 1.25, 0.05, ref) == pytest.approx(
        W_OBSTACLE * (0.99**2 - 0.4**2) + LETHAL_PENALTY)
    assert obstacle_cost_tdm33(window, 1.05, 0.35, ref) == pytest.approx(-W_OBSTACLE * 0.4**2)  # off it
    assert obstacle_cost_tdm33(window, 0.05, 0.35, ref) == 0.0      # free, and so is the path there
    # Per metre: half the speed, half the cost per step. Standing still, only the penalty is left.
    assert obstacle_cost_tdm33(window, 1.15, 0.05, ref, v=CRUISE_SPEED / 2) == pytest.approx(
        W_OBSTACLE * (0.6**2 - 0.4**2) / 2)
    assert obstacle_cost_tdm33(window, 1.25, 0.05, ref, v=0.0) == pytest.approx(LETHAL_PENALTY)
    # Behind the first reference point (where the robot is now) nothing is compared
    assert obstacle_cost_tdm33(window, 1.15, 0.05, [(1.6, 0.05), (2.05, 0.05)]) == 0.0


def test_free_window_costs_nothing():
    states = jax.random.uniform(jax.random.PRNGKey(3), (2000, 6), minval=-300.0, maxval=300.0)
    ref = jax.random.uniform(jax.random.PRNGKey(4), (50, 3), minval=-300.0, maxval=300.0)
    costs = jax.vmap(obstacle_cost, in_axes=(0, None, None, None))(
        states, free_window(), ref, jnp.zeros(50))
    assert float(jnp.abs(costs).max()) == 0.0


def drive_tdm33(costmap, ticks=60, path_end=20.0, cut=True, v_cap=1.5, bend=None):
    """Follow a path along odom +x from (0, 0) to (path_end, 0) for up to `ticks` 10 Hz control
    ticks: MPPI, the end taper and the termination check as the controller runs them (default
    parameters), the dynamics model as the robot. The path is straight, or with bend=(x, dy) it
    swings out by dy round x and back over 5 m, like local_planner's paths round a box.
    cut=False charges the obstacle cost over the whole horizon; v_cap caps the target speed like
    the obstacle speed limit. Returns the track in odom and the speeds sent."""
    x = np.arange(0.0, path_end + 1e-9, 0.1)
    y = np.zeros_like(x)
    if bend is not None:
        near = np.abs(x - bend[0]) < 2.5
        y[near] = bend[1] * 0.5 * (1.0 + np.cos(np.pi * (x[near] - bend[0]) / 2.5))
    path_xy = np.array([to_map_tdm33(px, py) for px, py in zip(x, y)])
    step = np.diff(path_xy, axis=0)
    path_theta = np.unwrap(np.append(np.arctan2(step[:, 1], step[:, 0]), np.arctan2(*step[-1, ::-1])))
    path_s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(step, axis=1))))
    state = np.array([*path_xy[0], MAP_TO_ODOM_TDM33[2], 1.5, 0.0, 0.0])
    U = jnp.zeros((50, INPUT_DIMENSION)).at[:, 0].set(1.5)
    key, progress, track, speeds = jax.random.PRNGKey(0), 0, [], []
    for _ in range(ticks):
        if np.hypot(*(state[:2] - path_xy[-1])) < 0.25:
            break
        remaining = path_s[-1] - path_progress(state, path_xy, path_s, progress)
        v_target = min(v_cap, end_taper_speed(remaining, 0.8, 0.15))
        lower, upper = jnp.array([-0.5, -0.6]), jnp.array([v_target, 0.6])
        ref, progress = get_local_reference(state, path_xy, path_theta, path_s, 50, 0.1,
                                            v_target=v_target, progress_idx=progress)
        window = free_window() if costmap is None else crop_window(costmap, state, MAP_TO_ODOM_TDM33)
        steps = charged_steps(remaining, v_target, 50) if cut else np.int32(51)
        key, sub = jax.random.split(key)
        U = jnp.clip(mppi_step(sub, state, jnp.clip(U, lower, upper), jnp.array(ref), v_target,
                               lower, upper, window, steps), lower, upper)
        state = np.asarray(dynamics_model(state, U[0], dt=0.1), dtype=float)
        speeds.append(float(U[0, 0]))
        U = jnp.roll(U, -1, axis=0).at[-1].set(U[-1])
        track.append(to_frame(state[None, :2], MAP_TO_ODOM_TDM33)[0])
    return np.array(track), np.array(speeds)


def box_gap_tdm33(track, bx, by, size=0.6):
    """Closest approach of the robot's centre to the box's edge."""
    gap = np.maximum(np.abs(track - (bx, by)) - size / 2, 0.0)
    return float(np.hypot(gap[:, 0], gap[:, 1]).min())


def test_charged_steps():
    assert charged_steps(12.0, 1.5, 50) == 51                 # end out of reach: every state
    assert charged_steps(2.95, 1.5, 50) == 20                 # a 1.5 m/s rollout is there at step 20
    assert charged_steps(0.31, 0.15, 50) == 21
    assert charged_steps(0.0, 0.15, 50) == 0
    assert charged_steps(-0.2, 0.15, 50) == 0                 # past the end


def test_mppi_keeps_to_the_middle_of_the_gap_round_a_box_on_the_path():
    # The worst case: the path runs straight through a 0.6 m box in the middle of a 3 m corridor.
    # (local_planner's paths go round what the costmap shows.)
    blind, _ = drive_tdm33(None)
    assert box_gap_tdm33(blind, 4.0, 0.0) == 0.0             # straight through it
    track, _ = drive_tdm33(box_costmap_tdm33(4.0, 0.0, half_width=1.5))
    side = 0.315                                              # half the footprint's width
    assert box_gap_tdm33(track, 4.0, 0.0) - side > 0.15       # clear of the box (0.24 m here)
    assert 1.5 - np.abs(track[:, 1]).max() - side > 0.15      # and of the wall (0.29 m)
    assert track[-1, 0] > 7.0                                 # passed, not stopped
    assert abs(track[-1, 1]) < 0.3                            # and back on the path


def test_slowed_down_beside_a_box_mppi_still_drives_on():
    # The obstacle speed limit slows the robot to a crawl beside a box, on a path that goes round
    # it through the middle of the gap. The cost of the cells round the box must not make
    # standing still, or backing away, cheaper than driving past (it did when it was charged per
    # step and not relative to the path).
    corridor = box_costmap_tdm33(2.5, 0.0, half_width=1.5)
    track, speeds = drive_tdm33(corridor, ticks=120, v_cap=0.4, bend=(2.5, -0.9))
    assert speeds.min() >= 0.0                                # never reversed
    assert track[-1, 0] > 4.0                                 # got past the box
    assert box_gap_tdm33(track, 2.5, 0.0) - 0.315 > 0.2


def test_a_wall_past_the_end_of_the_path_does_not_slow_the_approach():
    # A dead end like the goal node's in the sim: the path ends 1.4 m before the end wall
    dead_end = box_costmap_tdm33(100.0, 100.0, half_width=1.5, end_wall=7.4)
    free_track, free_speeds = drive_tdm33(None, ticks=100, path_end=6.0)
    track, speeds = drive_tdm33(dead_end, ticks=100, path_end=6.0)
    assert len(track) < 100                                   # arrived
    assert len(track) <= len(free_track) + 3                  # no more than 0.3 s later
    # Charging the whole horizon would have crawled
    _, crawl = drive_tdm33(dead_end, ticks=100, path_end=6.0, cut=False)
    assert len(crawl) > len(track) + 10
