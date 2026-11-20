"""Closed-loop replay of how the controller ends a goal, with the Pi's settings (TDM-43)."""
import math
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from autonomy.utils.costmap_speed import end_taper_speed, path_progress
from autonomy.utils.dynamics import INPUT_DIMENSION, Vehicle, dynamics_model
from autonomy.utils.goal_end import goal_end
from autonomy.utils.mppi_step import charged_steps, free_window, mppi_step
from autonomy.utils.reference_trajectory import get_local_reference, process_path

pytestmark = pytest.mark.slow

# delivery-robo's hardware.yaml for the controller: forward only, at most 1.0 m/s
MAX_SPEED_TDM43, MIN_SPEED_TDM43 = 1.0, 0.0
XY_TOLERANCE_TDM43, YAW_TOLERANCE_TDM43 = 0.25, 0.15
GOAL_DECEL_TDM43, GOAL_CREEP_TDM43 = 0.8, 0.15
VEHICLE_TDM43 = Vehicle(wheelbase=0.36322, max_steer_rad=0.4363)
T_TDM43, DT_TDM43 = 50, 0.1
REPLAY_S_TDM43 = 60.0
# m from the goal: within this the robot counts as stopped at it
STOP_WITHIN_TDM43 = 0.6
# m: with a position error up to this the true position is checked too. Beyond it the robot
# stops where its estimate puts the goal, and only a better estimate brings it closer
TRUTH_CHECKED_TDM43 = 0.2

# name: (path, position error m, heading error deg, heading bias deg). The errors drift
CASES_TDM43 = {
    'straight, perfect estimate': ('straight', 0.0, 0.0, 0.0),
    'corner, perfect estimate': ('corner', 0.0, 0.0, 0.0),
    'straight, RTK fixed (3 cm, 1 deg)': ('straight', 0.03, 1.0, 0.0),
    'corner, RTK fixed (3 cm, 1 deg)': ('corner', 0.03, 1.0, 0.0),
    'straight, RTK float (20 cm, 3 deg)': ('straight', 0.20, 3.0, 0.0),
    'corner, RTK float (20 cm, 3 deg)': ('corner', 0.20, 3.0, 0.0),
    'straight, plain GPS (1 m, 3 deg)': ('straight', 1.0, 3.0, 0.0),
    'straight, IMU yaw bias 6 deg': ('straight', 0.0, 0.0, 6.0),
    'straight, IMU yaw bias 12 deg': ('straight', 0.0, 0.0, 12.0),
    'corner, IMU yaw bias 12 deg': ('corner', 0.0, 0.0, 12.0),
}


def path_msg_tdm43(points):
    return SimpleNamespace(poses=[SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)))
                                  for x, y in points])


def straight_tdm43(length=6.0, step=0.1):
    return [(s, 0.0) for s in np.arange(0.0, length + 1e-9, step)]


def corner_tdm43(straight_len=4.0, radius=1.5, step=0.1):
    points = [(s, 0.0) for s in np.arange(0.0, straight_len, step)]
    n = int(math.ceil(radius * math.pi / 2 / step))
    for k in range(n + 1):
        a = -math.pi / 2 + (math.pi / 2) * k / n
        points.append((straight_len + radius * math.cos(a), radius + radius * math.sin(a)))
    return points


class EstimateErrorTDM43:
    """Error of the pose the controller sees: a Gauss-Markov drift (like an EKF riding on GPS)
    plus a constant heading bias (like an IMU yaw offset)."""

    def __init__(self, rng, sigma_xy=0.0, sigma_yaw_deg=0.0, yaw_bias_deg=0.0, tau=3.0):
        self.rng, self.tau = rng, tau
        self.sigma = np.array([sigma_xy, sigma_xy, math.radians(sigma_yaw_deg)])
        self.bias = math.radians(yaw_bias_deg)
        self.e = self.rng.normal(0.0, 1.0, 3) * self.sigma

    def step(self):
        a = math.exp(-DT_TDM43 / self.tau)
        self.e = a * self.e + math.sqrt(1 - a * a) * self.rng.normal(0.0, 1.0, 3) * self.sigma
        return np.array([self.e[0], self.e[1], self.e[2] + self.bias, 0.0, 0.0, 0.0])


class ReplayTDM43(NamedTuple):
    end: object             # goal_end's answer, None if the goal never ended
    time: float             # s until the goal ended, or the whole replay
    end_estimate: float     # m from the goal by the estimate when it ended
    stop_distance: float    # m from the goal (true position) 1 s after the zero command
    stop_speed: float       # m/s then
    closest: float          # m, the closest the robot came to the goal
    driven_on: float        # m driven after first coming within 0.5 m of the goal


def replay_tdm43(points, error, end_check=goal_end, replay_s=REPLAY_S_TDM43):
    """Follow a path through `points` from the origin as control_loop_callback does (no
    costmap), the controller seeing the true state plus `error`, until `end_check` ends it."""
    path_xy, path_theta, path_s = process_path(path_msg_tdm43(points))
    goal = path_xy[-1]
    true = np.zeros(6)
    U_nominal = jnp.zeros((T_TDM43, INPUT_DIMENSION))
    key = jax.random.PRNGKey(42)
    progress_idx, last_u = 0, None
    closest, driven_on, near = math.inf, 0.0, False
    for tick in range(round(replay_s / DT_TDM43)):
        estimate = true + error.step()
        distance = float(np.hypot(*(goal - true[:2])))
        closest = min(closest, distance)
        near = near or distance <= 0.5
        end = end_check(estimate, path_xy, path_theta, path_s, progress_idx,
                        XY_TOLERANCE_TDM43, YAW_TOLERANCE_TDM43)
        if end is not None:
            # The zero command, held for a second
            for _ in range(10):
                true = np.asarray(dynamics_model(true, np.zeros(INPUT_DIMENSION), dt=DT_TDM43,
                                                 vehicle=VEHICLE_TDM43))
            return ReplayTDM43(end, tick * DT_TDM43, float(np.hypot(*(goal - estimate[:2]))),
                               float(np.hypot(*(goal - true[:2]))), abs(float(true[3])), closest,
                               driven_on)
        future = (dynamics_model(estimate, last_u, dt=DT_TDM43, vehicle=VEHICLE_TDM43)
                  if last_u is not None else estimate)
        remaining = path_s[-1] - path_progress(future, path_xy, path_s, progress_idx)
        v_target = float(min(MAX_SPEED_TDM43, end_taper_speed(remaining, GOAL_DECEL_TDM43, GOAL_CREEP_TDM43)))
        ref, progress_idx = get_local_reference(future, path_xy, path_theta, path_s, T_TDM43, DT_TDM43,
                                                v_target=v_target, progress_idx=progress_idx)
        key, step_key = jax.random.split(key)
        lower = jnp.array([MIN_SPEED_TDM43, -VEHICLE_TDM43.max_steer_rad])
        upper = jnp.array([v_target, VEHICLE_TDM43.max_steer_rad])
        U = mppi_step(key=step_key, x0=future, U_nominal=jnp.clip(U_nominal, lower, upper),
                      ref_traj=jnp.array(ref), v_target=v_target, lower_bounds=lower,
                      upper_bounds=upper, window=free_window(),
                      obstacle_steps=charged_steps(remaining, v_target, T_TDM43), vehicle=VEHICLE_TDM43)
        U = jnp.clip(U, lower, upper)
        last_u = np.asarray(U[0])
        U_nominal = jnp.roll(U, -1, axis=0).at[-1].set(U[-1])
        before = true
        true = np.asarray(dynamics_model(true, last_u, dt=DT_TDM43, vehicle=VEHICLE_TDM43))
        if near:
            driven_on += float(np.hypot(*(true[:2] - before[:2])))
    return ReplayTDM43(None, replay_s, math.nan, float(np.hypot(*(goal - true[:2]))),
                       abs(float(true[3])), closest, driven_on)


def replay_case_tdm43(name, end_check=goal_end):
    shape, sigma_xy, sigma_yaw_deg, yaw_bias_deg = CASES_TDM43[name]
    points = straight_tdm43() if shape == 'straight' else corner_tdm43()
    seed = list(CASES_TDM43).index(name)
    error = EstimateErrorTDM43(np.random.default_rng(seed), sigma_xy, sigma_yaw_deg, yaw_bias_deg)
    return replay_tdm43(points, error, end_check)


@pytest.mark.parametrize('name', list(CASES_TDM43))
def test_the_robot_stops_at_the_goal(name):
    result = replay_case_tdm43(name)
    assert result.end is not None, f'never stopped; closest {result.closest:.2f} m'
    assert result.stop_speed < 0.01
    assert result.end_estimate <= STOP_WITHIN_TDM43, result
    if CASES_TDM43[name][1] <= TRUTH_CHECKED_TDM43:
        assert result.stop_distance <= STOP_WITHIN_TDM43, result
    # The goal succeeds exactly when the estimate is within the tolerance
    assert result.end.reached == (result.end_estimate <= XY_TOLERANCE_TDM43), result
