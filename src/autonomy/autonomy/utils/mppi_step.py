import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from autonomy.utils.dynamics import SIM_VEHICLE, dynamics_model
from autonomy.algorithms.mppi import mppi
from autonomy.utils.costmap_speed import INSCRIBED, to_frame

# TODO: make a parameter if you can
DT = 0.1

# Every rollout, one step: the same vehicle for all of them
batch_dynamics = jax.vmap(dynamics_model, in_axes=(0, 0, None, None))

sigma = jnp.array([0.35, 0.35])

def ell(state, action, t, ref_traj_window, v_target, vehicle):
    x, y, theta, v, omega, b_omega = state
    v_cmd, raw_delta = action
    delta_cmd = jnp.clip(raw_delta, -vehicle.max_steer_rad, vehicle.max_steer_rad)
    
    target_state = ref_traj_window[t]
    x_ref, y_ref, theta_ref = target_state[0], target_state[1], target_state[2]
    
    dx = x - x_ref
    dy = y - y_ref
    
    cte = -dx * jnp.sin(theta_ref) + dy * jnp.cos(theta_ref)

    theta_error = theta - theta_ref
    theta_error = (theta_error + jnp.pi) % (2 * jnp.pi) - jnp.pi
    v_error = v - v_target

    # Pseudo-Huber, so a large excursion does not dominate every other term
    cte_cost = jnp.sqrt(cte**2 + 0.05) - jnp.sqrt(0.05)

    # Penalising the change in steering angle rather than its magnitude lets the
    # robot hold a steady curve for free while still damping rapid reversals
    safe_v = jnp.where(jnp.abs(v) < 1e-5, jnp.sign(v + 1e-10) * 1e-5, v)
    current_delta = jnp.arctan((omega * vehicle.wheelbase) / safe_v)
    steer_effort_cost = (delta_cmd - current_delta)**2

    w_cte = 90.0
    w_theta = 30.0
    w_steer_effort = 25.0
    w_omega = 5.0
    w_v = 20.0

    return (w_cte * cte_cost +
            w_theta * (theta_error**2) +
            w_steer_effort * steer_effort_cost +
            w_omega * (omega**2) +
            w_v * (v_error**2))

def phi(state, ref_traj_window):
    x, y, theta, v, omega, b_omega = state

    target_state = ref_traj_window[-1]
    x_ref, y_ref, theta_ref = target_state[0], target_state[1], target_state[2]

    dx = x - x_ref
    dy = y - y_ref

    cte = -dx * jnp.sin(theta_ref) + dy * jnp.cos(theta_ref)

    theta_error = theta - theta_ref
    theta_error = (theta_error + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # Only lateral and heading error are penalised. Charging for distance along
    # the path would make slowing down for a sharp corner prohibitively costly.
    w_cte_term = 200.0
    w_theta_term = 50.0

    return w_cte_term * (cte**2) + w_theta_term * (theta_error**2)

# Cells per side of the costmap window MPPI checks rollouts against: 20 m at 0.1 m, centred on the
# robot. A fixed size, so jax compiles mppi_step only once.
WINDOW_CELLS = 200


class CostWindow(NamedTuple):
    cells: np.ndarray        # (WINDOW_CELLS, WINDOW_CELLS) float32 costmap values 0-100; row = y
    origin: np.ndarray       # (2,) float32: outer corner of cell [0, 0], in the costmap's frame
    resolution: np.ndarray   # () float32: m per cell
    frame_pose: np.ndarray   # (3,) float32: the costmap frame's pose in map, (x, y, yaw)


def free_window():
    """A window with nothing in it: the obstacle cost is then exactly zero everywhere."""
    return CostWindow(np.zeros((WINDOW_CELLS, WINDOW_CELLS), dtype=np.float32),
                      np.zeros(2, dtype=np.float32), np.float32(0.1), np.zeros(3, dtype=np.float32))


def crop_window(costmap, centre, frame_pose):
    """The WINDOW_CELLS-square part of `costmap` (a costmap_speed.Costmap) round `centre` (map
    frame). frame_pose is the costmap frame's pose in map, (x, y, yaw). Unknown cells (-1) and
    cells off the costmap come out free (0)."""
    res = costmap.resolution
    ox, oy = costmap.origin
    (cx, cy), = to_frame(np.reshape(np.asarray(centre, dtype=float)[:2], (1, 2)), frame_pose)
    c0 = math.floor((cx - ox) / res) - WINDOW_CELLS // 2
    r0 = math.floor((cy - oy) / res) - WINDOW_CELLS // 2
    height, width = costmap.cells.shape
    cells = np.zeros((WINDOW_CELLS, WINDOW_CELLS), dtype=np.float32)
    rs, re = max(r0, 0), min(r0 + WINDOW_CELLS, height)
    cs, ce = max(c0, 0), min(c0 + WINDOW_CELLS, width)
    if rs < re and cs < ce:
        cells[rs - r0:re - r0, cs - c0:ce - c0] = np.maximum(costmap.cells[rs:re, cs:ce], 0)
    return CostWindow(cells, np.array([ox + c0 * res, oy + r0 * res], dtype=np.float32),
                      np.float32(res), np.asarray(frame_pose, dtype=np.float32))

W_OBSTACLE = 300.0        # times the change in (cost / 100)^2 from the path, per step at CRUISE_SPEED
CRUISE_SPEED = 1.5        # m/s; slower states pay in proportion (per metre)
LETHAL_PENALTY = 1000.0   # per step in an inscribed (99) or lethal (100) cell

def cell_cost(state, window):
    """The costmap value (0-100) under the robot's centre, state in map; 0 off the window."""
    fx, fy, fyaw = window.frame_pose[0], window.frame_pose[1], window.frame_pose[2]
    c, s = jnp.cos(fyaw), jnp.sin(fyaw)
    dx, dy = state[0] - fx, state[1] - fy
    # Into the costmap's frame, then to a cell
    col = jnp.floor((c * dx + s * dy - window.origin[0]) / window.resolution).astype(jnp.int32)
    row = jnp.floor((c * dy - s * dx - window.origin[1]) / window.resolution).astype(jnp.int32)
    cells = jnp.asarray(window.cells)
    rows, cols = cells.shape
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return jnp.where(inside, cells[jnp.clip(row, 0, rows - 1), jnp.clip(col, 0, cols - 1)], 0.0)


def obstacle_cost(state, window, ref_traj, ref_costs):
    """What one state (map frame) pays for the costmap cell under the robot's centre: the change
    in W_OBSTACLE (cost / 100)^2 from the nearest reference point (ref_costs holds their
    cell_cost; nothing behind the first one), plus LETHAL_PENALTY in an inscribed or lethal cell."""
    cost = cell_cost(state, window)
    nearest = jnp.argmin((ref_traj[:, 0] - state[0])**2 + (ref_traj[:, 1] - state[1])**2)
    x0, y0, theta0 = ref_traj[0, 0], ref_traj[0, 1], ref_traj[0, 2]
    along = (state[0] - x0) * jnp.cos(theta0) + (state[1] - y0) * jnp.sin(theta0)
    behind = (nearest == 0) & (along < 0.0)
    change = jnp.where(behind, 0.0, (cost / 100.0)**2 - (ref_costs[nearest] / 100.0)**2)
    # Per metre driven, not per step: lingering beside the path neither pays nor earns
    per_metre = jnp.abs(state[3]) / CRUISE_SPEED
    return W_OBSTACLE * per_metre * change + jnp.where(cost >= INSCRIBED, LETHAL_PENALTY, 0.0)


def charged_steps(remaining, v_target, horizon, dt=DT):
    """How many states of the horizon the obstacle cost counts for: those before a rollout at
    v_target could reach the end of the path, `remaining` m (arc length) ahead. horizon + 1 (every
    state, the final one too) when the end is out of reach."""
    if v_target <= 0.0:
        return np.int32(horizon + 1)
    return np.int32(min(max(math.ceil(remaining / (v_target * dt)), 0), horizon + 1))

@jax.jit
def mppi_step(key, x0, U_nominal, ref_traj, v_target, lower_bounds, upper_bounds, window,
              obstacle_steps, vehicle=SIM_VEHICLE):
    """
    x0: (STATE_DIMENSION,)
    U_nominal: (T, INPUT_DIMENSION)
    ref_traj: (T, 3) containing [x, y, theta]
    lower_bounds, upper_bounds: (INPUT_DIMENSION,) [v, delta] the samples are clipped to
    window: CostWindow, the costmap round the robot (free_window() for none)
    obstacle_steps: int32, the obstacle cost counts for the states before this step (T + 1: all
        of them, the final state too)
    vehicle: Vehicle, the car the rollouts drive
    """
    T = U_nominal.shape[0]
    ref_costs = jax.vmap(cell_cost, in_axes=(0, None))(ref_traj, window)

    def obstacles(state):
        return obstacle_cost(state, window, ref_traj, ref_costs)

    # Create closures that inject ref_traj into the unbatched functions
    def ell_closure(state, action, t):
        return (ell(state, action, t, ref_traj, v_target, vehicle)
                + jnp.where(t < obstacle_steps, obstacles(state), 0.0))

    def phi_closure(state):
        return phi(state, ref_traj) + jnp.where(T < obstacle_steps, obstacles(state), 0.0)

    # Vmap the closures over the batch dimension (axis 0 for state and action).
    # t is a scalar, so we use None for its axis.
    batch_ell = jax.vmap(ell_closure, in_axes=(0, 0, None))
    batch_phi = jax.vmap(phi_closure, in_axes=(0,))

    return mppi(
        key=key, 
        f=lambda states, actions: batch_dynamics(states, actions, DT, vehicle),
        x0=x0, 
        U=U_nominal, 
        phi=batch_phi, 
        ell=batch_ell,
        sigma=sigma,
        K=3000,
        lam=30.0,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
