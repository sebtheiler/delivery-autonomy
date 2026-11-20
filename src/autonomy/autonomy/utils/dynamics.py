import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

STATE_DIMENSION = 6
INPUT_DIMENSION = 2


class Vehicle(NamedTuple):
    wheelbase: float = 0.5        # m, rear axle to front axle. The sim's is <wheel_base> in the robot description
    max_steer_rad: float = 0.6    # rad, the steering limit either way
    steer_tau: float = 0.545      # s, time constant of the steering
    drive_tau: float = 0.01       # s, time constant of the speed
    max_accel: float = 5.0        # m/s^2, the drive's acceleration limit


SIM_VEHICLE = Vehicle()
VEHICLE_PARAMS = SIM_VEHICLE._asdict()


def vehicle_param_problem(wheelbase, max_steer_rad, steer_tau, drive_tau, max_accel):
    """Return why this vehicle cannot be modelled, or None when it can."""
    for name, value in (('wheelbase', wheelbase), ('steer_tau', steer_tau),
                        ('drive_tau', drive_tau), ('max_accel', max_accel)):
        if not 0.0 < value < math.inf:
            return f'{name} must be positive, got {value}'
    if not 0.0 < max_steer_rad < math.pi / 2:
        return f'max_steer_rad must be in (0, pi/2) rad, got {max_steer_rad}'
    return None


@jax.jit
def dynamics_model(state: jnp.ndarray, inputs: jnp.ndarray, dt: float,
                   vehicle: Vehicle = SIM_VEHICLE) -> jnp.ndarray:
    x, y, theta, v, omega, b_omega = state
    v_cmd, delta_cmd = inputs
    wheelbase, max_steer, steer_tau, drive_tau, max_accel = vehicle
    # One division each per call, then products: MPPI runs this for every state of every rollout,
    # and dividing each of them by a parameter made its step 12 % slower than by a constant
    inv_drive_tau, inv_accel, inv_wheelbase = 1.0 / drive_tau, 1.0 / max_accel, 1.0 / wheelbase

    # Closed form acceleration
    v_gap = v_cmd - v
    accel = jnp.clip(v_gap * inv_drive_tau, -max_accel, max_accel)

    # Time spent at full acceleration before the
    # remaining gap is small enough
    t_clipped = jnp.clip(
        (jnp.abs(v_gap) - max_accel * drive_tau) * inv_accel, 0.0, dt)
    next_v = (v + accel * t_clipped
              + (v_gap - accel * t_clipped)
              * (1.0 - jnp.exp((t_clipped - dt) * inv_drive_tau)))

    # Recover the steering angle implied by the current yaw rate, holding the
    # sign through the zero crossing to avoid a singularity
    safe_v = jnp.where(jnp.abs(v) < 1e-5, jnp.sign(v + 1e-10) * 1e-5, v)
    current_delta = jnp.arctan((omega * wheelbase) / safe_v)

    delta_cmd = jnp.clip(delta_cmd, -max_steer, max_steer)
    # closed form
    next_delta = delta_cmd + (current_delta - delta_cmd) * jnp.exp(-dt / steer_tau)

    next_omega = (next_v * inv_wheelbase) * jnp.tan(next_delta)

    theta_mid = theta + 0.5 * next_omega * dt
    next_x = x + next_v * jnp.cos(theta_mid) * dt
    next_y = y + next_v * jnp.sin(theta_mid) * dt
    
    next_theta = theta + next_omega * dt
    next_theta = (next_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.array([next_x, next_y, next_theta, next_v, next_omega, b_omega])
