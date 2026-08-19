import jax
import jax.numpy as jnp

STATE_DIMENSION = 6
INPUT_DIMENSION = 2
WHEELBASE = 0.615  # Larger than the chassis to absorb tire slip
MAX_STEER = 0.6
DRIVE_TAU = 0.01
STEER_TAU = 0.545
MAX_ACCEL = 5.0

@jax.jit
def dynamics_model(state: jnp.ndarray, inputs: jnp.ndarray, dt: float) -> jnp.ndarray:
    x, y, theta, v, omega, b_omega = state
    v_cmd, delta_cmd = inputs

    # Closed form acceleration
    v_gap = v_cmd - v
    accel = jnp.clip(v_gap / DRIVE_TAU, -MAX_ACCEL, MAX_ACCEL)

    # Time spent at full acceleration before the
    # remaining gap is small enough
    t_clipped = jnp.clip(
        (jnp.abs(v_gap) - MAX_ACCEL * DRIVE_TAU) / MAX_ACCEL, 0.0, dt)
    next_v = (v + accel * t_clipped
              + (v_gap - accel * t_clipped)
              * (1.0 - jnp.exp(-(dt - t_clipped) / DRIVE_TAU)))

    # Recover the steering angle implied by the current yaw rate, holding the
    # sign through the zero crossing to avoid a singularity
    safe_v = jnp.where(jnp.abs(v) < 1e-5, jnp.sign(v + 1e-10) * 1e-5, v)
    current_delta = jnp.arctan((omega * WHEELBASE) / safe_v)

    delta_cmd = jnp.clip(delta_cmd, -MAX_STEER, MAX_STEER)
    # closed form
    next_delta = delta_cmd + (current_delta - delta_cmd) * jnp.exp(-dt / STEER_TAU)

    next_omega = (next_v / WHEELBASE) * jnp.tan(next_delta)

    theta_mid = theta + 0.5 * next_omega * dt
    next_x = x + next_v * jnp.cos(theta_mid) * dt
    next_y = y + next_v * jnp.sin(theta_mid) * dt
    
    next_theta = theta + next_omega * dt
    next_theta = (next_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.array([next_x, next_y, next_theta, next_v, next_omega, b_omega])
