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

    desired_accel = (v_cmd - v) / DRIVE_TAU
    actual_accel = jnp.clip(desired_accel, -MAX_ACCEL, MAX_ACCEL)
    next_v = v + actual_accel * dt

    # Recover the steering angle implied by the current yaw rate, holding the
    # sign through the zero crossing to avoid a singularity
    safe_v = jnp.where(jnp.abs(v) < 1e-5, jnp.sign(v + 1e-10) * 1e-5, v)
    current_delta = jnp.arctan((omega * WHEELBASE) / safe_v)

    # The lag belongs to the steering rack, not the chassis yaw
    delta_cmd = jnp.clip(delta_cmd, -MAX_STEER, MAX_STEER)
    next_delta = current_delta + ((delta_cmd - current_delta) / STEER_TAU) * dt

    next_omega = (next_v / WHEELBASE) * jnp.tan(next_delta)

    theta_mid = theta + 0.5 * next_omega * dt
    next_x = x + next_v * jnp.cos(theta_mid) * dt
    next_y = y + next_v * jnp.sin(theta_mid) * dt
    
    next_theta = theta + next_omega * dt
    next_theta = (next_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.array([next_x, next_y, next_theta, next_v, next_omega, b_omega])
