import jax
import jax.numpy as jnp

STATE_DIMENSION = 5  # [x, y, theta, v, omega]
INPUT_DIMENSION = 2  # [v_cmd, delta]
WHEELBASE = 0.5 
MAX_STEER = 0.785 
TAU = 0.25 # motor response time constant

@jax.jit
def dynamics_model(state: jnp.ndarray, inputs: jnp.ndarray, dt: float) -> jnp.ndarray:
    x, y, theta, v, omega = state
    v_cmd, delta = inputs

    theta_mid = theta + 0.5 * omega * dt
    next_x = x + v * jnp.cos(theta_mid) * dt
    next_y = y + v * jnp.sin(theta_mid) * dt
    next_theta = theta + omega * dt
    next_theta = (next_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    next_v = v + ((v_cmd - v) / TAU) * dt
    
    omega_cmd = (v / WHEELBASE) * jnp.tan(jnp.clip(delta, -MAX_STEER, MAX_STEER))
    next_omega = omega + ((omega_cmd - omega) / TAU) * dt

    return jnp.array([next_x, next_y, next_theta, next_v, next_omega])
