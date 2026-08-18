# import jax
# import jax.numpy as jnp

# STATE_DIMENSION = 6
# INPUT_DIMENSION = 2
# WHEELBASE = 0.5 
# MAX_STEER = 0.6 
# TAU = 0.25 # motor response time constant

# @jax.jit
# def dynamics_model(state: jnp.ndarray, inputs: jnp.ndarray, dt: float) -> jnp.ndarray:
#     x, y, theta, v, omega, b_omega = state
#     v_cmd, delta = inputs

#     theta_mid = theta + 0.5 * omega * dt
#     next_x = x + v * jnp.cos(theta_mid) * dt
#     next_y = y + v * jnp.sin(theta_mid) * dt
#     next_theta = theta + omega * dt
#     next_theta = (next_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
#     next_v = v + ((v_cmd - v) / TAU) * dt
    
#     omega_cmd = (v / WHEELBASE) * jnp.tan(jnp.clip(delta, -MAX_STEER, MAX_STEER))
#     next_omega = omega + ((omega_cmd - omega) / TAU) * dt

#     return jnp.array([next_x, next_y, next_theta, next_v, next_omega, b_omega])
import jax
import jax.numpy as jnp

# STATE_DIMENSION = 6
# INPUT_DIMENSION = 2
# WHEELBASE = 0.5 
# MAX_STEER = 0.6 
# DRIVE_TAU = 0.25  # Motor response time
# STEER_TAU = 0.20  # Servo response time
# MAX_ACCEL = 3.0   # Matches Gazebo exactly
STATE_DIMENSION = 6
INPUT_DIMENSION = 2
WHEELBASE = 0.615      # Accounts for true tire slip in Gazebo
MAX_STEER = 0.6        
DRIVE_TAU = 0.01       # Gazebo is instantly applying velocity
STEER_TAU = 0.545      # The true lag of your simulated steering rack
MAX_ACCEL = 5.0        # Effectively infinite for this use case

@jax.jit
def dynamics_model(state: jnp.ndarray, inputs: jnp.ndarray, dt: float) -> jnp.ndarray:
    x, y, theta, v, omega, b_omega = state
    v_cmd, delta_cmd = inputs

    # 1. Enforce Gazebo's Exact Acceleration Limits
    desired_accel = (v_cmd - v) / DRIVE_TAU
    actual_accel = jnp.clip(desired_accel, -MAX_ACCEL, MAX_ACCEL)
    next_v = v + actual_accel * dt

    # 2. Extract current physical steering angle from the state
    # Prevent zero-crossing singularity while maintaining the correct quadrant sign
    safe_v = jnp.where(jnp.abs(v) < 1e-5, jnp.sign(v + 1e-10) * 1e-5, v)
    current_delta = jnp.arctan((omega * WHEELBASE) / safe_v)
    
    # 3. Apply physical lag to the steering rack, not the chassis yaw
    delta_cmd = jnp.clip(delta_cmd, -MAX_STEER, MAX_STEER)
    next_delta = current_delta + ((delta_cmd - current_delta) / STEER_TAU) * dt
    
    # 4. Compute true resulting yaw rate based on the delayed wheel position
    next_omega = (next_v / WHEELBASE) * jnp.tan(next_delta)

    # 5. Integrate position using the UPDATED velocities
    theta_mid = theta + 0.5 * next_omega * dt
    next_x = x + next_v * jnp.cos(theta_mid) * dt
    next_y = y + next_v * jnp.sin(theta_mid) * dt
    
    next_theta = theta + next_omega * dt
    next_theta = (next_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.array([next_x, next_y, next_theta, next_v, next_omega, b_omega])
