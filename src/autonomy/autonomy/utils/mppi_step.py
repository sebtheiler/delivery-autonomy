import jax
import jax.numpy as jnp

from autonomy.utils.dynamics import dynamics_model
from autonomy.algorithms.mppi import mppi

# TODO: make a parameter if you can
DT = 0.1

def dynamics_wrapper(state, action):
    return dynamics_model(state, action, dt=DT)

batch_dynamics = jax.vmap(dynamics_wrapper, in_axes=(0, 0))

sigma = jnp.array([0.35, 0.35])

# def ell(state, action, t, ref_traj_window, v_target):
#     x, y, theta, v, omega, b_omega = state
#     v_cmd, raw_delta = action
#     delta = jnp.clip(raw_delta, -0.6, 0.6)
    
#     target_state = ref_traj_window[t]
#     x_ref, y_ref, theta_ref = target_state[0], target_state[1], target_state[2]
    
#     dx = x - x_ref
#     dy = y - y_ref
    
#     # 1. Cross-Track Error (CTE)
#     cte = -dx * jnp.sin(theta_ref) + dy * jnp.cos(theta_ref)
    
#     # 2. Pure Path Alignment
#     # Dropping the Stanley logic. Just align with the path's tangent.
#     theta_error = theta - theta_ref
#     theta_error = (theta_error + jnp.pi) % (2 * jnp.pi) - jnp.pi

#     v_error = v - v_target
    
#     # 3. Rebalanced Weights
#     w_cte = 15.0    # Primary objective: stay on the line
#     w_theta = 5.0   # Guide the nose, but CTE takes priority
#     w_steer = 2.0   # Dampens bang-bang oscillations fighting the 0.545s STEER_TAU
#     w_omega = 5.0   # Smooths out erratic yaw rates
#     w_v = 20.0      # Forward progress driver
    
#     # Note: w_ate is completely removed. Forward progress is handled by w_v.
    
#     return (w_cte * (cte**2) + 
#             w_theta * (theta_error**2) + 
#             w_steer * (delta**2) + 
#             w_omega * (omega**2) + 
#             w_v * (v_error**2))

def ell(state, action, t, ref_traj_window, v_target):
    x, y, theta, v, omega, b_omega = state
    v_cmd, raw_delta = action
    delta_cmd = jnp.clip(raw_delta, -0.6, 0.6)
    
    target_state = ref_traj_window[t]
    x_ref, y_ref, theta_ref = target_state[0], target_state[1], target_state[2]
    
    dx = x - x_ref
    dy = y - y_ref
    
    # Cross-Track Error
    cte = -dx * jnp.sin(theta_ref) + dy * jnp.cos(theta_ref)
    
    # Pure Path Alignment
    theta_error = theta - theta_ref
    theta_error = (theta_error + jnp.pi) % (2 * jnp.pi) - jnp.pi
    v_error = v - v_target
    
    # Pseudo-Huber
    cte_cost = jnp.sqrt(cte**2 + 0.05) - jnp.sqrt(0.05)

    # 4. THE MAGIC BULLET: Steering Effort Penalty
    # Reconstruct the current physical steering angle from the state
    WHEELBASE = 0.615 # TODO: needs to be param, also this is a little cooked
    safe_v = jnp.where(jnp.abs(v) < 1e-5, jnp.sign(v + 1e-10) * 1e-5, v)
    current_delta = jnp.arctan((omega * WHEELBASE) / safe_v)
    
    steer_effort_cost = (delta_cmd - current_delta)**2

    LANE_TOLERANCE = 1.0
    abs_cte = jnp.abs(cte)
    grass_death_cost = jnp.where(abs_cte > LANE_TOLERANCE, 10000.0 * (abs_cte - LANE_TOLERANCE)**2, 0.0)
    
    # weights
    w_cte = 90.0
    w_theta = 30.0
    w_steer_effort = 25.0
    w_omega = 5.0
    w_v = 20.0
    w_grass = 0.0
    
    return (w_cte * cte_cost + 
            w_theta * (theta_error**2) + 
            w_steer_effort * steer_effort_cost + 
            w_omega * (omega**2) + 
            w_v * (v_error**2) +
            w_grass * grass_death_cost)

def phi(state, ref_traj_window):
    # Extract the robot's state at the very last step of the horizon
    x, y, theta, v, omega, b_omega = state
    
    # Extract the ideal target state at the end of the horizon
    target_state = ref_traj_window[-1]
    x_ref, y_ref, theta_ref = target_state[0], target_state[1], target_state[2]
    
    dx = x - x_ref
    dy = y - y_ref
    
    # Transform Euclidean error into the path's local frame
    cte = -dx * jnp.sin(theta_ref) + dy * jnp.cos(theta_ref)
    
    # Terminal heading error
    theta_error = theta - theta_ref
    theta_error = (theta_error + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # ---------------------------------------------------------
    # TERMINAL WEIGHTS
    # ---------------------------------------------------------
    w_cte_term = 200.0   # Massive penalty for ending off-path
    w_theta_term = 50.0  # Moderate penalty for ending unaligned
    
    # Note: ATE penalty is removed. The robot is now free to slow down 
    # for sharp corners without incurring a massive cost for missing the 
    # "ghost robot" horizon target.
    
    return w_cte_term * (cte**2) + w_theta_term * (theta_error**2)

@jax.jit
def mppi_step(key, x0, U_nominal, ref_traj, v_target):
    """
    x0: (STATE_DIMENSION,)
    U_nominal: (T, INPUT_DIMENSION)
    ref_traj: (T, 3) containing [x, y, theta]
    """

    # Create closures that inject ref_traj into the unbatched functions
    def ell_closure(state, action, t):
        return ell(state, action, t, ref_traj, v_target)
        
    def phi_closure(state):
        return phi(state, ref_traj)

    # Vmap the closures over the batch dimension (axis 0 for state and action).
    # t is a scalar, so we use None for its axis.
    batch_ell = jax.vmap(ell_closure, in_axes=(0, 0, None))
    batch_phi = jax.vmap(phi_closure, in_axes=(0,))

    return mppi(
        key=key, 
        f=batch_dynamics,
        x0=x0, 
        U=U_nominal, 
        phi=batch_phi, 
        ell=batch_ell,
        sigma=sigma,
        K=3000,
        lam=30.0,
        lower_bounds = jnp.array([-0.5, -0.6]),
        upper_bounds = jnp.array([1.5, 0.6]),
    )
