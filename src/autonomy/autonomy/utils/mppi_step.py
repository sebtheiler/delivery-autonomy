import jax
import jax.numpy as jnp

from autonomy.utils.dynamics import dynamics_model, WHEELBASE
from autonomy.algorithms.mppi import mppi

# TODO: make a parameter if you can
DT = 0.1

def dynamics_wrapper(state, action):
    return dynamics_model(state, action, dt=DT)

batch_dynamics = jax.vmap(dynamics_wrapper, in_axes=(0, 0))

sigma = jnp.array([0.35, 0.35])

def ell(state, action, t, ref_traj_window, v_target):
    x, y, theta, v, omega, b_omega = state
    v_cmd, raw_delta = action
    delta_cmd = jnp.clip(raw_delta, -0.6, 0.6)
    
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
    current_delta = jnp.arctan((omega * WHEELBASE) / safe_v)
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
