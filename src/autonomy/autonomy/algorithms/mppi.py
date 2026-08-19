import jax
import jax.numpy as jnp

def mppi(key, f, x0, U, phi, ell, lower_bounds, upper_bounds, sigma=0.5, K=1000, lam=0.1):
    """
    key: PRNG key
    f: Dynamics (needs to be vmapped)
    x0: Initial state
    U: Nominal control trajectory (T, m)
    phi: Final cost function
    ell: Rolling cost function
    """
    T, m = U.shape

    raw_noise = jax.random.normal(key, (T, K, m))
    
    beta = 0.6  # 0.0 leaves the noise white, 1.0 freezes it entirely
    
    def smooth_noise(carry, noise_t):
        smoothed_val = beta * carry + (1.0 - beta) * noise_t
        return smoothed_val, smoothed_val
    
    _, eps_time_major = jax.lax.scan(smooth_noise, jnp.zeros((K, m)), raw_noise)
    
    epsilon = jnp.swapaxes(eps_time_major, 0, 1) * sigma
    
    V = U + epsilon
    V = jnp.clip(V, lower_bounds, upper_bounds)

    def scan_fn(carry, scan_elements):
        x, J_accum = carry
        V_t, t = scan_elements
    
        x_next = f(x, V_t)
    
        step_cost = ell(x, V_t, t)
        J_next = J_accum + step_cost
    
        return (x_next, J_next), None
    
    V_time_major = jnp.swapaxes(V, 0, 1) # (K, T, m) -> (T, K, m)
    
    t_array = jnp.arange(T)
    
    init_states = jnp.tile(x0, (K, 1))
    init_costs = jnp.zeros(K)
    init_carry = (init_states, init_costs)
    
    (x_final, J), _ = jax.lax.scan(
        scan_fn,
        init_carry,
        (V_time_major, t_array),
    )
    J += phi(x_final)

    rho = jnp.min(J)
    w = jnp.exp(-(1/lam) * (J - rho))
    w /= jnp.sum(w)

    U_hat = U + jnp.sum(w[:, None, None] * epsilon, axis=0)

    return U_hat
