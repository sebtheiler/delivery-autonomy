import jax
import jax.numpy as jnp
from autonomy.utils.dynamics import dynamics_model, STATE_DIMENSION, INPUT_DIMENSION
from seblib.ekf import ekf_predict, ekf_update

Q = jnp.diag(jnp.array([
    0.05,  # X
    0.05,  # Y
    0.1,   # Theta
    1.5,   # Velocity
    1.0,   # Omega
    1e-6,  # Omega bias
]))

def h_imu(x):
    return jnp.array([x[4] + x[5]])

@jax.jit
def process_imu(x, P, u, dt, z):
    R = jnp.diag(jnp.array([0.05])) 

    x_pred, P_pred = ekf_predict(dynamics_model, x, u, dt, P, Q*dt)
    x_new, P_new = ekf_update(x_pred, P_pred, z, h_imu, R)

    return x_new, P_new

def h_encoders(x):
    return x[3:5]

@jax.jit
def process_encoders(x, P, u, dt, z):
    R = jnp.diag(jnp.array([0.25, 0.1]))

    x_pred, P_pred = ekf_predict(dynamics_model, x, u, dt, P, Q*dt)
    x_new, P_new = ekf_update(x_pred, P_pred, z, h_encoders, R)

    return x_new, P_new

def h_gps(x):
    return x[0:2]

@jax.jit
def process_gps(x, P, u, dt, z):
    R = jnp.diag(jnp.array([0.001, 0.001]))

    x_pred, P_pred = ekf_predict(dynamics_model, x, u, dt, P, Q*dt)
    x_new, P_new = ekf_update(x_pred, P_pred, z, h_gps, R)

    return x_new, P_new

def h_gps_heading(x):
    return x[2:3]

gps_heading_angle_mask = jnp.array([True]) # angles must be wrapped

@jax.jit
def process_gps_heading(x, P, u, dt, z):
    # higher than positional variance
    R = jnp.diag(jnp.array([0.05])) 
    
    x_pred, P_pred = ekf_predict(dynamics_model, x, u, dt, P, Q*dt)
    x_new, P_new = ekf_update(x_pred, P_pred, z, h_gps_heading, R, gps_heading_angle_mask)

    return x_new, P_new

MEASUREMENT_FCNS = {
    'imu': process_imu,
    'odom': process_encoders,
    'gps': process_gps,
    'gps_heading': process_gps_heading,
}
