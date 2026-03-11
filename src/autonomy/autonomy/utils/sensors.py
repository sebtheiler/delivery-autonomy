import jax
import jax.numpy as jnp
from autonomy.utils.dynamics import dynamics_model, STATE_DIMENSION, INPUT_DIMENSION
from seblib.ekf import ekf_predict, ekf_update

Q = jnp.diag(jnp.array([
    0.05,   # X position variance 
    0.05,   # Y position variance
    0.1,    # Theta variance
    0.5     # Velocity variance
]))

def h_imu(x):
    return x[2:3]

imu_angle_mask = jnp.array([True])

@jax.jit
def process_imu(x, P, u, dt, z):
    R = jnp.diag(jnp.array([0.01]))

    x_pred, P_pred = ekf_predict(dynamics_model, x, u, dt, P, Q*dt)
    x_new, P_new = ekf_update(x_pred, P_pred, z, h_imu, R, imu_angle_mask)

    return x_new, P_new

def h_encoders(x):
    return x[3:4]

@jax.jit
def process_encoders(x, P, u, dt, z):
    R = jnp.diag(jnp.array([0.05]))

    x_pred, P_pred = ekf_predict(dynamics_model, x, u, dt, P, Q*dt)
    x_new, P_new = ekf_update(x_pred, P_pred, z, h_encoders, R)

    return x_new, P_new

def h_gps(x):
    return x[0:2]

@jax.jit
def process_gps(x, P, u, dt, z):
    R = jnp.diag(jnp.array([0.5, 0.5]))

    x_pred, P_pred = ekf_predict(dynamics_model, x, u, dt, P, Q*dt)
    x_new, P_new = ekf_update(x_pred, P_pred, z, h_gps, R)

    return x_new, P_new

MEASUREMENT_FCNS = {
    'imu': process_imu,
    'odom': process_encoders,
    'gps': process_gps,
}
