import jax
import jax.numpy as jnp
import bisect

def ekf_update(
    x_k_pred,
    P_k_pred,
    z_k,
    h,
    R,
    angle_mask=None,
):
    """
    z_k: Measurement
    h: Measurement function
    R: Measurement noise covariance
    angle_mask: Boolean array of same shape as z_k, True if the measurement is an angle
    """

    n = x_k_pred.shape[0]
    H_k = jax.jacfwd(h)(x_k_pred)

    y_k = z_k - h(x_k_pred)
    if angle_mask is not None:
        y_wrapped = (y_k + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        y_k = jnp.where(angle_mask, y_wrapped, y_k)

    S_k = H_k @ P_k_pred @ H_k.T + R
    K_k = jnp.linalg.solve(S_k.T, (P_k_pred @ H_k.T).T).T

    x_k = x_k_pred + K_k @ y_k
    I_KH = jnp.eye(n) - K_k @ H_k
    P_k = I_KH @ P_k_pred @ I_KH.T + K_k @ R @ K_k.T

    return x_k, P_k

def ekf_predict(
    f,
    x_k_1,
    u_k,
    dt,
    P_k_1,
    Q,
):
    """
    f : R^n x R^m -> R^n: Dynamics model
    x_k_1 in R^n: Current state
    u_k in R^m: Input
    dt: Time since last
    P_k_1: Current covariance
    Q: Process noise covariance
    """
    F_k = jax.jacfwd(f)(x_k_1, u_k, dt)

    x_k_pred = f(x_k_1, u_k, dt)
    P_k_pred = F_k @ P_k_1 @ F_k.T + Q * dt

    return x_k_pred, P_k_pred

class MultiRateEKF:
    def __init__(self, initial_x, initial_P, measurement_fcns, timestamp, buffer_size=50):
        """
        measurement_fcns: Dictionary mapping strings (measurement types) to functions.
        The functions should take as input the state, the covariance, the input, dt, and the measurement
        """
        self.measurement_fcns = measurement_fcns

        # len(self.history) == len(self.measurements) + 1
        self.history = [(timestamp, initial_x, initial_P)]
        self.measurements = []
        self.buffer_size = buffer_size

    def receive_measurement(self, measurement_type, z, u, timestamp):
        # Reject very old measurements
        if len(self.measurements) == self.buffer_size:
            oldest_timestamp = self.measurements[0][0]
            if timestamp < oldest_timestamp:
                print(f"Warning: {measurement_type} entry too old")
                return

        new_entry = (timestamp, measurement_type, z, u)
        bisect.insort(self.measurements, new_entry)

        while len(self.measurements) > self.buffer_size:
            self.measurements.pop(0)
            if len(self.history) > 1:
                self.history.pop(0)

        k = self.measurements.index(new_entry)

        # Roll back history to the exact state before this measurement and then replay
        self.history = self.history[:k+1]
        for i in range(k, len(self.measurements)):
            t_meas, m_type, m_z, m_u = self.measurements[i]
            t_prev, x, P = self.history[i]
            
            dt = t_meas - t_prev
            
            if dt < 0:
                self.history.append((t_prev, x, P))
                continue

            x_new, P_new = self.measurement_fcns[m_type](
                x, P, m_u, dt, m_z
            )
            
            self.history.append((t_meas, x_new, P_new))

        self.last_time = timestamp

    @property
    def x(self):
        return self.history[-1][1]

    @property
    def P(self):
        return self.history[-1][2]
