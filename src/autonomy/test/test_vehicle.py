import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from autonomy.utils.dynamics import (
    INPUT_DIMENSION, SIM_VEHICLE, VEHICLE_PARAMS, Vehicle, dynamics_model, vehicle_param_problem)

# The real robot's geometry (delivery-robo hardware.yaml: wheelbase, max_steer_rad); its lags are unmeasured
ROBOT_TDM41 = Vehicle(wheelbase=0.36322, max_steer_rad=0.4363)


def test_defaults_are_the_sim_car_and_valid():
    assert SIM_VEHICLE == Vehicle(0.5, 0.6, 0.545, 0.01, 5.0)
    assert vehicle_param_problem(**VEHICLE_PARAMS) is None
    assert vehicle_param_problem(**ROBOT_TDM41._asdict()) is None


@pytest.mark.parametrize('name, value', [
    ('wheelbase', 0.0), ('wheelbase', -0.36), ('wheelbase', math.inf), ('wheelbase', math.nan),
    ('max_steer_rad', 0.0), ('max_steer_rad', -0.44), ('max_steer_rad', math.pi / 2),
    ('max_steer_rad', math.nan), ('steer_tau', 0.0), ('drive_tau', -0.01), ('max_accel', 0.0)])
def test_unusable_vehicles_are_refused(name, value):
    assert vehicle_param_problem(**{**VEHICLE_PARAMS, name: value}) is not None


def turning_radius_tdm41(vehicle, delta_cmd, v=1.0, seconds=8.0):
    """Radius of the circle the model settles on at v with delta_cmd held, from straight ahead."""
    state = jnp.array([0.0, 0.0, 0.0, v, 0.0, 0.0])
    for _ in range(round(seconds / 0.1)):
        state = dynamics_model(state, jnp.array([v, delta_cmd]), 0.1, vehicle)
    return v / float(state[4])


def test_full_lock_turns_on_the_vehicles_smallest_circle():
    assert turning_radius_tdm41(SIM_VEHICLE, 1.0) == pytest.approx(0.5 / math.tan(0.6), rel=1e-3)
    assert turning_radius_tdm41(ROBOT_TDM41, 1.0) == pytest.approx(0.36322 / math.tan(0.4363), rel=1e-3)
    # Within the limit the angle is the angle: the wheelbase alone sets the radius
    assert turning_radius_tdm41(ROBOT_TDM41, 0.2) == pytest.approx(0.36322 / math.tan(0.2), rel=1e-3)


def test_the_lags_and_the_acceleration_limit_are_the_vehicles():
    rest, go = jnp.zeros(6), jnp.array([1.5, 0.0])
    assert float(dynamics_model(rest, go, 0.1)[3]) == pytest.approx(0.5, abs=1e-4)  # 5 m/s^2 for 0.1 s
    gentle = SIM_VEHICLE._replace(max_accel=2.0)
    assert float(dynamics_model(rest, go, 0.1, gentle)[3]) == pytest.approx(0.2, abs=1e-4)
    sluggish = SIM_VEHICLE._replace(drive_tau=1.0, max_accel=100.0)
    assert float(dynamics_model(rest, go, 0.1, sluggish)[3]) == pytest.approx(
        1.5 * (1.0 - math.exp(-0.1)), rel=1e-4)
    straight, turn = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), jnp.array([1.0, 0.3])
    for tau in (0.545, 0.2):
        omega = float(dynamics_model(straight, turn, 0.1, SIM_VEHICLE._replace(steer_tau=tau))[4])
        assert math.atan(omega * 0.5) == pytest.approx(0.3 * (1.0 - math.exp(-0.1 / tau)), rel=1e-4)


def test_leaving_the_vehicle_out_models_the_sim_car():
    state, u = jnp.array([1.0, 2.0, 0.3, 0.8, 0.2, 0.0]), jnp.array([1.2, 0.4])
    np.testing.assert_array_equal(dynamics_model(state, u, 0.1),
                                  dynamics_model(state, u, 0.1, SIM_VEHICLE))

from autonomy.utils.mppi_step import free_window, mppi_step
from autonomy.utils.reference_trajectory import get_local_reference


def circle_plan_tdm41(vehicle, iterations=10, radius=2.0, v=1.0):
    """The steering MPPI plans with `vehicle` on a left-hand circle, from a state turning steadily
    on it, after `iterations` steps from that one state (warm started, clipped like the
    controller's)."""
    s = np.arange(0.0, 1.5 * math.pi * radius, 0.05)
    path_xy = np.stack([radius * np.sin(s / radius), radius * (1.0 - np.cos(s / radius))], axis=-1)
    state = np.array([0.0, 0.0, 0.0, v, v / radius, 0.0])
    ref, _ = get_local_reference(state, path_xy, s / radius, s, 50, 0.1, v_target=v)
    lower = jnp.array([0.0, -vehicle.max_steer_rad])
    upper = jnp.array([v, vehicle.max_steer_rad])
    U = jnp.zeros((50, INPUT_DIMENSION)).at[:, 0].set(v)
    key = jax.random.PRNGKey(0)
    for _ in range(iterations):
        key, sub = jax.random.split(key)
        U = jnp.clip(mppi_step(sub, state, U, jnp.array(ref), v, lower, upper, free_window(),
                               np.int32(51), vehicle=vehicle), lower, upper)
    return np.asarray(U[:, 1])


def test_a_new_vehicle_does_not_compile_the_step_again():
    circle_plan_tdm41(SIM_VEHICLE, iterations=1)
    compiled = mppi_step._cache_size()
    circle_plan_tdm41(ROBOT_TDM41, iterations=1)
    circle_plan_tdm41(Vehicle(0.4, 0.5, 0.3, 0.05, 2.0), iterations=1)
    assert mppi_step._cache_size() == compiled


def test_mppi_plans_the_steering_its_vehicle_needs():
    for vehicle in (SIM_VEHICLE, ROBOT_TDM41):
        planned = float(circle_plan_tdm41(vehicle)[:40].mean())
        assert planned == pytest.approx(math.atan(vehicle.wheelbase / 2.0), abs=0.02), vehicle
