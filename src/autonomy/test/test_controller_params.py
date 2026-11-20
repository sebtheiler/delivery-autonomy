import numpy as np

from autonomy.controller import CONTROLLER_PARAMS, controller_param_problem, param_problem, vehicle_from
from autonomy.utils.costmap_speed import GOVERNOR_PARAMS
from autonomy.utils.dynamics import VEHICLE_PARAMS, Vehicle


def test_defaults_are_accepted():
    assert controller_param_problem(**CONTROLLER_PARAMS) is None


def test_speeds_must_stay_inside_mppi_bounds():
    for name, value in [('max_speed', 0.0), ('max_speed', -0.3), ('max_speed', 1.6),
                        ('min_speed', 0.1), ('min_speed', -0.6)]:
        assert controller_param_problem(**{**CONTROLLER_PARAMS, name: value}) is not None, (name, value)


def test_slow_and_forward_only_settings_are_accepted():
    assert controller_param_problem(**{**CONTROLLER_PARAMS, 'max_speed': 0.8, 'min_speed': 0.0}) is None


def test_tolerances_must_be_positive():
    for name in ('xy_goal_tolerance', 'yaw_goal_tolerance'):
        assert controller_param_problem(**{**CONTROLLER_PARAMS, name: 0.0}) is not None, name


def test_the_vehicle_is_checked_with_the_rest():
    defaults = {**CONTROLLER_PARAMS, **GOVERNOR_PARAMS, **VEHICLE_PARAMS}
    assert param_problem(defaults) is None
    assert param_problem({**defaults, 'wheelbase': 0.36322, 'max_steer_rad': 0.4363}) is None
    for name, value in [('wheelbase', 0.0), ('max_steer_rad', 1.6), ('steer_tau', 0.0),
                        ('drive_tau', -0.01), ('max_accel', 0.0)]:
        assert param_problem({**defaults, name: value}) is not None, (name, value)


def test_the_vehicle_is_plain_floats():
    settings = {**CONTROLLER_PARAMS, **GOVERNOR_PARAMS, **VEHICLE_PARAMS, 'wheelbase': np.float64(0.36322)}
    vehicle = vehicle_from(settings)
    assert vehicle == Vehicle(wheelbase=0.36322)
    assert all(type(value) is float for value in vehicle)
