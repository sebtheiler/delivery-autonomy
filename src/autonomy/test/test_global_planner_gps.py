from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pyproj import Transformer
from sensor_msgs.msg import NavSatFix
from autonomy.global_planner import GlobalPlanningServer


def planner():
    node = SimpleNamespace(
        use_gps_start=True, latest_fix=None, gps_max_age=2.0,
        G_projected=SimpleNamespace(graph={'crs': 'EPSG:32615'}),
        utm_origin=(734856.4630941747, 4281187.006303415),
        get_logger=lambda: Mock(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100000000000)),
    )
    return node


def fix(lat=38.648, lon=-90.3025):
    msg = NavSatFix()
    msg.status.status = 0
    msg.latitude, msg.longitude = lat, lon
    msg.header.stamp.sec = 100
    return msg


def test_start_follows_latest_gps_and_preserves_map_frame():
    node = planner()
    project = Transformer.from_crs(4326, 32615, always_xy=True)
    positions = []
    for msg in (fix(), fix(38.6485, -90.302)):
        GlobalPlanningServer.gps_callback(node, msg)
        local = GlobalPlanningServer.get_current_robot_pose(node)
        expected = project.transform(msg.longitude, msg.latitude)
        assert tuple(local[i] + node.utm_origin[i] for i in (0, 1)) == pytest.approx(expected)
        positions.append(local)
    assert positions[0] != positions[1]


@pytest.mark.parametrize('problem', ['missing', 'no_fix', 'nan', 'range', 'old_stamp', 'old_receipt'])
def test_bad_gps_never_falls_back_to_a_fixed_start(problem):
    node = planner()
    msg = fix()
    if problem == 'no_fix':
        msg.status.status = -1
    elif problem == 'nan':
        msg.latitude = float('nan')
    elif problem == 'range':
        msg.longitude = 181.0
    elif problem == 'old_stamp':
        msg.header.stamp.sec = 90
    if problem != 'missing':
        GlobalPlanningServer.gps_callback(node, msg)
    if problem == 'old_receipt':
        node.latest_fix = (msg, -100.0)
    assert GlobalPlanningServer.get_current_robot_pose(node) == (None, None)


def test_simulation_keeps_calibrated_tf_position():
    node = planner()
    node.use_gps_start = False
    node.tf_buffer = Mock()
    node.tf_buffer.lookup_transform.return_value.transform.translation = SimpleNamespace(x=5.0, y=7.0)
    assert GlobalPlanningServer.get_current_robot_pose(node) == (5.0, 7.0)
