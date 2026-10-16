"""Standard PointCloud2 in/out; keep acquisition timestamp and optical frame."""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from .cloud_core import filter_xyz


class CloudProcessor(Node):
    def __init__(self):
        super().__init__('depth_cloud_processor')
        defaults = {
            'input_topic': '/camera/camera/depth/color/points',
            'min_depth_m': 0.2, 'max_depth_m': 4.0,
            'sample_stride': 4, 'stale_after_sec': 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
        self.minimum = self.get_parameter('min_depth_m').value
        self.maximum = self.get_parameter('max_depth_m').value
        self.stride = self.get_parameter('sample_stride').value
        self.timeout = self.get_parameter('stale_after_sec').value
        filter_xyz([], self.minimum, self.maximum, self.stride)
        if not 0 < self.timeout < float('inf'):
            raise ValueError('stale_after_sec must be positive and finite')
        self.publisher = self.create_publisher(PointCloud2, '/delivery/depth/points', qos_profile_sensor_data)
        self.diagnostics = self.create_publisher(DiagnosticArray, '/delivery/depth/status', 10)
        self.subscription = self.create_subscription(
            PointCloud2, self.get_parameter('input_topic').value,
            self.on_cloud, qos_profile_sensor_data)
        self.last_received = None
        self.create_timer(0.5, self.check_stale)

    def status(self, level, message, count=0):
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [DiagnosticStatus(
            level=level, name=self.get_fully_qualified_name(),
            hardware_id='realsense_depth', message=message,
            values=[KeyValue(key='output_points', value=str(count))])]
        self.diagnostics.publish(msg)

    def on_cloud(self, msg):
        self.last_received = time.monotonic()
        try:
            if not msg.header.frame_id:
                raise ValueError('Input cloud has no frame_id')
            xyz = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'))
            xyz = filter_xyz(xyz, self.minimum, self.maximum, self.stride)
            output = point_cloud2.create_cloud_xyz32(msg.header, xyz)
        except (ValueError, TypeError, AssertionError) as exc:
            self.status(DiagnosticStatus.ERROR, f'Invalid cloud: {exc}')
            return
        self.publisher.publish(output)
        self.status(DiagnosticStatus.OK if len(xyz) else DiagnosticStatus.WARN,
                    'ok' if len(xyz) else 'no_valid_points', len(xyz))

    def check_stale(self):
        if self.last_received is None:
            self.status(DiagnosticStatus.WARN, 'no_frames')
        elif time.monotonic() - self.last_received > self.timeout:
            self.status(DiagnosticStatus.WARN, 'stale')


def main(args=None):
    rclpy.init(args=args)
    node = CloudProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
