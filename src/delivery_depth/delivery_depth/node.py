"""Subscribe to RealSense depth; publish measurements, never motor commands."""
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from .core import measure


class DepthProcessor(Node):
    def __init__(self):
        super().__init__('depth_processor')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('roi_size', 40)
        self.declare_parameter('min_valid_fraction', 0.5)
        self.declare_parameter('stale_after_sec', 2.0)
        self.roi_size = self.get_parameter('roi_size').value
        self.min_valid = self.get_parameter('min_valid_fraction').value
        self.stale_after = self.get_parameter('stale_after_sec').value
        if self.roi_size <= 0 or not 0 <= self.min_valid <= 1 or self.stale_after <= 0:
            raise ValueError('Invalid ROI, coverage, or timeout parameter')
        self.metrics = self.create_publisher(String, '/delivery/depth_metrics', 10)
        self.distance = self.create_publisher(Float32, '/delivery/center_depth_m', 10)
        self.last_received = None
        self.subscription = self.create_subscription(
            Image, self.get_parameter('depth_topic').value,
            self.on_image, qos_profile_sensor_data)
        self.timer = self.create_timer(0.5, self.check_stale)
        self.get_logger().info('Publishing depth metrics and center depth in meters')

    def publish(self, result):
        self.metrics.publish(String(data=json.dumps(result, allow_nan=False)))
        value = result.get('median_depth_m')
        self.distance.publish(Float32(data=value if value is not None else math.nan))

    def on_image(self, msg):
        self.last_received = time.monotonic()
        try:
            result = measure(msg.data, msg.width, msg.height, msg.step,
                             msg.encoding, msg.is_bigendian,
                             self.roi_size, self.min_valid)
        except (ValueError, TypeError) as exc:
            result = {'status': 'invalid_image', 'error': str(exc),
                      'median_depth_m': None, 'nearest_depth_m': None}
        result['frame_id'] = msg.header.frame_id
        result['stamp'] = {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec}
        self.publish(result)

    def check_stale(self):
        age = None if self.last_received is None else time.monotonic() - self.last_received
        if age is None or age > self.stale_after:
            self.publish({'status': 'no_frames' if age is None else 'stale',
                          'seconds_since_frame': age,
                          'median_depth_m': None, 'nearest_depth_m': None})


def main(args=None):
    rclpy.init(args=args)
    node = DepthProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
