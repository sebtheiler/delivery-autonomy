"""Explicit start/finish traversal recorder. No motor commands or auto map matching."""
import math
import uuid
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger

from delivery_metrics.core import Measurement, Store


class Recorder(Node):
    def __init__(self):
        super().__init__('travel_time_recorder')
        for key, value in {
            'odom_topic': '/odom', 'database_path': 'delivery_metrics.sqlite3',
            'robot_id': 'delivery_01', 'route_id': 'manual_trial',
            'from_node': '', 'to_node': '', 'edge_key': '',
            'speed_threshold': 0.05, 'max_odom_gap': 0.5,
        }.items():
            self.declare_parameter(key, value)
        self.store = Store(self.get_parameter('database_path').value)
        self.active = None
        self.last_clock = self.now()
        self.publisher = self.create_publisher(String, '/delivery/traversal_metrics', 10)
        self.subscription = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value,
            self.on_odom, qos_profile_sensor_data)
        self.start_service = self.create_service(Trigger, '~/start', self.start)
        self.finish_service = self.create_service(Trigger, '~/finish', self.finish)
        self.cancel_service = self.create_service(Trigger, '~/cancel', self.cancel)
        self.clock_timer = self.create_timer(0.1, self.check_clock)
        self.get_logger().info('Ready: call ~/start, then ~/finish or ~/cancel.')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def check_clock(self):
        current = self.now()
        if self.active is not None and current < self.last_clock:
            self.complete('clock_reset', self.last_clock)
        self.last_clock = current
        return current

    def start(self, request, response):
        current = self.check_clock()
        if self.active is not None:
            response.message = 'Already recording; finish or cancel first.'
            return response
        if self.get_parameter('use_sim_time').value and current <= 0:
            response.message = 'Waiting for simulation /clock.'
            return response
        threshold = self.get_parameter('speed_threshold').value
        max_gap = self.get_parameter('max_odom_gap').value
        if not math.isfinite(threshold) or threshold < 0 or not math.isfinite(max_gap) or max_gap <= 0:
            response.message = 'Invalid speed_threshold or max_odom_gap.'
            return response
        record = {key: self.get_parameter(key).value for key in
                  ('robot_id', 'route_id', 'from_node', 'to_node', 'edge_key')}
        record.update({
            'record_id': str(uuid.uuid4()), 'status': 'recording',
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'start_ros_seconds': current,
            'is_simulation': self.get_parameter('use_sim_time').value,
            'odom_topic': self.get_parameter('odom_topic').value,
            'boundary_source': 'explicit_service',
        })
        self.store.save(record)
        self.active = record, Measurement(current, threshold, max_gap)
        response.success = True
        response.message = record['record_id']
        return response

    def on_odom(self, message):
        current = self.check_clock()
        if self.active is None:
            return
        measurement = self.active[1]
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        # Header stamps must use the same clock as the recorder. Never silently
        # substitute receipt time for missing/future sensor timestamps.
        if stamp <= 0 or stamp > current:
            measurement.rejected += 1
            return
        measurement.sample(stamp, message.twist.twist.linear.x)

    def complete(self, status, end):
        record, measurement = self.active
        result = dict(record)
        result.update(measurement.result(end))
        result.update(status=status, end_ros_seconds=end,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        result['usable_for_route_learning'] = (
            status == 'completed' and result['elapsed_seconds'] > 0
            and result['odom_coverage_fraction'] >= 0.95
            and measurement.samples >= 2)
        payload = self.store.save(result)  # Durable copy before publication.
        self.active = None
        self.publisher.publish(String(data=payload))
        self.get_logger().info(payload)
        return result['record_id']

    def finish(self, request, response):
        current = self.check_clock()
        if self.active is None:
            response.message = 'No active traversal.'
            return response
        response.message = self.complete('completed', current)
        response.success = True
        return response

    def cancel(self, request, response):
        current = self.check_clock()
        if self.active is None:
            response.message = 'No active traversal.'
            return response
        response.message = self.complete('canceled', current)
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Unfinished traversals remain 'recording' on disk; startup marks them
        # 'interrupted'. Do not publish after ROS context shutdown.
        node.store.close()
        node.destroy_node()
        rclpy.try_shutdown()
