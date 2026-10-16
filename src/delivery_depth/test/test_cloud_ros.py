"""Exercise real ROS serialization and pub/sub using isolated test topics."""
import time
import unittest
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs_py import point_cloud2
from delivery_depth.cloud_node import CloudProcessor


class RosCloudTests(unittest.TestCase):
    def test_roundtrip_and_stale(self):
        rclpy.init(args=['--ros-args',
            '-r', '/camera/camera/depth/color/points:=/test_delivery/input',
            '-r', '/delivery/depth/points:=/test_delivery/output',
            '-r', '/delivery/depth/status:=/test_delivery/status',
            '-p', 'sample_stride:=1'])
        processor = CloudProcessor()
        probe = Node('depth_test_probe')
        executor = SingleThreadedExecutor()
        executor.add_node(processor)
        executor.add_node(probe)
        outputs, statuses = [], []
        publisher = probe.create_publisher(PointCloud2, '/test_delivery/input', qos_profile_sensor_data)
        sub = probe.create_subscription(PointCloud2, '/test_delivery/output', outputs.append, qos_profile_sensor_data)
        status_sub = probe.create_subscription(DiagnosticArray, '/test_delivery/status', statuses.append, 10)
        header = Header(frame_id='test_optical_frame')
        header.stamp.sec = 123
        sample = point_cloud2.create_cloud_xyz32(header, [[1., 2., 1.], [0., 0., 0.], [0., 0., 9.]])
        try:
            deadline = time.monotonic() + 10
            while not outputs and time.monotonic() < deadline:
                publisher.publish(sample)
                executor.spin_once(timeout_sec=0.05)
            self.assertTrue(outputs, 'No cloud output received over ROS')
            self.assertEqual(outputs[0].header, header)
            np.testing.assert_allclose(point_cloud2.read_points_numpy(outputs[0]), [[1, 2, 1]])
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
                if any(s.status[0].message == 'stale' for s in statuses):
                    break
            self.assertTrue(any(s.status[0].message == 'stale' for s in statuses))
        finally:
            executor.shutdown()
            processor.destroy_node()
            probe.destroy_node()
            rclpy.shutdown()
