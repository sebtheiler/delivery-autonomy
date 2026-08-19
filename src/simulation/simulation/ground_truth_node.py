import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
import math

MIN_DT = 1e-4

class GroundTruthExtractor(Node):
    def __init__(self):
        super().__init__("gt_extractor")

        self.subscription = self.create_subscription(
            TFMessage,
            "/world/washu_campus/dynamic_pose/info",
            self.listener_callback,
            10,
        )

        self.publisher = self.create_publisher(Odometry, "/ground_truth/odom", 10)

        # State history for numerical differentiation
        self.last_time = None
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0

    def listener_callback(self, msg):
        for t in msg.transforms:
            if t.child_frame_id != "delivery_robot":
                continue

            # Gazebo stamps these poses, but the bridge drops the stamp on the way
            # to a TFMessage, so the node clock is the only time source available.
            # It must therefore run with use_sim_time, or dt is measured against
            # the wall clock and every reported velocity is scaled by the
            # simulator's real time factor.
            current_time = self.get_clock().now().nanoseconds * 1e-9

            # Guards against a divide by roughly zero when two poses land together
            if self.last_time is not None and current_time - self.last_time < MIN_DT:
                return

            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation

            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = "world"
            odom.child_frame_id = "delivery_robot"

            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = t.transform.translation.z
            odom.pose.pose.orientation = q

            dt = current_time - self.last_time if self.last_time is not None else 0.0
            if dt > 0.0:
                # Project displacement onto the robot's forward axis
                odom.twist.twist.linear.x = (
                    (x - self.last_x) * math.cos(yaw) + (y - self.last_y) * math.sin(yaw)
                ) / dt

                dyaw = (yaw - self.last_yaw + math.pi) % (2.0 * math.pi) - math.pi
                odom.twist.twist.angular.z = dyaw / dt

            self.last_time = current_time
            self.last_x = x
            self.last_y = y
            self.last_yaw = yaw

            self.publisher.publish(odom)
            break

def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
