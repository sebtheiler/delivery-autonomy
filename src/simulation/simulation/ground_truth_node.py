import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
import math

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
            if t.child_frame_id == "delivery_robot":
                current_time = self.get_clock().now()

                x = t.transform.translation.x
                y = t.transform.translation.y
                z = t.transform.translation.z
                q = t.transform.rotation

                # Extract yaw from quaternion
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                odom = Odometry()
                odom.header.stamp = current_time.to_msg()
                odom.header.frame_id = "world" 
                odom.child_frame_id = "delivery_robot"

                # Copy position and orientation
                odom.pose.pose.position.x = x
                odom.pose.pose.position.y = y
                odom.pose.pose.position.z = z
                odom.pose.pose.orientation = q

                # Compute Twist using numerical differentiation
                if self.last_time is not None:
                    # Convert to seconds
                    dt = (current_time.nanoseconds - self.last_time.nanoseconds) * 1e-9

                    if dt > 0.001:  # Prevent division by zero
                        dx = x - self.last_x
                        dy = y - self.last_y

                        # Project global displacement onto the robot's local forward axis
                        v = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt

                        # Calculate wrapped yaw rate
                        dyaw = yaw - self.last_yaw
                        dyaw = (dyaw + math.pi) % (2.0 * math.pi) - math.pi
                        omega = dyaw / dt

                        odom.twist.twist.linear.x = v
                        odom.twist.twist.angular.z = omega
                else:
                    # First frame: zero velocity
                    odom.twist.twist.linear.x = 0.0
                    odom.twist.twist.angular.z = 0.0

                # Update history for the next frame
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
