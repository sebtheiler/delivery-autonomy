import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry


class GroundTruthExtractor(Node):
    def __init__(self):
        super().__init__("gt_extractor")

        # Subscribe to the raw topic coming from ros_ign_bridge
        self.subscription = self.create_subscription(
            TFMessage,
            "/world/washu_campus/dynamic_pose/info",
            self.listener_callback,
            10,
        )

        # Publish the clean Odometry message
        self.publisher = self.create_publisher(Odometry, "/ground_truth/odom", 10)

    def listener_callback(self, msg):
        for t in msg.transforms:
            # Filter out the wheels and everything else
            if t.child_frame_id == "delivery_robot":
                odom = Odometry()

                # Fix the header (Ignition sets it to 0, so we use ROS time)
                odom.header.stamp = self.get_clock().now().to_msg()
                odom.header.frame_id = "world"  # Inject the missing parent frame
                odom.child_frame_id = "delivery_robot"

                # Copy translation to pose position
                odom.pose.pose.position.x = t.transform.translation.x
                odom.pose.pose.position.y = t.transform.translation.y
                odom.pose.pose.position.z = t.transform.translation.z

                # Copy rotation
                odom.pose.pose.orientation = t.transform.rotation

                self.publisher.publish(odom)
                break  # Found it, no need to check the rest of the array


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
