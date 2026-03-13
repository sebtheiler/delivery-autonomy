import rclpy
from rclpy.node import Node
from autonomy.utils.sensors import MEASUREMENT_FCNS
from autonomy.utils.dynamics import STATE_DIMENSION, INPUT_DIMENSION, WHEELBASE
from autonomy.utils.rpy_to_quaternion import rpy_to_quaternion
from autonomy.utils.latlon2meters import latlon2meters 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, NavSatFix
from seblib.ekf import MultiRateEKF
import jax.numpy as jnp
import math

class DeliveryStateEstimationNode(Node):
    def __init__(self):
        super().__init__("delivery_state_estimation")

        self.get_logger().info("State estimation node initialized.")

        self.ekf = None
        self.received_first_gps = False
        self.origin_lat = 0.0
        self.origin_lon = 0.0

        self.declare_parameter("imu_topic", "/imu/data")
        imu_topic = self.get_parameter("imu_topic").value
        
        self.imu = None
        self.imu_subscription = self.create_subscription(
            Imu, imu_topic, self.process_imu_callback, 10
        )

        self.declare_parameter("odom_topic", "/odom")
        odom_topic = self.get_parameter("odom_topic").value
        
        self.odom = None
        self.odom_subscription = self.create_subscription(
            Odometry, odom_topic, self.process_odom_callback, 10
        )

        self.declare_parameter("gps_topic", "/gps/fix")
        gps_topic = self.get_parameter("gps_topic").value
        
        self.gps = None
        self.gps_subscription = self.create_subscription(
            NavSatFix, gps_topic, self.process_gps_callback, 10
        )

        self.current_u = jnp.zeros(INPUT_DIMENSION, dtype=jnp.float32)
        
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        
        self.cmd_vel_subscription = self.create_subscription(
            Twist, cmd_vel_topic, self.process_cmd_vel_callback, 10
        )

        self.declare_parameter("desired_hz", 100)
        self.desired_hz = self.get_parameter("desired_hz").value
        
        self.loop_timer = self.create_timer(1/self.desired_hz, self.main_loop)
        
        self.state_pub = self.create_publisher(
            Odometry, "/state_estimation/odom", 10
        )

    def process_imu_callback(self, msg: Imu):
        if not self.received_first_gps:
            return
    
        omega_z = msg.angular_velocity.z
        z = jnp.array([omega_z]) 
        timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
    
        self.ekf.receive_measurement('imu', z, self.current_u, timestamp)

    def process_odom_callback(self, msg: Odometry):
        if not self.received_first_gps:
            return
    
        v_meas = msg.twist.twist.linear.x
        omega_meas = msg.twist.twist.angular.z
        z = jnp.array([v_meas, omega_meas])
        
        timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        self.ekf.receive_measurement('odom', z, self.current_u, timestamp)

    def process_gps_callback(self, msg: NavSatFix):
        lat_rad = math.radians(msg.latitude)
        lon_rad = math.radians(msg.longitude)
        timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
    
        if not self.received_first_gps:
            self.get_logger().info('Setting GPS datum and initializing state to (0,0)')
            self.origin_lat = lat_rad
            self.origin_lon = lon_rad
            self.received_first_gps = True
    
            self.ekf = MultiRateEKF(
                jnp.zeros(STATE_DIMENSION, dtype=jnp.float32), 
                jnp.eye(STATE_DIMENSION, dtype=jnp.float32) * 10, 
                MEASUREMENT_FCNS, 
                timestamp
            )
    
            return
    
        x, y = latlon2meters(lon_rad, lat_rad, self.origin_lon, self.origin_lat)
    
        z = jnp.array([x, y], dtype=jnp.float32)
        u = self.current_u
        self.ekf.receive_measurement('gps', z, u, timestamp)

    def process_cmd_vel_callback(self, msg: Twist):
        v_cmd = msg.linear.x
        omega_cmd = msg.angular.z
    
        # Prevent division by zero when the robot is stopped
        if abs(v_cmd) < 1e-3:
            delta = 0.0
        else:
            # Calculate steering angle required to achieve the commanded yaw rate
            delta = math.atan((omega_cmd * WHEELBASE) / v_cmd)
    
        self.current_u = jnp.array([v_cmd, delta], dtype=jnp.float32)

    def main_loop(self):
        if self.ekf is None:
            return
    
        x = self.ekf.x
        odom = Odometry()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        # print(x)
        # print(len(self.ekf.history))
        print(f"{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}, {x[3]:.2f}")
        odom.pose.pose.position.x = x[0].item()
        odom.pose.pose.position.y = x[1].item()
        odom.pose.pose.orientation = rpy_to_quaternion(0, 0, x[2].item())
        odom.twist.twist.linear.x = x[3].item()
    
        self.state_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    state_estimation_node = DeliveryStateEstimationNode()
    rclpy.spin(state_estimation_node)

    state_estimation_node.destroy_node()
    rclpy.shutdown()
