import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from shared_types.action import FollowPath
from autonomy.utils.dynamics import INPUT_DIMENSION
from autonomy.utils.reference_trajectory import process_path, get_local_reference
from autonomy.utils.mppi_step import mppi_step
from autonomy.utils.dynamics import dynamics_model
import time
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import jax
import jax.numpy as jnp
import numpy as np

class ControllerActionServer(Node):
    def __init__(self):
        super().__init__('controller_server')

        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self.state_cb_group = MutuallyExclusiveCallbackGroup()

        self._action_server = ActionServer(
            self,
            FollowPath,
            '/follow_path',
            self.execute_callback,
            callback_group=self.action_cb_group,
        )


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter("state_topic", "/state_estimation/odom")
        state_topic = self.get_parameter("state_topic").value
        
        self.current_state = None
        self.state_subscription = self.create_subscription(
            Odometry,
            state_topic,
            self.process_state_callback,
            10,
            callback_group=self.state_cb_group,
        )

        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel", 10
        )

        self.T = 50 # TODO: make param
        self.steering_wheel_base = 0.5  # Must match <wheel_base> in robot.urdf.xacro
        self.dt = 0.1
        self.rng = jax.random.PRNGKey(42)
        self.U_nominal = jnp.zeros((self.T, INPUT_DIMENSION))
        
        self.active_goal = False
        self.current_path = None
        self.control_timer = self.create_timer(self.dt, self.control_loop_callback)
        
        self.progress_idx = 0
        self.last_u = None

        self.get_logger().info('Controller Action Server initialized.')

    def process_state_callback(self, odom_msg: Odometry):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                odom_msg.header.stamp, # Sync with the velocity measurement
                rclpy.duration.Duration(seconds=0.1) # Small timeout
            )
        except TransformException as ex:
            self.get_logger().warning(f'Could not transform map to base_link: {ex}')
            return None
    
        # Extract Position in Map frame
        x = t.transform.translation.x
        y = t.transform.translation.y
        
        # Extract Yaw (theta) in Map frame
        q = t.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        theta = np.arctan2(siny_cosp, cosy_cosp)
        
        v = odom_msg.twist.twist.linear.x
        omega = odom_msg.twist.twist.angular.z
    
        # TODO: get this somehow
        b_omega = 0
    
        # self.get_logger().info(f"({x}, {y}) @ {theta}")
    
        self.current_state = np.array([x, y, theta, v, omega, b_omega])

    def publish_command(self, u):
        v_cmd, delta = u
    
        v_cmd_float = float(v_cmd)
        delta_float = float(delta)
    
        V_LIMIT = 1.5 # TODO: make param
        max_steer = 0.6
        v_cmd_float = max(min(v_cmd_float, V_LIMIT), -V_LIMIT)
        delta_clipped = np.clip(delta_float, -max_steer, max_steer)
        omega = (v_cmd_float / self.steering_wheel_base) * np.tan(delta_clipped)
    
        cmd_vel = Twist()
    
        cmd_vel.linear.x = v_cmd_float
        cmd_vel.angular.z = omega
    
        self.cmd_pub.publish(cmd_vel)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Received a new path to follow!')
    
        path = process_path(goal_handle.request.path)
        self.current_path = path
        self.progress_idx = 0
        self.active_goal = True
    
        # Block this callback until the timer thread marks it done
        while self.active_goal and rclpy.ok():
            time.sleep(0.05)
    
        goal_handle.succeed()
        return FollowPath.Result(success=True)

    def control_loop_callback(self):
        if not self.active_goal or self.current_state is None:
            return
    
        path_xy, path_theta_unwrapped, path_s = self.current_path
    
        if len(path_xy) == 0:
            self.get_logger().warning("Received empty path. Stopping robot.")
            self.publish_command(jnp.zeros(INPUT_DIMENSION)) 
            self.active_goal = False
            return
    
        # doesn't have to be calculated every step but minimal overhead
        goal_x, goal_y = path_xy[-1]
        goal_theta = path_theta_unwrapped[-1]
    
        # TODO: make params
        xy_goal_tolerance = 0.25  # meters
        yaw_goal_tolerance = 0.15 # radians
        V_TARGET = 1.5
    
        # Assuming state is [x, y, theta, v, omega, b_omega]
        curr_x, curr_y = self.current_state[0], self.current_state[1]
        curr_theta = self.current_state[2]
        
        # 3. Calculate errors
        dist_to_goal = np.hypot(goal_x - curr_x, goal_y - curr_y)
        yaw_err = abs((curr_theta - goal_theta + np.pi) % (2 * np.pi) - np.pi)
        
        # 4. Check termination condition
        if dist_to_goal < xy_goal_tolerance and yaw_err < yaw_goal_tolerance:
            # Stop the motors
            self.publish_command(np.zeros(INPUT_DIMENSION)) 
        
            self.active_goal = False
            return
    
        # Optimise from where the robot will be once this solve finishes, not where
        # it was when the solve started
        future_state = dynamics_model(self.current_state, self.last_u, dt=self.dt) if self.last_u is not None else self.current_state
    
        ref_traj_np, self.progress_idx = get_local_reference(
            future_state,
            path_xy,
            path_theta_unwrapped,
            path_s,
            self.T,
            self.dt,
            v_target=V_TARGET,
            progress_idx=self.progress_idx,
        )
        ref_traj_jax = jnp.array(ref_traj_np)
    
        self.rng, iter_key = jax.random.split(self.rng)
    
        U_opt = mppi_step(
            key=iter_key,
            x0=future_state,
            U_nominal=self.U_nominal,
            ref_traj=ref_traj_jax,
            v_target=V_TARGET,
        )
    
        # TODO: make params
        # [min_v, min_delta]
        lower_bounds = jnp.array([-0.5, -0.6])
        upper_bounds = jnp.array([1.5, 0.6])
        U_opt = jnp.clip(U_opt, lower_bounds, upper_bounds)
    
        u = U_opt[0]
        self.publish_command(u)
        self.last_u = u
    
        # Warm start the next solve by shifting the horizon forward one step
        self.U_nominal = jnp.roll(U_opt, shift=-1, axis=0)
        self.U_nominal = self.U_nominal.at[-1].set(U_opt[-1])

def main(args=None):
    rclpy.init(args=args)
    node = ControllerActionServer()

    # Use a multi-threaded executor so callbacks can run in parallel
    executor = MultiThreadedExecutor()
    
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
