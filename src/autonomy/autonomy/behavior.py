import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from shared_types.action import ComputePath, FollowPath

class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')
        self.planner_client = ActionClient(self, ComputePath, '/compute_shortest_path')
        self.controller_client = ActionClient(self, FollowPath, '/follow_path')

    def plan_and_move(self, target_node_id):
        # 1. Block and wait for the path
        self.get_logger().info('Connecting to planner...')
        self.planner_client.wait_for_server()

        goal_msg = ComputePath.Goal(goal_node_id=target_node_id)
        
        send_goal_future = self.planner_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected.')
            return

        self.get_logger().info('Computing path...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        computed_path = result_future.result().result.path

        # 2. Fire the path off to the controller
        self.get_logger().info('Path received! Connecting to controller...')
        self.controller_client.wait_for_server()

        follow_msg = FollowPath.Goal(path=computed_path)
        
        self.controller_client.send_goal_async(follow_msg)
        self.get_logger().info('Execution command sent.')

def main(args=None):
    # Parse the command line argument
    if len(sys.argv) < 2:
        print("Usage: python3 quick_start.py <target_node_id>")
        sys.exit(1)
        
    try:
        target_id = int(sys.argv[1])
    except ValueError:
        print("Error: target_node_id must be an integer.")
        sys.exit(1)

    rclpy.init(args=args)
    node = BehaviorNode()
    
    # Pass the argument into the sequence
    node.plan_and_move(target_id)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
