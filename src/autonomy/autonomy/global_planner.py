import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import osmnx as ox
import networkx as nx
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from shared_types.action import ComputePath 
from shapely.geometry import Point as ShapelyPoint
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

class GlobalPlanningServer(Node):
    def __init__(self):
        super().__init__('path_planning_server')
        
        # TODO: make dynamic
        self.origin_lat = 38.648000
        self.origin_lon = -90.302500
        
        # Load graph
        self.get_logger().info("Generating and projecting OSM graph...")
        self.G_projected, self.utm_origin = self._initialize_graph()
        
        # Action server
        self._action_server = ActionServer(
            self,
            ComputePath,
            'compute_shortest_path',
            self.execute_callback
        )
        self.get_logger().info("Path Planning Action Server is ready.")

        # Visualization
        self.path_viz_pub = self.create_publisher(Marker, 'planned_path_viz', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _initialize_graph(self):
        bbox = (-90.304114, 38.647310, -90.301293, 38.649011)
        pedestrian_filter = ('["highway"~"footway|path|pedestrian|track"]["steps"!="yes"]["stairs"!="yes"]')
    
        # Generate the graph
        G = ox.graph_from_bbox(bbox=bbox, custom_filter=pedestrian_filter, network_type='walk')
        self.get_logger().info(f"Sample Node IDs for testing: {list(G.nodes)[:5]}")
        G_proj = ox.project_graph(G)
    
        # Calculate the UTM coordinates of the Gazebo origin
        # origin_point = ShapelyPoint(self.origin_lon, self.origin_lat)
        # origin_proj = ox.projection.project_geometry(origin_point, to_crs=G_proj.graph['crs'])[0]
    
        # this is hardcoded because the Gazebo sim is messed up in some weird way. Not great
        true_origin_x = 734765 + 91.46309417473815
        true_origin_y = 4281212 - 24.993696585102057
    
        return G_proj, (true_origin_x, true_origin_y)

    def get_current_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
    
            x = t.transform.translation.x
            y = t.transform.translation.y
    
            self.get_logger().info(f"Robot position acquired: x={x:.2f}, y={y:.2f}")
            return x, y
    
        except TransformException as ex:
            self.get_logger().error(f"Could not get robot pose: {ex}")
            return None, None

    def execute_callback(self, goal_handle):
        self.get_logger().info(f"Received request for goal node: {goal_handle.request.goal_node_id}")
        result = ComputePath.Result()
    
        local_x, local_y = self.get_current_robot_pose()
        if local_x is None or local_y is None:
            self.get_logger().error("Aborting path planning: Robot pose is unknown.")
            goal_handle.abort()
            return result
    
        utm_x = local_x + self.utm_origin[0]
        utm_y = local_y + self.utm_origin[1]
        
        # Snap to nearest graph node
        start_node_id = ox.distance.nearest_nodes(self.G_projected, X=utm_x, Y=utm_y)
        goal_node_id = goal_handle.request.goal_node_id
        
        # Compute shortest path using edge lengths
        try:
            path_node_ids = nx.shortest_path(
                self.G_projected, 
                source=start_node_id, 
                target=goal_node_id, 
                weight='length'
            )
        except nx.NetworkXNoPath:
            self.get_logger().error("No path found between nodes.")
            goal_handle.abort()
            return ComputePath.Result()
    
        dense_path = Path()
        dense_path.header.stamp = self.get_clock().now().to_msg()
        dense_path.header.frame_id = 'map'
    
        for i in range(len(path_node_ids) - 1):
            u = path_node_ids[i]
            v = path_node_ids[i + 1]
        
            edge_data = self.G_projected.get_edge_data(u, v)[0]
        
            # Check if the edge has a curved geometry
            if 'geometry' in edge_data:
                # Extract the coordinates from the Shapely LineString
                xs, ys = edge_data['geometry'].xy
                points = list(zip(xs, ys))
            else:
                # It's a straight line, just use the start and end nodes
                points = [
                    (self.G_projected.nodes[u]['x'], self.G_projected.nodes[u]['y']),
                    (self.G_projected.nodes[v]['x'], self.G_projected.nodes[v]['y'])
                ]
        
            # Convert the points to the local metric frame and append to the path
            for pt_x, pt_y in points:
                pose = PoseStamped()
                pose.header = dense_path.header
        
                # Translate from UTM back to local Gazebo 'map' frame
                pose.pose.position.x = pt_x - self.utm_origin[0]
                pose.pose.position.y = pt_y - self.utm_origin[1]
                pose.pose.position.z = 0.0
        
                dense_path.poses.append(pose)
        
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = "spline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        for pose in dense_path.poses:
            p = Point()
            p.x = pose.pose.position.x
            p.y = pose.pose.position.y
            p.z = 0.0
            marker.points.append(p)
        
        marker.scale.x = 0.2  # Line width in meters
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        # Set lifetime to 0 (infinite) so it persists after the action server finishes
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0 
        
        self.path_viz_pub.publish(marker)
    
        goal_handle.succeed()
        result.path = dense_path
    
        return result

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanningServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
