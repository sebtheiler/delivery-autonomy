from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    sim_time_param = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    state_estimation_node = Node(
        package='autonomy',
        executable='state_estimation',
        name='state_estimation',
        output='screen',
        parameters=[sim_time_param]
    )

    # The robot boots up at global (-80, 20, 2), so the odom origin is offset by that amount
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_broadcaster',
        # Arguments: x y z yaw pitch roll parent_frame child_frame
        arguments=['-80', '20', '2', '0', '0', '0', 'map', 'odom'],
        parameters=[sim_time_param]
    )

    # Path Planning Action Server
    path_planning_node = Node(
        package='autonomy', 
        executable='global_planner', 
        name='global_planning_server',
        output='screen',
        parameters=[sim_time_param]
    )

    return LaunchDescription([
        state_estimation_node,
        static_tf_node,
        path_planning_node
    ])
