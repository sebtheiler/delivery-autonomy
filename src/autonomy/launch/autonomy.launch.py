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

    # The odom frame origin is the robot's first GPS fix, i.e. the spawn pose in
    # run_simulator.sh. Keep this in sync with that script or every downstream
    # frame inherits the difference as a constant lateral offset.
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_broadcaster',
        arguments=['--x', '-95.0', '--y', '25.5', '--z', '2', '--roll', '0', '--pitch', '0', '--yaw', '0', '--frame-id', 'map', '--child-frame-id', 'odom'],
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

    # Controller Action Server
    controller_node = Node(
        package='autonomy', 
        executable='controller', 
        name='controller_server',
        output='screen',
        parameters=[sim_time_param]
    )

    return LaunchDescription([
        use_sim_time_arg,
        state_estimation_node,
        static_tf_node,
        path_planning_node,
        controller_node,
    ])
