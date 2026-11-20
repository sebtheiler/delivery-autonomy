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

    # The odom frame origin is the robot's first GPS fix, so map->odom must be
    # where the robot is at that fix. run_simulator.sh asks for (-95, 25.5, 5);
    # the robot drops onto the terrain and settles 0.37 m east, at
    # (-94.630, 25.504) with base_link 2.82 m up (Gazebo ground truth, the same
    # after every remove + create). This is that settled pose, so start state
    # estimation once the robot has settled: a datum taken during the drop puts
    # every downstream frame up to 0.37 m off. Keep it in sync with the spawn
    # request in run_simulator.sh. delivery-robo's autonomy_real.launch.py
    # sim:=true has no such race: its gps_datum_tf derives map->odom from the
    # same fix the EKF takes first.
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_broadcaster',
        arguments=['--x', '-94.630', '--y', '25.504', '--z', '2.82', '--roll', '0', '--pitch', '0', '--yaw', '0', '--frame-id', 'map', '--child-frame-id', 'odom'],
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
