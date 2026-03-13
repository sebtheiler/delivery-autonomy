import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
import xacro
from launch_ros.actions import Node


def generate_launch_description():
    pkg_name = "simulation"
    pkg_share = get_package_share_directory(pkg_name)

    # Configuration Variables
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    world_file = LaunchConfiguration("world", default="washu.world")

    # Path to the world file
    world_path = PathJoinSubstitution([pkg_share, "worlds", world_file])

    # We construct the path manually to ensure it includes the 'models' subdir
    model_path = os.path.join(pkg_share, "models")

    set_res_path = SetEnvironmentVariable(
        name="IGN_GAZEBO_RESOURCE_PATH",
        value=[model_path, ":", os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")],
    )

    # Run Ignition Gazebo directly via ExecuteProcess
    run_ign_gazebo = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-v", "4", world_path],
        output="screen",
    )

    # ROS-Gazebo Bridge
    config_file_path = os.path.join(pkg_share, "config", "bridge.yaml")
    print(config_file_path)
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[
            {
                "use_sim_time": True,
                "config_file": config_file_path,
            }
        ],
        output="screen",
    )

    # Spawn The Robot
    xacro_file = os.path.join(pkg_share, "urdf", "robot.urdf.xacro")
    doc = xacro.process_file(xacro_file)
    robot_desc = doc.toxml()

    # Robot State Publisher (RSP)
    # This takes the URDF string and publishes the /tf tree based on /joint_states
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_desc}],
    )

    # TODO: automatically process to SDF
    sdf_path = os.path.join(pkg_share, "urdf", "robot.sdf")

    spawn_cmd = f"""ign service -s /world/washu_campus/create \
    --reqtype ignition.msgs.EntityFactory \
    --reptype ignition.msgs.Boolean \
    --timeout 1000 \
    --req 'sdf_filename: "{sdf_path}", name: "delivery_robot", pose: {{position: {{x: -80, y: 20, z: 10.0}}}}'"""

    spawn_robot = ExecuteProcess(
        cmd=["bash", "-c", spawn_cmd],
        output="screen",
    )

    # RViz
    rviz_config_file = os.path.join(pkg_share, "config", "view.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=['-d', rviz_config_file],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            set_res_path,
            run_ign_gazebo,
            bridge,
            spawn_robot,
            Node(
                package="simulation",
                executable="ground_truth_node",
            ),
            rsp,
            rviz,
        ]
    )
