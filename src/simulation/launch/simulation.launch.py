import os
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
import xacro
from launch_ros.actions import Node
from ros_ign_gazebo.parameters.rviz import create_sdf_from_urdf


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

    # Process the URDF and convert to SDF
    robot_desc = xacro.process_file(xacro_file).toxml()

    # Create a temporary SDF file from the URDF string
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sdf") as sdf_file:
        sdf_filename = sdf_file.name
        sdf_file.write(create_sdf_from_urdf(robot_desc))

    # Path to the temporary SDF file
    sdf_path = sdf_filename

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
        arguments=["-d", rviz_config_file],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            set_res_path,
            run_ign_gazebo,
            bridge,
            spawn_robot,
            rsp,
            rviz,
        ]
    )
