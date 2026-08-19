import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

# Gazebo and the ROS <-> Gazebo bridge are started
# separately via src/simulation/scripts/run_simulator.sh.

def generate_launch_description():
    pkg_share = get_package_share_directory("simulation")
    sim_time = {"use_sim_time": True}

    robot_desc = xacro.process_file(
        os.path.join(pkg_share, "urdf", "robot.urdf.xacro")
    ).toxml()

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{**sim_time, "robot_description": robot_desc}],
    )

    ground_truth = Node(
        package="simulation",
        executable="ground_truth_node",
        parameters=[sim_time],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(pkg_share, "config", "view.rviz")],
        output="screen",
        parameters=[sim_time],
    )

    return LaunchDescription([rsp, ground_truth, rviz])
