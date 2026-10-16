"""Use the vendor driver for calibrated images and XYZ point-cloud generation."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('enable_color', default_value='true'),
        DeclareLaunchArgument('align_depth', default_value='false'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')),
            condition=IfCondition(LaunchConfiguration('start_camera')),
            launch_arguments={
                'camera_namespace': 'camera', 'camera_name': 'camera',
                'enable_depth': 'true',
                'enable_color': LaunchConfiguration('enable_color'),
                'align_depth.enable': LaunchConfiguration('align_depth'),
                'depth_module.depth_profile': '640x480x15',
                'rgb_camera.color_profile': '640x480x15',
                'pointcloud.enable': 'true',
                # XYZ only: do not make geometry depend on color-frame delivery.
                'pointcloud.stream_filter': '0',
                'pointcloud.allow_no_texture_points': 'true',
            }.items()),
        Node(package='delivery_depth', executable='cloud_processor', output='screen'),
    ])
