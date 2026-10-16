# D435i depth processing node

For the standard PointCloud2 integration and MCAP recording, see
[realsense-ros-interface.md](realsense-ros-interface.md). This page describes
the optional center-depth demo, not the primary navigation-facing interface.

Data flow: D435i → RealSense ROS driver → depth image topic → `depth_processor` → measurement topics.

This is a standalone, hand-written `ament_python` package; it does not change the Org-generated autonomy code. It observes images only and does not command the robot or change routes.

## Build and run (Ubuntu / WSL, ROS Jazzy)

From the repository root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths src/delivery_depth --packages-select delivery_depth
source install/setup.bash
ros2 run delivery_depth processor
```

The RealSense driver must already be running. Default input is `/camera/camera/depth/image_rect_raw` (`sensor_msgs/msg/Image`, sensor-data QoS), independent of the color stream. To use color-aligned depth instead, enable alignment in the driver and run:

```bash
ros2 run delivery_depth processor --ros-args -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw
```

## Outputs

- `/delivery/depth_metrics` (`std_msgs/msg/String`): JSON status, median and nearest valid depth in meters, valid pixel fraction, center ROI coordinates, original image timestamp and optical frame ID.
- `/delivery/center_depth_m` (`std_msgs/msg/Float32`): median depth in meters; NaN when unavailable. Consumers that need timestamps and status should use the JSON topic.

Default ROI is a centered 40×40 pixel square, clipped to the image. At least 50% of its pixels must contain positive finite depth. Zeros, negative values, NaN and infinity are excluded. `16UC1` is interpreted as millimeters (RealSense convention); `32FC1` as meters. Other encodings are rejected. Row padding and image endianness are respected.

Parameters: `depth_topic`, `roi_size` (40), `min_valid_fraction` (0.5), `stale_after_sec` (2.0). Set them at startup. Status is `ok`, `insufficient_depth`, `invalid_image`, `no_frames`, or `stale`. Missing/stale status is published every 0.5 seconds, with null depths in JSON and NaN on the scalar topic. Staleness measures time since receipt, not transport latency from acquisition.

Depth means distance along the optical Z axis, not Euclidean range. The nearest value is only within the center patch, may include noise, and is not the closest obstacle anywhere in view. This prototype is not collision avoidance or a safety system. It does not persist data, upload to a server, or feed route-time recommendations.

## Demonstrate communication

In another sourced ROS terminal:

```bash
ros2 node info /depth_processor
ros2 topic echo /delivery/depth_metrics
# Or watch just meters:
ros2 topic echo /delivery/center_depth_m
```

Move an object in front of the image center; the published depth should change. Use Ctrl+C to stop the echo. To record locally, if rosbag2 is installed:

```bash
ros2 bag record /delivery/depth_metrics /delivery/center_depth_m
```

Core tests (from repository root):

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="src/delivery_depth:$PYTHONPATH" python3 -m unittest discover -s src/delivery_depth/test -v
```
