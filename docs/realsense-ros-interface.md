# D435i standard ROS 2 interface

Use the official `realsense2_camera` driver to publish calibrated images and XYZ
point clouds. `delivery_depth` adds a small consumer, `depth_cloud_processor`,
that parses the driver's standard PointCloud2 with `sensor_msgs_py` and publishes
finite, range-filtered XYZ points. No custom sensor format or camera driver.

## Install, build, launch (Ubuntu 24.04 / ROS Jazzy)

```bash
sudo apt install ros-jazzy-realsense2-camera ros-jazzy-sensor-msgs-py \
  ros-jazzy-rosbag2 ros-jazzy-rosbag2-storage-mcap python3-colcon-common-extensions
source /opt/ros/jazzy/setup.bash
# From the repository root:
colcon build --base-paths src/delivery_depth --packages-select delivery_depth
source install/setup.bash
ros2 launch delivery_depth realsense_delivery.launch.py
```

Stop any existing camera launch before starting this one: only one process should
own the camera. On Windows, first forward USB into WSL with usbipd. Linux on the
robot does not require USB forwarding. IMU streams are intentionally not enabled.
If a correctly configured driver already runs, use `start_camera:=false`.
For geometry-only capture use `enable_color:=false`; enable optional color-aligned
depth with `align_depth:=true` while color is enabled.

## Published interface

| Topic | Type | Meaning |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | RGB image (when enabled) |
| `/camera/camera/depth/image_rect_raw` | `sensor_msgs/msg/Image` | Depth image; RealSense 16UC1 millimeters |
| `/camera/camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | Depth calibration |
| `/camera/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Color calibration |
| `/camera/camera/depth/color/points` | `sensor_msgs/msg/PointCloud2` | Driver-generated XYZ, meters; no color texture required |
| `/delivery/depth/points` | `sensor_msgs/msg/PointCloud2` | Filtered XYZ, meters, original acquisition stamp/frame |
| `/delivery/depth/status` | `diagnostic_msgs/msg/DiagnosticArray` | OK, empty, malformed, missing, or stale input |

The `/color/points` name is the driver's naming convention even for untextured
clouds. This launch explicitly disables point-cloud texture requirements so a
stalled RGB stream need not prevent geometry generation. The filtered output
discards RGB fields, samples every fourth point, and keeps finite points with
optical Z in [0.2, 4.0] meters. It is unorganized (height 1).

Startup-only processor parameters: `input_topic`, `min_depth_m`, `max_depth_m`,
`sample_stride`, `stale_after_sec`. To customize, run the driver separately and:

```bash
ros2 run delivery_depth cloud_processor --ros-args -p sample_stride:=2
```

Cloud pub/sub uses sensor-data QoS (best effort, volatile). Configure consumers
accordingly. Diagnostics use reliable QoS. Empty clouds mean **unknown/no valid
returns**, not free space. Malformed input is not republished. Missing/stale
input produces diagnostics, not fabricated clouds. Consumers must enforce their
own acquisition-time freshness limit; diagnostics measure time since receipt.

## Inspect and parse

```bash
ros2 node info /depth_cloud_processor
ros2 topic hz /delivery/depth/points
ros2 topic echo /delivery/depth/status
```

In a Python subscriber callback receiving PointCloud2:

```python
from sensor_msgs_py import point_cloud2
xyz = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
# xyz contains X/Y/Z in meters in msg.header.frame_id.
```

RViz can display `/delivery/depth/points` as PointCloud2 with Best Effort
reliability. Initially use the message's optical frame as Fixed Frame.

## Record and replay: MCAP, not CSV or JSON for full frames

Run from a directory for recordings, outside the source repository:

```bash
ros2 bag record -s mcap -o d435i_demo --topics \
  /camera/camera/color/image_raw /camera/camera/color/camera_info \
  /camera/camera/depth/image_rect_raw /camera/camera/depth/camera_info \
  /camera/camera/depth/color/points /delivery/depth/points \
  /delivery/depth/status /tf /tf_static
```

Ctrl+C finishes the recording. Inspect with `ros2 bag info d435i_demo` and replay
with `ros2 bag play d435i_demo`. For replay inspection, stop live publishers first.
To rerun processing, replay only the raw cloud and run `cloud_processor` separately
(do not also replay its recorded output). Record short, consent-appropriate clips;
RGB can contain identifiable people and full-frame bags grow rapidly.

## Tests and limitations

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="src/delivery_depth:$PYTHONPATH" python3 -m unittest discover -s src/delivery_depth/test -v
```

Tests cover depth decoding, XYZ filtering, real ROS pub/sub, preservation of
timestamp/frame, and stale diagnostics. The earlier `processor` executable and
its JSON/scalar center-distance topics remain a demo, not the primary interface.

This is sensor integration, not object recognition or validated collision
avoidance. Ground points are not removed. Before navigation use, calibrate the
camera-to-base TF, validate timing/outdoor performance, and agree on ground
filtering and observation timeout behavior with the navigation team. No navigation
configuration or motor commands are changed. On the development Windows/WSL
setup, USB incomplete-frame errors and invalid depth have been observed.
