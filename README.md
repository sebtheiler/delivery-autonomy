# Delivery

Demo delivery robot that drives to a given point on the east-end of WashU campus.

The robot uses Ackermann steering. It finds its position with an Extended Kalman
Filter. The filter mixes GPS, IMU, and wheel encoder data. A global planner makes
a path from OpenStreetMap sidewalk data. An MPPI controller drives the robot along
the path.

## The code comes from Org files

Most of the code in `src/autonomy` is generated. Do not edit those Python files.
Edit the Org source in `docs/`, then tangle it again. The Org files are these:

| Org file in `docs/` | It generates |
|---|---|
| `20260109142710-delivery.org` | `src/autonomy/`, `src/shared_types/action/` |
| `20260222072526-extended_kalman_filter.org` | `autonomy/algorithms/ekf.py` |
| `20260314170540-model_predictive_path_integral.org` | `autonomy/algorithms/mppi.py` |

The `src/simulation` package is not generated. Edit it directly.
	
## Requirements

The project only requires Nix with flakes. Nothing additional needs to be installed.
Support for legacy distros that do not have Nix installed (e.g., Arch, Ubuntu)
is not guaranteed, to try at your own risk.

## Set-up

### 1. Build the workspace

Open the development shell:

```sh
nix develop
```

Build the three packages:

```sh
colcon build
source install/setup.bash
```

### 2. Start the simulator

The simulator starts Gazebo and the bridge to ROS 2. Start it in the devshell:

```sh
src/simulation/scripts/run_simulator.sh
```

Set `HEADLESS=1` before the command if you do not want the Gazebo window. Do this
for automatic tests.

The simulator is ready when the log shows 12 bridges.

### 3. Start the ROS 2 nodes

In a second terminal, start the nodes that show the robot:

```sh
nix develop
source install/setup.bash
ros2 launch simulation simulation.launch.py
```

This starts RViz, the robot state publisher, and the ground truth publisher.

In a third terminal, start the autonomy stack:

```sh
nix develop
source install/setup.bash
ros2 launch autonomy autonomy.launch.py use_sim_time:=true
```

This starts state estimation, the global planner, and the controller.

The planner writes example goal node IDs to the log when it starts. The message
starts with `Sample Node IDs for testing:`. Keep one of these IDs.

### 4. Send a delivery goal

Give the robot a goal. Use one of the node IDs from step 3:

```sh
python src/autonomy/autonomy/behavior.py 1282043935
```

### 5. Read the result

The robot drives to the goal. To see the result, do one or more of these:

- Look at the Gazebo window. The robot moves along a sidewalk.
- Look at RViz. The blue line is the planned path. The robot follows this line.
- Read the robot position:

  ```sh
  ros2 topic echo /state_estimation/odom --once
  ```

- Compare the estimate with the true position:

  ```sh
  ros2 topic echo /ground_truth/odom --once
  ```

The controller stops the robot when it is close to the goal.

## Limits

- The robot must start at the spawn point. The transform from the `map` frame to
  the `odom` frame is a constant. State estimation makes its origin at the first
  GPS position it receives. If you move the robot and then start the autonomy
  stack again, all positions are wrong by the distance you moved it.
- The camera and the LiDAR make data, but no node reads that data yet.
