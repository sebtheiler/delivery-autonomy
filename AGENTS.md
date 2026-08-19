# Notes for agents

## Code is tangled from org, not written here

Most of `src/autonomy` and both `.action` files are generated. Edit the org block, re-tangle, never edit the `.py`.

| Source | Tangles to |
|---|---|
| `~/org/roam/20260109142710-delivery.org` | `src/autonomy/**`, `src/shared_types/action/*` |
| `~/org/roam/20260222072526-extended_kalman_filter.org` | `autonomy/algorithms/ekf.py` |
| `~/org/roam/20260314170540-model_predictive_path_integral.org` | `autonomy/algorithms/mppi.py` |

`src/simulation` is **not** literate — edit it directly.

Before re-tangling, tangle to a scratch dir and diff against the repo. Org and repo can drift, and tangling silently overwrites repo-side edits:

```sh
sed 's|~/code/delivery/|/tmp/tc/|g' <note>.org > /tmp/tc/doc.org   # mkdir the tree first
emacs --batch -Q --eval '(progn (require (quote org)) (org-babel-tangle-file "/tmp/tc/doc.org"))'
```

## pkill/pgrep kill the calling shell

The tool wrapper's command line contains the pattern, so `pkill -f ground_truth_node` matches its own shell and dies with exit 144. Never combine a `pkill` with the command it's meant to clean up. Resolve PIDs first, then kill by number.

## Gazebo

Not packaged for Nix, and not fixable — nixpkgs has no `gazebo`/`gz-sim`, and Humble has zero `gz-*-vendor` packages (those start at Jazzy). It runs in the `gazebo-fortress` distrobox container (Ubuntu 22.04, Ignition Fortress 6.18).

Creating it, once:

```sh
distrobox create -n gazebo-fortress --image ubuntu:22.04 --yes
distrobox enter gazebo-fortress
sudo apt update && sudo apt install -y curl gnupg lsb-release software-properties-common mesa-utils
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install -y ros-humble-ros-gz \
  ros-humble-ros2topic ros-humble-ros2run ros-humble-ros2node \
  ros-humble-ros2service ros-humble-ros2launch ros-humble-ros2param
```

`ros-humble-ros-gz` pulls Ignition Fortress but is a thin metapackage — the `ros2` CLI verbs are the separate packages above, and are easy to miss.

- **`DISPLAY=:0` is required.** `--headless-rendering` segfaults in OgreNext during material parsing. With a display, ogre2 renders fine on the Intel iGPU.
- `ros2 topic list` returning *zero* topics while the stack is plainly running means a stale ROS daemon, not a dead stack. `ros2 daemon stop` clears it.
- `distrobox --nvidia` reports success but does nothing on NixOS — it expects driver libs at FHS paths. Mesa on the iGPU needs nothing mounted; don't chase the discrete GPU.
- `ros-humble-ros-gz` is a thin metapackage. The `ros2` CLI verbs need `ros-humble-ros2{topic,run,node,service,launch,param}` installed separately.
- ROS `setup.bash` references unbound variables, so `set -u` breaks it. Wrap the source in `set +u` / `set -u`.
- Gazebo and the autonomy stack talk over DDS across the container boundary. This works with no configuration.

Split: container runs Gazebo + `ros_gz_bridge` (`src/simulation/scripts/run_simulator.sh`); the Nix devshell runs everything else (`simulation.launch.py`).

## Simulator behaviour

- **Use `HEADLESS=1` for anything scripted.** The GUI shares a process with the server, and Qt segfaults on teardown when an entity is removed — taking the whole simulation with it.
- Reset the robot with `remove` then `create`, not `set_pose`. Teleporting leaves the physics engine with stale contact state, so the robot ends up embedded in the terrain and silently refuses every command, which looks exactly like a broken control path.
- Spawn needs ground clearance. `z: 5.0` settles correctly; `z: 2.0` drops it through the terrain mesh.
- Driving straight from spawn wedges the robot into a building after ~76 m. Keep test drives short.
- Measure yaw rate and speed as total swept angle / distance over a settled window. Per-sample `np.gradient` over a window that still contains transients reports gains ~25% high — the simulator actually tracks commanded yaw rate to within 0.5%.

## The map->odom transform is a hardcoded calibration

`autonomy.launch.py` publishes a fixed `map`->`odom` transform, and `state_estimation` sets its origin at the **first GPS fix it receives**. The two only agree if the robot is sitting at the spawn point when state estimation starts. After driving the robot around, restart the stack *and* respawn the robot, or every downstream frame is offset by however far it moved.
- The bridge **drops Ignition's header stamp** converting `Pose_V` to `TFMessage`, so poses arrive stamped `0`. `ground_truth_node` must use the node clock, and therefore must run with `use_sim_time` or every velocity is scaled by the real time factor.

## Two different wheelbases, deliberately

`dynamics.WHEELBASE = 0.615` is the prediction model, inflated to absorb tire slip. `controller.steering_wheel_base = 0.5` converts a steering angle into the yaw rate the Ackermann plugin decodes, so it must track `<wheel_base>` in `robot.urdf.xacro`. Unifying them understeers every command.

## Background build exit codes lie

Long `nix develop` / `colcon` runs launched in the background can report exit 0 while their log ends in `error: interrupted by the user`. Check the log, not the status.
