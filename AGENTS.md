# Notes for agents

## Code is tangled from org, not written here

Most of `src/autonomy` and both `.action` files are generated. Edit the org block, re-tangle, never edit the `.py`.

| Source | Tangles to |
|---|---|
| `docs/20260109142710-delivery.org` | `src/autonomy/**`, `src/shared_types/action/*` |
| `docs/20260222072526-extended_kalman_filter.org` | `autonomy/algorithms/ekf.py` |
| `docs/20260314170540-model_predictive_path_integral.org` | `autonomy/algorithms/mppi.py` |

`:tangle` paths are relative, and Org resolves those against `buffer-file-name`.
Each file therefore carries a `Local Variables` block that pins
`default-directory` to its truename. Without it, tangling from a symlink writes to
the wrong location silently. Emacs asks once whether to allow that `eval:`.

`src/simulation` is **not** literate — edit it directly.

Before re-tangling, tangle to a scratch dir and diff against the repo. Org and repo can drift, and tangling silently overwrites repo-side edits:

```sh
sed 's|~/code/delivery/|/tmp/tc/|g' <note>.org > /tmp/tc/doc.org   # mkdir the tree first
emacs --batch -Q --eval '(progn (require (quote org)) (org-babel-tangle-file "/tmp/tc/doc.org"))'
```

## pkill/pgrep kill the calling shell

The tool wrapper's command line contains the pattern, so `pkill -f ground_truth_node` matches its own shell and dies with exit 144. Never combine a `pkill` with the command it's meant to clean up. Resolve PIDs first, then kill by number.

## Gazebo

Comes from the flake. It ships as `gz-*-vendor` packages, which the overlay generates from rosdistro like any other ROS package. The devshell pulls `gz-sim-vendor` (Gazebo Sim, the engine) and `gz-tools-vendor` (the `gz` CLI dispatcher; without it there is no `gz` binary, only libraries).

- **`DISPLAY=:0` is required.** `--headless-rendering` segfaults in OgreNext during material parsing. With a display, ogre2 renders fine on the Intel iGPU.
- `ros2 topic list` returning *zero* topics while the stack is plainly running means a stale ROS daemon, not a dead stack. `ros2 daemon stop` clears it.
- `GZ_SIM_RESOURCE_PATH` (not `IGN_GAZEBO_RESOURCE_PATH`) is what finds `models/`. The Ignition-era names are silently ignored, so a wrong one looks like a missing mesh.
- `<gz_frame_id>` draws a "not defined in SDF" warning from the parser. That is expected — it is a custom element, so sdformat preserves it and gz-sim reads it. `/scan` really does come out stamped `lidar_link`.
- If a stale Ignition-era server is still running anywhere on the box, gz-transport logs `Unknown message type [9]` and the new server refuses the global `/clock` with "Another world of the same name is running". Both are that leftover, not a broken world — find it with `ps` and kill it by PID.

Everything runs in the one devshell now: `src/simulation/scripts/run_simulator.sh` for Gazebo + `ros_gz_bridge`, `simulation.launch.py` for the rest.

## Simulator behaviour

- **Use `HEADLESS=1` for anything scripted.** The GUI shares a process with the server, and Qt segfaults on teardown when an entity is removed — taking the whole simulation with it.
- Reset the robot with `remove` then `create`, not `set_pose`. Teleporting leaves the physics engine with stale contact state, so the robot ends up embedded in the terrain and silently refuses every command, which looks exactly like a broken control path.
- Spawn needs ground clearance. `z: 5.0` settles correctly; `z: 2.0` drops it through the terrain mesh.
- Driving straight from spawn wedges the robot into a building after ~76 m. Keep test drives short.
- Measure yaw rate and speed as total swept angle / distance over a settled window. Per-sample `np.gradient` over a window that still contains transients reports gains ~25% high — the simulator actually tracks commanded yaw rate to within 0.5%.
- The bridge **drops Gazebo's header stamp** converting `Pose_V` to `TFMessage`, so poses arrive stamped `0`. `ground_truth_node` must use the node clock, and therefore must run with `use_sim_time` or every velocity is scaled by the real time factor.

## The map->odom transform is a hardcoded calibration

`autonomy.launch.py` publishes a fixed `map`->`odom` transform, and `state_estimation` sets its origin at the **first GPS fix it receives**. The two only agree if the robot is sitting at the spawn point when state estimation starts. After driving the robot around, restart the stack *and* respawn the robot, or every downstream frame is offset by however far it moved.

## Two different wheelbases, deliberately

`dynamics.WHEELBASE = 0.615` is the prediction model, inflated to absorb tire slip. `controller.steering_wheel_base = 0.5` converts a steering angle into the yaw rate the Ackermann plugin decodes, so it must track `<wheel_base>` in `robot.urdf.xacro`. Unifying them understeers every command.
