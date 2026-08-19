#!/usr/bin/env bash
# Runs Gazebo and the ROS <-> Gazebo bridge.
#
# Ignition Fortress is not packaged for Nix, so the simulator lives in an Ubuntu
# container while the autonomy stack runs in the Nix devshell. The two halves
# talk over DDS on the shared host network. Run this from the host with:
#
#   distrobox enter gazebo-fortress -- src/simulation/scripts/run_simulator.sh
#
# then launch the ROS side separately with `ros2 launch simulation simulation.launch.py`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_SHARE="$REPO_ROOT/src/simulation"
WORLD="${WORLD:-$SIM_SHARE/worlds/washu.world}"

# Ogre2 segfaults without a display, so fall back to the host's X server
export DISPLAY="${DISPLAY:-:0}"
export IGN_GAZEBO_RESOURCE_PATH="$SIM_SHARE/models:${IGN_GAZEBO_RESOURCE_PATH:-}"

# ROS setup scripts reference unbound variables
set +u
source /opt/ros/humble/setup.bash
set -u

children=()
cleanup() {
  for pid in "${children[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# The GUI shares a process with the server, and Qt has been seen to segfault on
# teardown when an entity is removed, which takes the simulation down with it.
# Scripted runs should set HEADLESS=1.
gui_flag=""
if [ -n "${HEADLESS:-}" ]; then
  gui_flag="-s"
fi

ign gazebo -r -v 4 $gui_flag "$WORLD" &
children+=($!)

# The create service only exists once the world is up
until ign service --list 2>/dev/null | grep -q "/world/washu_campus/create"; do
  sleep 1
done

ign service -s /world/washu_campus/create \
  --reqtype ignition.msgs.EntityFactory \
  --reptype ignition.msgs.Boolean \
  --timeout 5000 \
  --req "sdf_filename: \"$SIM_SHARE/urdf/robot.sdf\", name: \"delivery_robot\", pose: {position: {x: -95, y: 25.5, z: 5.0}}"

ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p "config_file:=$SIM_SHARE/config/bridge.yaml" \
  -p use_sim_time:=true &
children+=($!)

wait
