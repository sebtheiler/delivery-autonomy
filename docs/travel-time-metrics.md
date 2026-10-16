# Travel time measurements

## What this branch contains

- The earlier ETA/CSV edits in the Behavior Org block.
- `tools/demo_metrics.py`: synthetic route comparisons. It generates 7,008,000
  observations; these are invented data, not campus measurements.
- `src/delivery_metrics`: a standalone ROS 2 recorder with SQLite storage and
  completed-record JSON publication. This package is not generated from Org.

The earlier Behavior edits still need Org tangling before their generated
Python changes. They currently use wall elapsed time, not simulation time.
The standalone recorder uses the node clock (including simulation time when
configured), and can be built and used independently of the Behavior changes.

## Build and run on the ROS machine

From the repository root, in the team's ROS 2/Nix environment:

```bash
nix develop
colcon build --packages-select delivery_metrics
source install/setup.bash
ros2 topic info /odom -v
ros2 topic echo /odom --once
```

Confirm `/odom` is `nav_msgs/msg/Odometry` and its longitudinal velocity is in
meters/second. This repo's simulation and state estimator use that interface;
the real robot's encoder driver/topic still needs verification. If the device
only publishes encoder ticks, it needs a driver that converts ticks using wheel
circumference, ticks/revolution, and gearing. Do not guess those values.

```bash
ros2 run delivery_metrics recorder --ros-args \
  -p odom_topic:=/odom \
  -p robot_id:=delivery_01 \
  -p route_id:=trial_A_B \
  -p database_path:=delivery_metrics.sqlite3
```

For Gazebo, add `-p use_sim_time:=true`. The sensor header timestamps must use
the same clock. System UTC timestamps label real-world runs; do not use host
UTC as the simulated campus time of day. Clock resets invalidate active trials.

In another sourced terminal, listen before finishing to see a published record:

```bash
ros2 topic echo /delivery/traversal_metrics
```

Start one traversal, operate the robot using the existing controls, then finish:

```bash
ros2 service call /travel_time_recorder/start std_srvs/srv/Trigger '{}'
ros2 service call /travel_time_recorder/finish std_srvs/srv/Trigger '{}'
```

For a failed/abandoned trial, use `/travel_time_recorder/cancel` instead.
These service calls mark boundaries explicitly; finish does NOT prove arrival.
This node never issues motor commands. No automatic Behavior integration yet.

## What is measured

Each traversal has a unique ID, robot/route labels, UTC start/end, node-clock
elapsed duration, and integrated absolute longitudinal wheel speed. Reverse
motion contributes positive distance. Stationary time remains part of elapsed
time. Moving/stopped classification uses a default 0.05 m/s threshold.

Missing data intervals over 0.5 seconds and unsampled start/end boundaries are
reported as `unobserved_seconds`, never invented as stopped time. Reverse wheel
slip and speed interpolation can still affect distance accuracy. Completed
traversals require at least 95% odometry coverage to get
`usable_for_route_learning=true`; this is a basic quality filter, not proof of
correct map matching or arrival. Out-of-order/non-finite samples are rejected.

SQLite `records` stores `record_id`, JSON `payload`, and `uploaded` (initially 0).
Persistence happens before publishing `std_msgs/msg/String` JSON. ROS topics
are live transport, not cloud upload or a durable queue. Incomplete records
survive crashes and are marked interrupted when the recorder next starts.
Run only one recorder against a database. A storage error should be investigated
before continuing trials; this prototype does not hide disk failures.

## Next integration steps

1. Verify the hardware odometry source; run short known-distance trials.
2. Connect start/finish/cancel to accepted navigation goals and real terminal
   status, replacing manual trial boundaries.
3. Preserve ordered `(from_node, to_node, edge_key)` IDs when the planner builds
   its path. Its current ComputePath action returns a dense path, not edge IDs.
4. Add route-progress matching using fused position in the map frame. Start a
   traversal on entry and finish it on exit with hysteresis; do not infer an
   edge from encoders alone. The current recorder accepts optional string
   parameters `from_node`, `to_node`, `edge_key` but does no automatic matching.
5. Add a separate uploader once the team chooses an endpoint/auth scheme. Send
   stored completed records with record_id as an idempotency key, retry failures,
   and mark uploaded only after server acknowledgement. No network upload is
   configured in this branch.
6. Aggregate real, completed, well-covered traversals by directed edge and local
   half-hour entry time. Handle timezone/DST and use distance estimates where
   observations are sparse. Do not mix synthetic or canceled runs into training.

## Checks and branch workflow

Pure Python checks (no ROS needed):

```bash
PYTHONPATH=src/delivery_metrics python -m unittest discover -s src/delivery_metrics/test -v
```

ROS build and live topic/service testing must run in the team's ROS environment.
Org regeneration, when Emacs is available:

```bash
emacs --batch -Q --eval '(progn (require (quote org)) (org-babel-tangle-file "docs/20260109142710-delivery.org"))'
git diff -- src/
```

Work is on `feature/travel-time-metrics`. Review with `git status` and `git diff`.
After committing, publish it yourself with:

```bash
git push -u origin feature/travel-time-metrics
```

That requires write access to sebtheiler/delivery-autonomy; otherwise push to
your fork and open a pull request. Publishing is separate from a local commit.
