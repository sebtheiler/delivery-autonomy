"""When the controller ends a goal: inside the goal tolerances, or at the end of the path."""
import math
from typing import NamedTuple

from autonomy.utils.costmap_speed import path_progress

# m of arc left that counts as the end of the path. path_progress clamps its projection to the
# path, so abeam of the last point or past it the arc left is 0
AT_PATH_END = 1e-3


class GoalEnd(NamedTuple):
    reached: bool       # the goal succeeds (True) or is aborted (False)
    at_path_end: bool   # it ended because the robot reached the end of the path
    distance: float     # m from the robot to the last point of the path
    yaw_error: float    # rad between the robot's heading and the path's at its last point

    def message(self):
        """The line the controller logs."""
        heading = f'heading off by {math.degrees(self.yaw_error):.1f} deg'
        if not self.at_path_end:
            return f'Goal reached {self.distance:.2f} m from the goal, {heading}: stopping'
        if self.reached:
            return (f'Reached the end of the path {self.distance:.2f} m from the goal, {heading}: '
                    f'stopping, goal reached')
        return (f'Passed the end of the path {self.distance:.2f} m from the goal, {heading}: '
                f'stopping, goal aborted')


def goal_end(state, path_xy, path_theta, path_s, start_idx, xy_tolerance, yaw_tolerance):
    """How the goal ends with the robot at `state` (x, y, yaw first), or None while it goes on.

    Within xy_tolerance of the last point of the path, heading within yaw_tolerance of the
    path's there (any heading when yaw_tolerance >= pi): reached. Abeam of the last point or
    past it: reached within xy_tolerance whatever the heading, aborted beyond it. Like
    path_progress, the robot is looked for along the path from `start_idx` on."""
    x, y, yaw = (float(value) for value in state[:3])
    goal_x, goal_y = path_xy[-1]
    distance = math.hypot(goal_x - x, goal_y - y)
    yaw_error = abs((yaw - float(path_theta[-1]) + math.pi) % (2 * math.pi) - math.pi)
    if distance < xy_tolerance and (yaw_tolerance >= math.pi or yaw_error < yaw_tolerance):
        return GoalEnd(True, False, distance, yaw_error)
    if len(path_xy) >= 2 and path_s[-1] - path_progress(state, path_xy, path_s, start_idx) <= AT_PATH_END:
        return GoalEnd(distance <= xy_tolerance, True, distance, yaw_error)
    return None
