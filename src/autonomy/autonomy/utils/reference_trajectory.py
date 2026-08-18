import numpy as np

def extract_xy_theta_from_msg(path_msg):
    if not path_msg.poses:
        return np.empty((0, 3))

    data = np.array([
        [
            p.pose.position.x,
            p.pose.position.y,
            p.pose.orientation.x,
            p.pose.orientation.y,
            p.pose.orientation.z,
            p.pose.orientation.w
        ]
        for p in path_msg.poses
    ])

    x = data[:, 0]
    y = data[:, 1]
    qx = data[:, 2]
    qy = data[:, 3]
    qz = data[:, 4]
    qw = data[:, 5]

    # Convert from quaternion to yaw (theta)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    theta = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack((x, y, theta))

# def process_path(path_msg):
#     path_data = extract_xy_theta_from_msg(path_msg)
#     print(path_data)
#     path_xy = path_data[:, :2]
#     path_theta = path_data[:, 2]
    
#     # Calculate Euclidean distance between consecutive waypoints
#     diffs = np.diff(path_xy, axis=0)
#     distances = np.linalg.norm(diffs, axis=1)
    
#     # Cumulative sum to get arc length 's' at each waypoint
#     # Insert 0.0 at the beginning for the first waypoint
#     path_s = np.insert(np.cumsum(distances), 0, 0.0)
    
#     # Unwrap theta to prevent interpolation errors across the -pi/pi boundary
#     path_theta_unwrapped = np.unwrap(path_theta)
    
#     return path_xy, path_theta_unwrapped, path_s
import numpy as np

def process_path(path_msg):
    # 1. Extract raw coordinates from the ROS message
    raw_x = [pose.pose.position.x for pose in path_msg.poses]
    raw_y = [pose.pose.position.y for pose in path_msg.poses]
    raw_xy = np.column_stack((raw_x, raw_y))

    if len(raw_xy) < 2:
        return raw_xy, np.zeros(len(raw_xy)), np.zeros(len(raw_xy))

    # 2. Enforce the 10cm Rule (Spatial Resampling)
    # Only keep a waypoint if it is at least 0.1 meters away from the previous one.
    # This destroys dense clusters and prevents arctan2 singularities.
    MIN_DIST = 0.1 
    filtered_xy = [raw_xy[0]]
    
    for pt in raw_xy[1:]:
        dist = np.linalg.norm(pt - filtered_xy[-1])
        if dist >= MIN_DIST:
            filtered_xy.append(pt)
            
    # Always include the exact final goal point
    if np.linalg.norm(raw_xy[-1] - filtered_xy[-1]) > 0.01:
        filtered_xy.append(raw_xy[-1])
        
    filtered_xy = np.array(filtered_xy)

    # 3. Calculate reliable path arc length (s)
    diffs = np.diff(filtered_xy, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    path_s = np.concatenate(([0.0], np.cumsum(dists)))

    # 4. Calculate stable headings using the filtered points
    dx = diffs[:, 0]
    dy = diffs[:, 1]
    path_theta = np.arctan2(dy, dx)
    
    # Duplicate the last heading for the final point to keep arrays equal length
    path_theta = np.append(path_theta, path_theta[-1])

    # 5. Unwrap the heading to prevent 2*pi jumps during linear interpolation
    path_theta_unwrapped = np.unwrap(path_theta)

    return filtered_xy, path_theta_unwrapped, path_s

def get_local_reference(state, path_xy, path_theta_unwrapped, path_s, T, dt, v_target, progress_idx=0):
    x, y = state[0], state[1]
    
    search_start = progress_idx
    search_end = min(len(path_s), progress_idx + 50) 
    
    distances_to_path = np.linalg.norm(
        path_xy[search_start:search_end] - np.array([x, y]), axis=1
    )
    closest_idx = search_start + np.argmin(distances_to_path)
    
    # NEW: Calculate continuous projection to prevent the carrot from lagging
    p_closest = path_xy[closest_idx]
    if closest_idx < len(path_xy) - 1:
        p_next = path_xy[closest_idx + 1]
        vec_path = p_next - p_closest
        vec_robot = np.array([x, y]) - p_closest
        
        path_len2 = np.dot(vec_path, vec_path)
        if path_len2 > 1e-5:
            # Vector projection to find exact fractional progress 't'
            t = np.dot(vec_robot, vec_path) / path_len2
            t = np.clip(t, 0.0, 1.0)
            s_current = path_s[closest_idx] + t * np.linalg.norm(vec_path)
        else:
            s_current = path_s[closest_idx]
    else:
        s_current = path_s[closest_idx]
        
    # Add lookahead so the horizon starts decisively in front of the axle
    lookahead_dist = 0.5
    s_target = s_current + lookahead_dist + np.arange(T) * (v_target * dt)
    
    # 3. Linearly interpolate x, y, and unwrapped theta at the target 's' values
    ref_x = np.interp(s_target, path_s, path_xy[:, 0])
    ref_y = np.interp(s_target, path_s, path_xy[:, 1])
    ref_theta_unwrapped = np.interp(s_target, path_s, path_theta_unwrapped)
    
    # 4. Re-wrap theta back to [-pi, pi]
    ref_theta = (ref_theta_unwrapped + np.pi) % (2 * np.pi) - np.pi
    
    return np.stack([ref_x, ref_y, ref_theta], axis=-1), closest_idx
