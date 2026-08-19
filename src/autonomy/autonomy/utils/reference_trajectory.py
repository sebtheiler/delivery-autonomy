import numpy as np

MIN_DIST = 0.1

def process_path(path_msg):
    raw_x = [pose.pose.position.x for pose in path_msg.poses]
    raw_y = [pose.pose.position.y for pose in path_msg.poses]
    raw_xy = np.column_stack((raw_x, raw_y))

    if len(raw_xy) < 2:
        return raw_xy, np.zeros(len(raw_xy)), np.zeros(len(raw_xy))

    filtered_xy = [raw_xy[0]]
    for pt in raw_xy[1:]:
        if np.linalg.norm(pt - filtered_xy[-1]) >= MIN_DIST:
            filtered_xy.append(pt)

    # The goal must survive filtering even if it falls close to its predecessor
    if np.linalg.norm(raw_xy[-1] - filtered_xy[-1]) > 0.01:
        filtered_xy.append(raw_xy[-1])

    filtered_xy = np.array(filtered_xy)

    diffs = np.diff(filtered_xy, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    path_s = np.concatenate(([0.0], np.cumsum(dists)))

    path_theta = np.arctan2(diffs[:, 1], diffs[:, 0])

    # The final point has no successor, so it inherits the preceding heading
    path_theta = np.append(path_theta, path_theta[-1])

    # Unwrapped so that interpolation does not jump across the pi boundary
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
