"""Filter XYZ optical-frame coordinates, in meters. Not obstacle classification."""
import numpy as np


def filter_xyz(points, min_depth=0.2, max_depth=4.0, stride=4):
    if not np.isfinite([min_depth, max_depth]).all() or not 0 <= min_depth < max_depth:
        raise ValueError('Require finite 0 <= min_depth < max_depth')
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('stride must be a positive integer')
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    xyz = xyz[::stride]
    valid = np.isfinite(xyz).all(axis=1)
    valid &= (xyz[:, 2] >= min_depth) & (xyz[:, 2] <= max_depth)
    return xyz[valid]
