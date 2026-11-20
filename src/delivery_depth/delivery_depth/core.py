"""Depth image processing independent of ROS, for repeatable tests."""
import numpy as np


def measure(data, width, height, step, encoding, bigendian=False,
            roi_size=40, min_valid_fraction=0.5):
    formats = {'16UC1': ('u2', 0.001), '32FC1': ('f4', 1.0)}
    if encoding not in formats:
        raise ValueError(f'Unsupported depth encoding: {encoding}')
    kind, scale = formats[encoding]
    dtype = np.dtype(('>' if bigendian else '<') + kind)
    if width <= 0 or height <= 0 or roi_size <= 0:
        raise ValueError('Image and ROI dimensions must be positive')
    if not 0 <= min_valid_fraction <= 1:
        raise ValueError('min_valid_fraction must be between 0 and 1')
    if step < width * dtype.itemsize or len(data) < height * step:
        raise ValueError('Invalid row stride or truncated image buffer')
    pixels = np.ndarray((height, width), dtype=dtype, buffer=data,
                        strides=(step, dtype.itemsize))
    rw, rh = min(roi_size, width), min(roi_size, height)
    x, y = (width - rw) // 2, (height - rh) // 2
    region = pixels[y:y + rh, x:x + rw].astype(float) * scale
    valid = region[np.isfinite(region) & (region > 0)]
    fraction = valid.size / region.size
    usable = bool(valid.size and fraction >= min_valid_fraction)
    return {
        'status': 'ok' if usable else 'insufficient_depth',
        'median_depth_m': float(np.median(valid)) if usable else None,
        'nearest_depth_m': float(np.min(valid)) if usable else None,
        'valid_fraction': fraction,
        'roi': {'x': x, 'y': y, 'width': rw, 'height': rh},
    }
