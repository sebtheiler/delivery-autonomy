import math

# WGS84, the ellipsoid GPS fixes (and Gazebo's NavSat sensor) use
WGS84_A = 6378137.0                   # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563         # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared

def geodetic_to_ecef(lat_rad, lon_rad):
    """ECEF position [m] of a point on the WGS84 ellipsoid (height 0)."""
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)  # prime-vertical radius of curvature
    return (n * cos_lat * math.cos(lon_rad),
            n * cos_lat * math.sin(lon_rad),
            n * (1.0 - WGS84_E2) * sin_lat)

def latlon2meters(lon_rad, lat_rad, origin_lon, origin_lat):
    """East (x) and north (y) [m] of a fix from the origin, in the origin's local tangent plane."""
    x0, y0, z0 = geodetic_to_ecef(origin_lat, origin_lon)
    x1, y1, z1 = geodetic_to_ecef(lat_rad, lon_rad)
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

    sin_lat, cos_lat = math.sin(origin_lat), math.cos(origin_lat)
    sin_lon, cos_lon = math.sin(origin_lon), math.cos(origin_lon)
    x = -sin_lon * dx + cos_lon * dy
    y = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

    return x, y
