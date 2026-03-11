import math

def latlon2meters(lon_rad, lat_rad, origin_lon, origin_lat):
    EARTH_RADIUS = 6378137.0
    x = EARTH_RADIUS * (lon_rad - origin_lon) * math.cos(origin_lat)
    y = EARTH_RADIUS * (lat_rad - origin_lat)

    return x, y
