import math

import pytest

from autonomy.utils.latlon2meters import latlon2meters

# global_planner's bounding box, and Gazebo's world origin (washu.world)
LAT_MIN, LAT_MAX = 38.6473, 38.6490
LON_MIN, LON_MAX = -90.3041, -90.3013
GZ_ORIGIN = (38.64817527, -90.30128152)

MAX_ERROR_PER_METRE = 1e-4  # 1 cm per 100 m

# (lat, lon, east, north) from GZ_ORIGIN along the WGS84 geodesic (pyproj.Geod.inv, GeographicLib)
REFERENCE = [
    (38.64730000, -90.30410000, -245.362276, -97.158876),
    (38.64900000, -90.30130000, -1.608735, 91.552275),
    (38.64730000, -90.30130000, -1.608773, -97.162645),
    (38.64900000, -90.30410000, -245.356479, 91.556044),
    (38.64817527, -90.30410000, -245.359291, 0.003769),
    (38.64900000, -90.30128152, 0.000000, 91.552275),
    (38.64810000, -90.30270000, -123.484152, -8.354676),
]


def east_north(lat_deg, lon_deg, origin):
    return latlon2meters(math.radians(lon_deg), math.radians(lat_deg),
                         math.radians(origin[1]), math.radians(origin[0]))


def test_datum_maps_to_zero():
    assert east_north(*GZ_ORIGIN, GZ_ORIGIN) == (0.0, 0.0)


def test_matches_reference_geodesics():
    for lat, lon, ref_e, ref_n in REFERENCE:
        e, n = east_north(lat, lon, GZ_ORIGIN)
        err = math.hypot(e - ref_e, n - ref_n)
        assert err < MAX_ERROR_PER_METRE * math.hypot(ref_e, ref_n), (lat, lon, e, n, err)


def test_matches_pyproj_over_planner_bbox():
    pyproj = pytest.importorskip('pyproj')
    geod = pyproj.Geod(ellps='WGS84')
    centre = ((LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2)
    corners = [(lat, lon) for lat in (LAT_MIN, LAT_MAX) for lon in (LON_MIN, LON_MAX)]
    steps = 20
    worst = 0.0
    for origin in [GZ_ORIGIN, centre] + corners:
        for i in range(steps + 1):
            for j in range(steps + 1):
                lat = LAT_MIN + (LAT_MAX - LAT_MIN) * i / steps
                lon = LON_MIN + (LON_MAX - LON_MIN) * j / steps
                az, _, s = geod.inv(origin[1], origin[0], lon, lat)
                e, n = east_north(lat, lon, origin)
                err = math.hypot(e - s * math.sin(math.radians(az)), n - s * math.cos(math.radians(az)))
                worst = max(worst, err / max(s, 1.0))
    assert worst < MAX_ERROR_PER_METRE, worst
