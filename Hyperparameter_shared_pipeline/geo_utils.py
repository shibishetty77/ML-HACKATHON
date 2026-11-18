"""
shared_pipeline/geo_utils.py
============================
Pure-numpy geodesic geometry utilities.

All distance calculations use the Haversine great-circle formula with
EARTH_RADIUS_M = 6 371 000 m (Decision 1).  For the arc lengths involved
(< 1 km) the Haversine and WGS-84 geodesic differ by < 0.01 %, which is
negligible relative to GPS noise.

Point-to-segment projection uses a local tangent-plane approximation centred
on the arc midpoint: longitude differences are scaled by cos(lat_mid) so that
one degree in each direction corresponds to the same number of metres, giving
a planar coordinate system in which ordinary Euclidean projection is correct.
The approximation error is < 0.1 % for arc lengths up to 500 m.
"""

import numpy as np
from .config import EARTH_RADIUS_M, MIN_MOVE_M


# ---------------------------------------------------------------------------
# Basic great-circle helpers
# ---------------------------------------------------------------------------

def haversine_m(lon1, lat1, lon2, lat2):
    """
    Vectorised Haversine distance in metres.

    All inputs may be numpy arrays of matching shape or Python scalars.
    """
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def forward_bearing_rad(lon1, lat1, lon2, lat2):
    """
    Forward azimuth from (lon1, lat1) → (lon2, lat2) in radians ∈ [0, 2π).
    """
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    dlon = np.radians(lon2 - lon1)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.mod(np.arctan2(x, y), 2.0 * np.pi)


def bearing_difference_rad(b1, b2):
    """
    Minimum absolute angular difference between two bearings in [0, 2π).
    Returns a value in [0, π].
    """
    diff = np.abs(b1 - b2) % (2.0 * np.pi)
    return np.where(diff > np.pi, 2.0 * np.pi - diff, diff)


# ---------------------------------------------------------------------------
# Local tangent-plane helpers
# ---------------------------------------------------------------------------

def _local_xy(lon, lat, lon0, lat0):
    """
    Convert (lon, lat) to approximate (x, y) in metres relative to (lon0, lat0).
    x increases eastward, y increases northward.
    """
    cos_lat0 = np.cos(np.radians(lat0))
    x = (lon - lon0) * cos_lat0 * (np.pi / 180.0) * EARTH_RADIUS_M
    y = (lat - lat0)             * (np.pi / 180.0) * EARTH_RADIUS_M
    return x, y


# ---------------------------------------------------------------------------
# Point-to-segment distance (vectorised over many segments at once)
# ---------------------------------------------------------------------------

def point_to_segment_distance_m(
    p_lon, p_lat,           # GPS point (scalar floats)
    a_lon, a_lat,           # from-node of each arc (1-D arrays, length N)
    b_lon, b_lat,           # to-node of each arc (1-D arrays, length N)
):
    """
    For a single GPS point (p_lon, p_lat) and N directed arc segments
    [A_i → B_i], return:

        distances   : (N,) array of Haversine point-to-segment distances in m
        fractions   : (N,) array of projection fractions clamped to [0, 1]

    Algorithm
    ---------
    1. Build a local tangent plane at the arc midpoint.
    2. Project P, A, B into the plane (approximate metres).
    3. Compute the clamped fraction t = dot(AP, AB) / dot(AB, AB).
    4. Find the closest point Q = A + t * AB.
    5. Convert Q back to lon/lat and return haversine(P, Q).

    For zero-length arcs (A == B) the fraction is 0.0 and the distance is
    haversine(P, A).
    """
    a_lon = np.asarray(a_lon, dtype=np.float64)
    a_lat = np.asarray(a_lat, dtype=np.float64)
    b_lon = np.asarray(b_lon, dtype=np.float64)
    b_lat = np.asarray(b_lat, dtype=np.float64)

    mid_lon = 0.5 * (a_lon + b_lon)
    mid_lat = 0.5 * (a_lat + b_lat)

    # Project all points to local tangent plane at each arc's midpoint
    px, py = _local_xy(p_lon, p_lat, mid_lon, mid_lat)
    ax, ay = _local_xy(a_lon, a_lat, mid_lon, mid_lat)
    bx, by = _local_xy(b_lon, b_lat, mid_lon, mid_lat)

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_sq = abx * abx + aby * aby
    # Avoid division by zero for zero-length arcs
    safe_ab_sq = np.where(ab_sq > 0.0, ab_sq, 1.0)

    t = np.clip((apx * abx + apy * aby) / safe_ab_sq, 0.0, 1.0)
    t = np.where(ab_sq > 0.0, t, 0.0)

    # Closest point in local plane → back to lon/lat
    qx = ax + t * abx
    qy = ay + t * aby

    cos_lat0 = np.cos(np.radians(mid_lat))
    # Avoid division by zero at poles (irrelevant for this dataset)
    safe_cos = np.where(np.abs(cos_lat0) > 1e-10, cos_lat0, 1e-10)
    q_lon = mid_lon + qx / (safe_cos * (np.pi / 180.0) * EARTH_RADIUS_M)
    q_lat = mid_lat + qy / ((np.pi / 180.0) * EARTH_RADIUS_M)

    distances = haversine_m(p_lon, p_lat, q_lon, q_lat)
    return distances, t
