"""Shared input validation and geometry. Arc/node row numbers are immutable IDs."""
from pathlib import Path
import numpy as np

EARTH_RADIUS_M = 6371008.8


def load_trip(directory, reference=True):
    directory = Path(directory)
    tid = directory.name
    arrays = {}
    for suffix, cols, dtype in [('track', 3, float), ('nodes', 2, float), ('arcs', 2, int)]:
        value = np.loadtxt(directory / f'{tid}.{suffix}', dtype=dtype, ndmin=2)
        if value.shape[1] != cols or not np.isfinite(value).all():
            raise ValueError(f'{tid}.{suffix}: invalid schema or non-finite value')
        arrays[suffix] = value
    if reference:
        arrays['route'] = np.loadtxt(directory / f'{tid}.route', dtype=int, ndmin=1)
    n = len(arrays['nodes'])
    if (arrays['arcs'] < 0).any() or (arrays['arcs'] >= n).any():
        raise ValueError(f'{tid}: invalid node ID')
    if reference and ((arrays['route'] < 0).any() or (arrays['route'] >= len(arrays['arcs'])).any()):
        raise ValueError(f'{tid}: invalid reference arc ID')
    for key in ['nodes', 'track']:
        a = arrays[key]
        if (np.abs(a[:, 0]) > 180).any() or (np.abs(a[:, 1]) > 90).any():
            raise ValueError(f'{tid}: invalid longitude/latitude')
    return arrays


def haversine(a, b):
    a, b = np.radians(a), np.radians(b)
    dl, dp = b[..., 0]-a[..., 0], b[..., 1]-a[..., 1]
    h = np.sin(dp/2)**2 + np.cos(a[..., 1])*np.cos(b[..., 1])*np.sin(dl/2)**2
    return 2*EARTH_RADIUS_M*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def project(lonlat, origin):
    """Local equirectangular metres; origin chosen per trip, no learned transform."""
    ll = np.radians(np.asarray(lonlat))
    o = np.radians(origin)
    return EARTH_RADIUS_M * (ll-o) * np.array([np.cos(o[1]), 1.0])


def segment_distance(point, starts, ends):
    delta = ends-starts
    denom = np.sum(delta*delta, axis=-1)
    fraction = np.clip(np.sum((point-starts)*delta, axis=-1)/np.maximum(denom, 1e-12), 0, 1)
    closest = starts + fraction[..., None]*delta
    return np.linalg.norm(point-closest, axis=-1), fraction


def trip_directories(data):
    for split in ['train', 'student_test']:
        for p in sorted((Path(data)/split).iterdir()):
            if p.is_dir():
                yield split, p
