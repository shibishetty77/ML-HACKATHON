"""
shared_pipeline/candidate_pipeline.py
======================================
Model-neutral pipeline: raw trip files → candidate-level feature tables.

Implements Decisions 1–6 from config.py.

Generates:
1. Candidate feature table (train_candidates.csv / student_test_candidates.csv):
   Contains only trainable candidate sets where the reference arc was found in
   the generated candidates.
   Columns:
     - trip_id (str)
     - observation_id (int64)
     - candidate_id (int64)
     - label (int64, 0 or 1)
     - distance_to_segment_m (float64)
     - projection_fraction (float64)
     - observation_speed_mps (float64)
     - heading_difference_rad (float64)

2. Coverage table (train_coverage.csv / student_test_coverage.csv):
   One row per GPS observation across the entire partition for miss-rate
   and coverage accounting.
   Columns:
     - trip_id (str)
     - observation_id (int64)
     - is_aligned (bool)
     - reference_arc_id (int64)
     - has_candidate (bool)
     - ref_in_candidates (bool)
     - num_candidates (int64)

3. Frozen 5-fold GroupKFold assignments (fold_assignments.json).
"""

import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .config import (
    TRAIN_DIR, STUDENT_TEST_DIR,
    TRAIN_CANDIDATES_PATH, STUDENT_TEST_CANDIDATES_PATH,
    TRAIN_COVERAGE_PATH, STUDENT_TEST_COVERAGE_PATH,
    FOLD_ASSIGNMENTS_PATH, OUTPUT_DIR,
    CANDIDATE_RADIUS_M, MAX_CANDIDATES,
    ALIGNMENT_MAX_DIST_M, ALIGNMENT_WINDOW, GAP_RESET_SECONDS,
    MIN_MOVE_M, FEATURE_COLUMNS,
    N_SPLITS, RANDOM_STATE,
    TRIP_COL, OBS_COL, CAND_COL, LABEL_COL,
    EARTH_RADIUS_M,
)
from .geo_utils import (
    haversine_m, forward_bearing_rad,
    bearing_difference_rad, point_to_segment_distance_m,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw file loading
# ---------------------------------------------------------------------------

def _read_raw(path, ncols):
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype="float64")
    if df.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} cols, got {df.shape[1]}")
    return df


def load_trip(trip_dir: Path):
    tid = trip_dir.name
    track = _read_raw(trip_dir / f"{tid}.track", 3)
    track.columns = ["longitude", "latitude", "timestamp_seconds"]
    nodes = _read_raw(trip_dir / f"{tid}.nodes", 2)
    nodes.columns = ["longitude", "latitude"]
    arcs  = _read_raw(trip_dir / f"{tid}.arcs",  2)
    arcs.columns  = ["from_node_id", "to_node_id"]
    route = _read_raw(trip_dir / f"{tid}.route", 1).iloc[:, 0].astype(int)
    return track, nodes, arcs, route


# ---------------------------------------------------------------------------
# Candidate generation (Decision 2)
# ---------------------------------------------------------------------------

def generate_candidates_for_trip(track, nodes, arcs):
    """
    For each GPS observation, find at most MAX_CANDIDATES arcs within
    CANDIDATE_RADIUS_M using point-to-segment Haversine distance.
    Ties broken deterministically by ascending candidate_id (arc row index).
    """
    n_obs  = len(track)
    n_arcs = len(arcs)

    node_lon = nodes["longitude"].to_numpy()
    node_lat = nodes["latitude"].to_numpy()

    from_ids = arcs["from_node_id"].astype(int).to_numpy()
    to_ids   = arcs["to_node_id"].astype(int).to_numpy()
    a_lon = node_lon[from_ids]
    a_lat = node_lat[from_ids]
    b_lon = node_lon[to_ids]
    b_lat = node_lat[to_ids]

    mid_lon = 0.5 * (a_lon + b_lon)
    mid_lat = 0.5 * (a_lat + b_lat)
    lat0 = mid_lat.mean()
    cos_lat = np.cos(np.radians(lat0))
    lon0 = mid_lon.mean()
    mid_x = (mid_lon - lon0) * cos_lat * (np.pi / 180.0) * EARTH_RADIUS_M
    mid_y = (mid_lat - lat0) * (np.pi / 180.0) * EARTH_RADIUS_M
    arc_tree = KDTree(np.column_stack([mid_x, mid_y]))

    arc_half_len = haversine_m(a_lon, a_lat, b_lon, b_lat) * 0.5
    expand = max(float(arc_half_len.max()) if n_arcs > 0 else 0.0, 50.0)
    search_radius = CANDIDATE_RADIUS_M + expand

    obs_lon = track["longitude"].to_numpy()
    obs_lat = track["latitude"].to_numpy()

    results = []
    for i in range(n_obs):
        px = (obs_lon[i] - lon0) * cos_lat * (np.pi / 180.0) * EARTH_RADIUS_M
        py = (obs_lat[i] - lat0) * (np.pi / 180.0) * EARTH_RADIUS_M

        candidate_indices = arc_tree.query_ball_point([px, py], search_radius)
        if not candidate_indices:
            results.append({
                "observation_id": i,
                "cand_arc_ids": np.array([], dtype=np.int64),
                "distances_m": np.array([], dtype=np.float64),
                "fractions": np.array([], dtype=np.float64),
            })
            continue

        ci = np.array(candidate_indices, dtype=np.int64)
        dists, fracs = point_to_segment_distance_m(
            obs_lon[i], obs_lat[i],
            a_lon[ci], a_lat[ci],
            b_lon[ci], b_lat[ci],
        )

        within = dists <= CANDIDATE_RADIUS_M
        ci = ci[within]
        dists = dists[within]
        fracs = fracs[within]

        if len(ci) == 0:
            results.append({
                "observation_id": i,
                "cand_arc_ids": np.array([], dtype=np.int64),
                "distances_m": np.array([], dtype=np.float64),
                "fractions": np.array([], dtype=np.float64),
            })
            continue

        # Rank by distance ascending, then candidate_id ascending as tie-breaker
        order = np.lexsort((ci, dists))
        ci = ci[order][:MAX_CANDIDATES]
        dists = dists[order][:MAX_CANDIDATES]
        fracs = fracs[order][:MAX_CANDIDATES]

        results.append({
            "observation_id": i,
            "cand_arc_ids": ci,
            "distances_m": dists,
            "fractions": fracs,
        })

    return results


# ---------------------------------------------------------------------------
# Observation speed and bearing (Decision 5)
# ---------------------------------------------------------------------------

def compute_obs_features(track):
    """
    Returns:
        speeds   : observation_speed_mps (length n)
        bearings : movement bearing in [0, 2π) or NaN when undefined (length n)
    """
    n = len(track)
    lon = track["longitude"].to_numpy()
    lat = track["latitude"].to_numpy()
    ts  = track["timestamp_seconds"].to_numpy()

    speeds   = np.zeros(n, dtype=np.float64)
    bearings = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        if dt <= 0.0 or dt > GAP_RESET_SECONDS:
            speeds[i] = 0.0
            bearings[i] = np.nan
            continue
        d = haversine_m(lon[i - 1], lat[i - 1], lon[i], lat[i])
        speeds[i] = d / dt
        if d >= MIN_MOVE_M:
            bearings[i] = forward_bearing_rad(lon[i - 1], lat[i - 1], lon[i], lat[i])

    return speeds, bearings


# ---------------------------------------------------------------------------
# Reference route alignment (Decision 3)
# ---------------------------------------------------------------------------

def align_reference(track, nodes, arcs, route):
    """
    Greedy forward window alignment along ordered route arcs.
    Returns:
        ref_arc_ids : (n_obs,) int64 array; -1 = unaligned
        is_aligned  : (n_obs,) bool array
    """
    n_obs   = len(track)
    n_route = len(route)
    ref_arc_ids = np.full(n_obs, -1, dtype=np.int64)
    is_aligned  = np.zeros(n_obs, dtype=bool)

    if n_route == 0:
        return ref_arc_ids, is_aligned

    node_lon = nodes["longitude"].to_numpy()
    node_lat = nodes["latitude"].to_numpy()
    from_ids = arcs["from_node_id"].astype(int).to_numpy()
    to_ids   = arcs["to_node_id"].astype(int).to_numpy()

    route_arr = route.to_numpy()
    ra_lon = node_lon[from_ids[route_arr]]
    ra_lat = node_lat[from_ids[route_arr]]
    rb_lon = node_lon[to_ids[route_arr]]
    rb_lat = node_lat[to_ids[route_arr]]

    obs_lon = track["longitude"].to_numpy()
    obs_lat = track["latitude"].to_numpy()
    obs_ts  = track["timestamp_seconds"].to_numpy()

    route_ptr = 0

    for i in range(n_obs):
        is_gap = (i > 0 and (obs_ts[i] - obs_ts[i - 1]) > GAP_RESET_SECONDS)
        cur_w = (n_route - route_ptr) if is_gap else min(ALIGNMENT_WINDOW, n_route - route_ptr)
        if cur_w <= 0:
            break

        sub_lon_a = ra_lon[route_ptr:route_ptr + cur_w]
        sub_lat_a = ra_lat[route_ptr:route_ptr + cur_w]
        sub_lon_b = rb_lon[route_ptr:route_ptr + cur_w]
        sub_lat_b = rb_lat[route_ptr:route_ptr + cur_w]

        dists, _ = point_to_segment_distance_m(
            obs_lon[i], obs_lat[i], sub_lon_a, sub_lat_a, sub_lon_b, sub_lat_b
        )
        best_idx = int(np.argmin(dists))

        if dists[best_idx] <= ALIGNMENT_MAX_DIST_M:
            route_ptr += best_idx
            ref_arc_ids[i] = route_arr[route_ptr]
            is_aligned[i] = True
        elif not is_gap:
            # Fallback forward search in case of fast travel over many arcs
            wider_w = min(100, n_route - route_ptr)
            sub_lon_a = ra_lon[route_ptr:route_ptr + wider_w]
            sub_lat_a = ra_lat[route_ptr:route_ptr + wider_w]
            sub_lon_b = rb_lon[route_ptr:route_ptr + wider_w]
            sub_lat_b = rb_lat[route_ptr:route_ptr + wider_w]
            dists2, _ = point_to_segment_distance_m(
                obs_lon[i], obs_lat[i], sub_lon_a, sub_lat_a, sub_lon_b, sub_lat_b
            )
            best_idx2 = int(np.argmin(dists2))
            if dists2[best_idx2] <= ALIGNMENT_MAX_DIST_M:
                route_ptr += best_idx2
                ref_arc_ids[i] = route_arr[route_ptr]
                is_aligned[i] = True

    return ref_arc_ids, is_aligned


# ---------------------------------------------------------------------------
# Trip candidate processing
# ---------------------------------------------------------------------------

def build_trip_data(trip_id, track, nodes, arcs, route):
    """
    Builds both candidate rows (for model training/scoring) and observation
    coverage rows (for coverage / miss rate accounting).
    """
    speeds, bearings = compute_obs_features(track)
    ref_arc_ids, is_aligned = align_reference(track, nodes, arcs, route)
    cand_results = generate_candidates_for_trip(track, nodes, arcs)

    node_lon = nodes["longitude"].to_numpy()
    node_lat = nodes["latitude"].to_numpy()
    from_ids = arcs["from_node_id"].astype(int).to_numpy()
    to_ids   = arcs["to_node_id"].astype(int).to_numpy()

    candidate_rows = []
    coverage_rows = []

    for obs_res in cand_results:
        i       = int(obs_res["observation_id"])
        arc_ids = obs_res["cand_arc_ids"]
        dists   = obs_res["distances_m"]
        fracs   = obs_res["fractions"]
        ref_arc = int(ref_arc_ids[i])
        aligned = bool(is_aligned[i])
        has_cand = len(arc_ids) > 0
        ref_in_cands = aligned and (ref_arc in arc_ids)

        coverage_rows.append({
            TRIP_COL: trip_id,
            OBS_COL: i,
            "is_aligned": aligned,
            "reference_arc_id": ref_arc,
            "has_candidate": has_cand,
            "ref_in_candidates": ref_in_cands,
            "num_candidates": len(arc_ids),
        })

        # Only trainable observations enter the candidate feature table
        if not ref_in_cands:
            continue

        for k, arc_id in enumerate(arc_ids):
            label = 1 if (arc_id == ref_arc) else 0

            arc_bearing = forward_bearing_rad(
                node_lon[from_ids[arc_id]], node_lat[from_ids[arc_id]],
                node_lon[to_ids[arc_id]],   node_lat[to_ids[arc_id]],
            )

            if np.isnan(bearings[i]):
                hdiff = 0.0
            else:
                arc_len = haversine_m(
                    node_lon[from_ids[arc_id]], node_lat[from_ids[arc_id]],
                    node_lon[to_ids[arc_id]],   node_lat[to_ids[arc_id]],
                )
                if arc_len < MIN_MOVE_M:
                    hdiff = 0.0
                else:
                    hdiff = float(bearing_difference_rad(bearings[i], arc_bearing))

            candidate_rows.append({
                TRIP_COL: trip_id,
                OBS_COL: i,
                CAND_COL: int(arc_id),
                LABEL_COL: int(label),
                "distance_to_segment_m": float(dists[k]),
                "projection_fraction": float(fracs[k]),
                "observation_speed_mps": float(speeds[i]),
                "heading_difference_rad": float(hdiff),
            })

    cand_df = pd.DataFrame(candidate_rows)
    cov_df  = pd.DataFrame(coverage_rows)

    if not cand_df.empty:
        cand_df = cand_df.astype({
            TRIP_COL: "object",
            OBS_COL: "int64",
            CAND_COL: "int64",
            LABEL_COL: "int64",
            "distance_to_segment_m": "float64",
            "projection_fraction": "float64",
            "observation_speed_mps": "float64",
            "heading_difference_rad": "float64",
        })

    return cand_df, cov_df


# ---------------------------------------------------------------------------
# Partition processing
# ---------------------------------------------------------------------------

def process_partition(partition_dir: Path, partition_name: str):
    trip_dirs = sorted(p for p in partition_dir.iterdir() if p.is_dir())
    all_cand_dfs = []
    all_cov_dfs  = []
    t_start = time.perf_counter()

    for trip_dir in trip_dirs:
        tid = trip_dir.name
        track, nodes, arcs, route = load_trip(trip_dir)
        t0 = time.perf_counter()
        cand_df, cov_df = build_trip_data(tid, track, nodes, arcs, route)
        elapsed = time.perf_counter() - t0

        n_obs = len(cov_df)
        n_aligned = int(cov_df["is_aligned"].sum())
        n_covered = int(cov_df["ref_in_candidates"].sum())
        log.info(
            f"[{partition_name}] trip {tid}: "
            f"obs={n_obs} | aligned={n_aligned} ({n_aligned/n_obs*100:.1f}%) | "
            f"covered={n_covered} ({n_covered/n_obs*100:.1f}%) | "
            f"rows={len(cand_df)} | {elapsed:.2f}s"
        )
        all_cand_dfs.append(cand_df)
        all_cov_dfs.append(cov_df)

    total_time = time.perf_counter() - t_start
    log.info(f"[{partition_name}] total processing time: {total_time:.2f}s")

    full_cand_df = pd.concat(all_cand_dfs, ignore_index=True)
    full_cov_df  = pd.concat(all_cov_dfs,  ignore_index=True)
    return full_cand_df, full_cov_df


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(force=False):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Train Partition ---
    if force or not TRAIN_CANDIDATES_PATH.exists() or not TRAIN_COVERAGE_PATH.exists():
        log.info("Processing train partition …")
        train_cand_df, train_cov_df = process_partition(TRAIN_DIR, "train")
        train_cand_df.to_csv(TRAIN_CANDIDATES_PATH, index=False)
        train_cov_df.to_csv(TRAIN_COVERAGE_PATH, index=False)
        log.info(f"Saved {TRAIN_CANDIDATES_PATH} ({len(train_cand_df):,} rows)")
        log.info(f"Saved {TRAIN_COVERAGE_PATH} ({len(train_cov_df):,} obs)")
    else:
        log.info(f"Training outputs already exist at {TRAIN_CANDIDATES_PATH}")
        train_cand_df = pd.read_csv(TRAIN_CANDIDATES_PATH)

    # --- Frozen GroupKFold (Decision 6) ---
    if force or not FOLD_ASSIGNMENTS_PATH.exists():
        from sklearn.model_selection import GroupKFold
        trip_ids = train_cand_df[TRIP_COL].to_numpy()
        X_dummy = np.zeros((len(train_cand_df), 1))
        gkf = GroupKFold(n_splits=N_SPLITS)
        folds = {}
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_dummy, groups=trip_ids)):
            tr_trips = sorted(set(trip_ids[tr_idx].tolist()))
            va_trips = sorted(set(trip_ids[va_idx].tolist()))
            folds[str(fold_idx)] = {"train_trips": tr_trips, "val_trips": va_trips}
        with open(FOLD_ASSIGNMENTS_PATH, "w") as f:
            json.dump({"n_splits": N_SPLITS, "random_state": RANDOM_STATE, "folds": folds}, f, indent=2)
        log.info(f"Saved frozen folds to {FOLD_ASSIGNMENTS_PATH}")
    else:
        log.info(f"Fold assignments already exist at {FOLD_ASSIGNMENTS_PATH}")

    # --- Student Test Partition ---
    if force or not STUDENT_TEST_CANDIDATES_PATH.exists() or not STUDENT_TEST_COVERAGE_PATH.exists():
        log.info("Processing student_test partition …")
        test_cand_df, test_cov_df = process_partition(STUDENT_TEST_DIR, "student_test")
        test_cand_df.to_csv(STUDENT_TEST_CANDIDATES_PATH, index=False)
        test_cov_df.to_csv(STUDENT_TEST_COVERAGE_PATH, index=False)
        log.info(f"Saved {STUDENT_TEST_CANDIDATES_PATH} ({len(test_cand_df):,} rows)")
        log.info(f"Saved {STUDENT_TEST_COVERAGE_PATH} ({len(test_cov_df):,} obs)")
    else:
        log.info(f"Student test outputs already exist at {STUDENT_TEST_CANDIDATES_PATH}")

    return TRAIN_CANDIDATES_PATH, STUDENT_TEST_CANDIDATES_PATH, FOLD_ASSIGNMENTS_PATH


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
