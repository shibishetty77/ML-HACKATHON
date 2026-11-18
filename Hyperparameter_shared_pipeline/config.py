"""
shared_pipeline/config.py
=========================
Canonical, frozen decisions for the PathSnap shared candidate-feature-label
pipeline.  Every model (XGBoost, LightGBM, Random-Forest, Logistic-Regression,
MLP) MUST import its parameters from this single file.

Decision log
------------
All 7 open decisions from Shared_Candidate_Pipeline_Proposal.md §6 are
resolved here with best-practice defaults.

Decision 1 — Metric geometry
    Method : Haversine great-circle formula (pure numpy, no pyproj dependency).
             Equivalent to WGS-84 geodesic for distances up to a few km.
    Units  : metres throughout.
    Bearing: forward azimuth computed from lon/lat differences in the same
             coordinate system; wraps to [0, 2π) before differencing.

Decision 2 — Candidate generation
    Distance : point-to-segment Haversine (project GPS point onto arc segment
               analytically in a local tangent plane centred on the arc
               midpoint, then convert the clamped closest-point distance back
               to metres).
    Radius   : 50 m  — standard urban map-matching first-pass radius.
    Max count: 10 candidates per GPS observation, ranked by distance (nearest
               first), ties broken by ascending candidate_id (arc row index).
    Direction: arcs kept as directed (from_node → to_node); reverse arcs are
               separate rows in the .arcs file and are treated independently.
    Empty-set: observations with no arc within 50 m are recorded as candidate
               misses in the coverage table. Their rows are excluded from classifier
               training but counted in coverage and Miss-Rate reporting.

Decision 3 — Reference alignment (binary labels)
    Algorithm   : greedy forward-window search along the ordered reference route.
                  For each GPS observation, search within a forward window of 30
                  reference arcs. Assign to the nearest reference arc if distance
                  <= ALIGNMENT_MAX_DIST_M (100m).
    Tolerance   : 100 m (looser than candidate radius to tolerate GPS noise on
                  the reference route).
    Timestamp gap: gaps > GAP_RESET_SECONDS (30s) allow a search window spanning
                  all remaining route arcs.
    Repeated arcs: the route pointer advances monotonically; a repeated arc ID
                  later in the route is treated as a fresh occurrence.
    Tie (two equal-distance arcs): lower reference-route index wins.

Decision 4 — Positive labels
    Single positive per observation (the aligned reference arc).
    Only trainable candidate sets enter the feature table (observations where the
    reference arc was found in the top-K candidates within 50m).
    The reference candidate is labelled 1; all other candidates for that observation
    are labelled 0.
    Observations where the reference arc is absent from the candidate set (or unaligned)
    are recorded in the coverage table for Miss Rate accounting, but their candidate rows
    are excluded from classifier training.

Decision 5 — Feature definitions (exact, ordered)
    Features passed to ALL models unchanged (no scaling for tree models;
    models that need scaling fit their scaler inside each training fold):
    1. distance_to_segment_m   : Haversine point-to-segment distance (metres)
    2. projection_fraction     : clamped [0,1] fraction along arc at nearest
                                 projected point
    3. observation_speed_mps   : Haversine distance to previous observation
                                 divided by positive timestamp difference;
                                 first observation of a trip → 0.0;
                                 gap > GAP_RESET_SECONDS → 0.0 (treat as rest)
    4. heading_difference_rad  : absolute wrapped angle ∈ [0, π] between the
                                 observation movement bearing and the directed
                                 arc bearing;
                                 first observation, or movement < MIN_MOVE_M
                                 → 0.0 (undefined bearing filled with zero);
                                 zero-length arc → 0.0.

Decision 6 — Cross-validation folds
    GroupKFold(n_splits=5), group = trip_id, iteration order = sklearn default.
    Random seed = 42.
    Fold assignments are frozen in shared_pipeline/fold_assignments.json and
    MUST NOT be regenerated.

Decision 7 — Evaluation metrics and edge cases
    Candidate-level (threshold = 0.5, pos_label = 1):
        accuracy, precision (binary), recall (binary), macro-F1,
        confusion matrix labels=[0,1], log-loss class order [0,1].
    Observation-level Top-1:
        per observation pick the candidate with highest positive-class proba;
        tie-break = lowest candidate_id.
    Candidate Miss Rate = #observations with no positive candidate /
                          #reference-labelled observations (denominator includes
                          unalignable observations).
    Route Continuity = fraction of consecutive predicted-arc pairs that share a
                       directed endpoint (to_node of arc_i == from_node of
                       arc_{i+1}); pairs spanning a timestamp gap >
                       GAP_RESET_SECONDS are excluded from the denominator.
    Timing: train time = time around clf.fit(); inference time = time around
            clf.predict_proba() on the full validation set.
            Candidate-generation time is reported separately.
    Reporting: mean ± std across 5 fold scores.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_ROOT  = WORKSPACE_ROOT / "Student folder" / "Student folder"
TRAIN_DIR      = RAW_DATA_ROOT / "train"
STUDENT_TEST_DIR = RAW_DATA_ROOT / "student_test"
OUTPUT_DIR     = WORKSPACE_ROOT / "shared_pipeline" / "outputs"

TRAIN_CANDIDATES_PATH        = OUTPUT_DIR / "train_candidates.csv"
STUDENT_TEST_CANDIDATES_PATH = OUTPUT_DIR / "student_test_candidates.csv"
TRAIN_COVERAGE_PATH          = OUTPUT_DIR / "train_coverage.csv"
STUDENT_TEST_COVERAGE_PATH   = OUTPUT_DIR / "student_test_coverage.csv"
FOLD_ASSIGNMENTS_PATH        = WORKSPACE_ROOT / "shared_pipeline" / "fold_assignments.json"

# ---------------------------------------------------------------------------
# Decision 1 — Metric geometry
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6_371_000.0   # mean Earth radius for Haversine

# ---------------------------------------------------------------------------
# Decision 2 — Candidate generation
# ---------------------------------------------------------------------------
CANDIDATE_RADIUS_M  = 50.0   # metres; maximum point-to-segment distance
MAX_CANDIDATES      = 10     # maximum candidates per observation

# ---------------------------------------------------------------------------
# Decision 3 — Reference alignment
# ---------------------------------------------------------------------------
ALIGNMENT_MAX_DIST_M   = 100.0  # metres; max GPS-to-reference-arc distance
ALIGNMENT_WINDOW       = 30     # forward search window in route
GAP_RESET_SECONDS      = 30.0   # timestamp gap that triggers route-pointer reset

# ---------------------------------------------------------------------------
# Decision 5 — Features
# ---------------------------------------------------------------------------
MIN_MOVE_M = 1.0   # minimum movement (metres) for a valid bearing estimate

FEATURE_COLUMNS = [
    "distance_to_segment_m",
    "projection_fraction",
    "observation_speed_mps",
    "heading_difference_rad",
]

# ---------------------------------------------------------------------------
# Decision 6 — CV
# ---------------------------------------------------------------------------
N_SPLITS    = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Column name constants (used identically by all models)
# ---------------------------------------------------------------------------
TRIP_COL        = "trip_id"
OBS_COL         = "observation_id"
CAND_COL        = "candidate_id"
LABEL_COL       = "label"
