"""
shared_pipeline/evaluation_utils.py
===================================
Shared, model-neutral evaluation utilities implementing Decision 7.

Every classifier (XGBoost, LightGBM, Random-Forest, Logistic-Regression, MLP)
MUST use these identical functions to report CV and test metrics.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss,
)

from .config import (
    TRIP_COL, OBS_COL, CAND_COL, LABEL_COL,
    GAP_RESET_SECONDS,
)


def evaluate_candidate_level(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes candidate-level metrics:
      - accuracy
      - precision (pos_label=1)
      - recall (pos_label=1)
      - macro_f1
      - log_loss
      - confusion_matrix (labels=[0, 1] -> [[TN, FP], [FN, TP]])
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # In case y_prob is 1D (probabilities of positive class):
    if y_prob.ndim == 1:
        prob_matrix = np.column_stack([1.0 - y_prob, y_prob])
    else:
        prob_matrix = y_prob

    return {
        "candidate_accuracy": float(accuracy_score(y_true, y_pred)),
        "candidate_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "candidate_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "candidate_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "candidate_log_loss": float(log_loss(y_true, prob_matrix, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def evaluate_top1_accuracy(metadata_df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Observation-level Top-1 accuracy on covered observations:
    For each (trip_id, observation_id), select the candidate with highest positive-class
    probability (tie-breaker: lowest candidate_id).
    """
    eval_df = metadata_df[[TRIP_COL, OBS_COL, CAND_COL]].copy()
    eval_df["y_true"] = np.asarray(y_true, dtype=int)
    eval_df["y_prob"] = np.asarray(y_prob, dtype=float)

    # Sort descending by prob, ascending by candidate_id for deterministic tie-breaking
    eval_df = eval_df.sort_values(
        by=["y_prob", CAND_COL],
        ascending=[False, True],
    )

    # Pick the top candidate per observation
    top1_per_obs = eval_df.drop_duplicates(subset=[TRIP_COL, OBS_COL], keep="first")
    n_obs = len(top1_per_obs)
    n_correct = int((top1_per_obs["y_true"] == 1).sum())
    top1_acc = float(n_correct / n_obs) if n_obs > 0 else 0.0

    return {
        "top1_accuracy": top1_acc,
        "covered_observations": n_obs,
        "top1_correct_observations": n_correct,
        "top1_predictions": top1_per_obs[[TRIP_COL, OBS_COL, CAND_COL]],
    }


def evaluate_miss_rate(coverage_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Candidate Miss Rate:
    Fraction of reference-aligned observations whose reference arc is missing
    from the candidate set.
    """
    aligned = coverage_df[coverage_df["is_aligned"]]
    total_aligned = len(aligned)
    covered = int(aligned["ref_in_candidates"].sum())
    missed = total_aligned - covered
    miss_rate = float(missed / total_aligned) if total_aligned > 0 else 0.0

    return {
        "candidate_miss_rate": miss_rate,
        "total_aligned_observations": total_aligned,
        "covered_observations": covered,
        "missed_observations": missed,
        "total_gps_observations": len(coverage_df),
    }


def evaluate_route_continuity(top1_df: pd.DataFrame, arcs_by_trip: Dict[str, pd.DataFrame], track_by_trip: Dict[str, pd.DataFrame] = None) -> float:
    """
    Route Continuity:
    Fraction of consecutive predicted arc pairs that share a directed endpoint
    (to_node of arc_t == from_node of arc_{t+1}).
    Pairs spanning a timestamp gap > GAP_RESET_SECONDS are excluded if track_by_trip is provided.
    """
    valid_transitions = 0
    total_transitions = 0

    for trip_id, group in top1_df.groupby(TRIP_COL):
        if trip_id not in arcs_by_trip:
            continue
        group = group.sort_values(OBS_COL)
        pred_arcs = group[CAND_COL].to_numpy()
        obs_ids = group[OBS_COL].to_numpy()

        if len(pred_arcs) < 2:
            continue

        arcs_table = arcs_by_trip[trip_id]
        from_nodes = arcs_table["from_node_id"].to_numpy()
        to_nodes   = arcs_table["to_node_id"].to_numpy()

        ts = None
        if track_by_trip and trip_id in track_by_trip:
            ts = track_by_trip[trip_id]["timestamp_seconds"].to_numpy()

        for t in range(len(pred_arcs) - 1):
            if ts is not None:
                o1, o2 = obs_ids[t], obs_ids[t + 1]
                if (ts[o2] - ts[o1]) > GAP_RESET_SECONDS:
                    continue  # exclude gap

            total_transitions += 1
            a1 = pred_arcs[t]
            a2 = pred_arcs[t + 1]
            if to_nodes[a1] == from_nodes[a2]:
                valid_transitions += 1

    return float(valid_transitions / total_transitions) if total_transitions > 0 else 0.0


def summarize_cv_scores(fold_metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a summary table showing each fold's score, plus mean and standard deviation.
    """
    df = pd.DataFrame(fold_metrics_list)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    mean_row = df[numeric_cols].mean()
    std_row  = df[numeric_cols].std()
    summary = df.copy()
    summary.loc["Mean"] = mean_row
    summary.loc["Std"]  = std_row
    return summary
