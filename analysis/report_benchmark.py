"""Re-evaluate three Team 9 model families on the shared four-feature data.

Run from the repository root with Python 3.11::

    py -3.11 analysis/report_benchmark.py --model svm
    py -3.11 analysis/report_benchmark.py --model random_forest
    py -3.11 analysis/report_benchmark.py --model knn

XGBoost and the four-feature MLP already have saved fold/test predictions on this
same data. This script does not tune on the student-test partition.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Hyperparameter_shared_pipeline" / "outputs"
FOLDS = ROOT / "Hyperparameter_shared_pipeline" / "fold_assignments.json"
OUT = ROOT / "analysis" / "results"
FEATURES = [
    "distance_to_segment_m", "projection_fraction", "observation_speed_mps",
    "heading_difference_rad",
]
CSV_TYPES = {"trip_id": str, "observation_id": np.int32, "candidate_id": np.int32,
             "label": np.int8}


def load_data(partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(DATA / f"{partition}_candidates.csv", dtype=CSV_TYPES)
    coverage = pd.read_csv(DATA / f"{partition}_coverage.csv", dtype={"trip_id": str})
    assert list(table.columns) == ["trip_id", "observation_id", "candidate_id", "label", *FEATURES]
    assert not table[FEATURES].isna().any().any()
    table["trip_id"] = table.trip_id.str.zfill(8)
    coverage["trip_id"] = coverage.trip_id.str.zfill(8)
    return table, coverage


def create_model(name: str):
    if name == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=42))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=120, criterion="gini", max_depth=18,
            min_samples_leaf=2, max_features="sqrt", bootstrap=True,
            class_weight="balanced_subsample", n_jobs=6, random_state=42)
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=13, weights="distance", metric="manhattan",
            algorithm="kd_tree", n_jobs=6))
    raise ValueError(name)


def score_model(model, name: str, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    x = data[FEATURES].to_numpy(np.float32)
    start = time.perf_counter()
    if name == "svm":
        score = model.decision_function(x)
        classification = (score >= 0).astype(np.int8)
    else:
        score = model.predict_proba(x)[:, 1]
        classification = (score >= 0.5).astype(np.int8)
    return score, classification, time.perf_counter() - start


def evaluate(data: pd.DataFrame, coverage: pd.DataFrame, scores: np.ndarray,
             classified: np.ndarray, *, probabilities: bool) -> tuple[dict, pd.DataFrame]:
    y = data.label.to_numpy(np.int8)
    winners = data[["trip_id", "observation_id", "candidate_id", "label"]].copy()
    winners["score"] = scores
    ix = winners.groupby(["trip_id", "observation_id"], sort=False).score.idxmax()
    winners = winners.loc[ix].sort_values(["trip_id", "observation_id"]).reset_index(drop=True)
    aligned = coverage[coverage.is_aligned]
    result = {
        "candidate_rows": int(len(data)),
        "candidate_macro_f1": float(f1_score(y, classified, average="macro")),
        "confusion_matrix": confusion_matrix(y, classified, labels=[0, 1]).tolist(),
        "conditional_top1": float(winners.label.mean()),
        "end_to_end_top1": float(winners.label.sum() / len(aligned)),
        "covered_observations": int(len(winners)),
        "aligned_observations": int(len(aligned)),
        "candidate_hit_rate": float(aligned.ref_in_candidates.mean()),
        "log_loss": float(log_loss(y, np.clip(scores, 1e-7, 1 - 1e-7))) if probabilities else None,
    }
    return result, winners


def continuity(winners: pd.DataFrame, raw_zip: Path) -> tuple[float, int, int]:
    valid, total = 0, 0
    with zipfile.ZipFile(raw_zip) as archive:
        for trip_id, group in winners.groupby("trip_id"):
            prefix = f"Student folder/student_test/{trip_id}/{trip_id}"
            with archive.open(prefix + ".arcs") as stream:
                arcs = np.loadtxt(io.BytesIO(stream.read()), dtype=np.int32).reshape(-1, 2)
            with archive.open(prefix + ".track") as stream:
                track = np.loadtxt(io.BytesIO(stream.read())).reshape(-1, 3)
            obs = group.observation_id.to_numpy(np.int32)
            arc = group.candidate_id.to_numpy(np.int32)
            dt = track[obs[1:], 2] - track[obs[:-1], 2]
            adjacent = (obs[1:] == obs[:-1] + 1) & (dt > 0) & (dt <= 30)
            match = (arc[1:] == arc[:-1]) | (arcs[arc[:-1], 1] == arcs[arc[1:], 0])
            total += int(adjacent.sum())
            valid += int((adjacent & match).sum())
    return (valid / total if total else 0.0), valid, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["svm", "random_forest", "knn"])
    parser.add_argument("--raw-zip", type=Path, default=Path.home() / "Downloads" / "Student folder.zip")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train, train_cov = load_data("train")
    test, test_cov = load_data("student_test")
    fold_cfg = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    folds = []
    for fold in range(5):
        val_trips = set(fold_cfg[str(fold)]["val_trips"])
        val_mask = train.trip_id.isin(val_trips).to_numpy()
        model = create_model(args.model)
        begin = time.perf_counter()
        model.fit(train.loc[~val_mask, FEATURES].to_numpy(np.float32),
                  train.loc[~val_mask, "label"].to_numpy(np.int8))
        fit_seconds = time.perf_counter() - begin
        val = train.loc[val_mask].reset_index(drop=True)
        scores, classified, infer_seconds = score_model(model, args.model, val)
        val_cov = train_cov[train_cov.trip_id.isin(val_trips)]
        metrics, _ = evaluate(val, val_cov, scores, classified,
                              probabilities=args.model != "svm")
        metrics.update({"fold": fold + 1, "fit_seconds": fit_seconds,
                        "inference_seconds": infer_seconds})
        folds.append(metrics)
        print(args.model, "fold", fold + 1, "f1", f"{metrics['candidate_macro_f1']:.6f}",
              "top1", f"{metrics['conditional_top1']:.6f}", flush=True)
    model = create_model(args.model)
    begin = time.perf_counter()
    model.fit(train[FEATURES].to_numpy(np.float32), train.label.to_numpy(np.int8))
    final_fit_seconds = time.perf_counter() - begin
    scores, classified, inference_seconds = score_model(model, args.model, test)
    test_metrics, winners = evaluate(test, test_cov, scores, classified,
                                     probabilities=args.model != "svm")
    continuity_value, connected, transitions = continuity(winners, args.raw_zip)
    test_metrics.update({"classifier_route_continuity": continuity_value,
                         "continuity_connected": connected,
                         "continuity_transitions": transitions,
                         "fit_seconds": final_fit_seconds,
                         "inference_seconds": inference_seconds})
    result = {
        "model": args.model, "features": FEATURES,
        "protocol": "Shared four-feature candidate CSVs; frozen five trip folds; full train and validation rows; 10 held-out student-test trips.",
        "configuration": model.get_params(), "folds": folds,
        "cv_macro_f1_mean": float(np.mean([f["candidate_macro_f1"] for f in folds])),
        "cv_macro_f1_sd": float(np.std([f["candidate_macro_f1"] for f in folds], ddof=1)),
        "student_test": test_metrics,
    }
    (OUT / f"{args.model}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    winners[["trip_id", "observation_id", "candidate_id", "score"]].to_csv(
        OUT / f"{args.model}_student_test_winners.csv", index=False)
    print("saved", OUT / f"{args.model}.json", flush=True)


if __name__ == "__main__":
    main()
