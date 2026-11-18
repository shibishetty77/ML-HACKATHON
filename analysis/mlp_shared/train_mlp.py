from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DATA = ROOT / "Hyperparameter_shared_pipeline" / "outputs"
OUT = BASE
FEATURES = ["distance_to_segment_m", "projection_fraction",
            "observation_speed_mps", "heading_difference_rad"]
SEED = 42
MAX_EPOCHS = 60
PATIENCE = 12


def fit(x, y, tag, validation=None):
    scaler = StandardScaler()
    xs = scaler.fit_transform(x).astype(np.float32, copy=False)
    xv = scaler.transform(validation[0]).astype(np.float32, copy=False) if validation else None
    model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                          alpha=0.001, batch_size=2048, learning_rate_init=0.001,
                          max_iter=1, shuffle=True, random_state=SEED, tol=0.0001)
    history = []
    best = float("inf")
    stale = 0
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        for epoch in range(1, MAX_EPOCHS + 1):
            if epoch == 1:
                model.partial_fit(xs, y, classes=np.array([0, 1]))
            else:
                model.partial_fit(xs, y)
            row = {"epoch": epoch, "training_loss": float(model.loss_)}
            if validation:
                pv = model.predict_proba(xv)[:, 1]
                row["validation_log_loss"] = float(log_loss(validation[1], pv, labels=[0, 1]))
            history.append(row)
            if model.loss_ < best - 0.0001:
                best = float(model.loss_)
                stale = 0
            else:
                stale += 1
            if epoch == 1 or epoch % 5 == 0:
                print(f"{tag}: epoch {epoch}/{MAX_EPOCHS}, loss={model.loss_:.5f}", flush=True)
            if stale >= PATIENCE:
                break
    elapsed = time.perf_counter() - start
    OUT.joinpath("histories", f"{tag}.csv").write_text(pd.DataFrame(history).to_csv(index=False), encoding="utf-8")
    return scaler, model, history, elapsed


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "candidate_rows": int(len(y)),
        "candidate_accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def rank_metrics(df, p, cov):
    work = df[["trip_id", "observation_id", "candidate_id", "label"]].copy()
    work["probability"] = p
    order = work.groupby(["trip_id", "observation_id"], sort=False)["probability"].idxmax()
    winners = work.loc[order, ["trip_id", "observation_id", "candidate_id", "label"]].copy()
    eligible = cov[cov.is_aligned]
    scored = eligible.merge(winners, on=["trip_id", "observation_id"], how="left")
    covered = scored.candidate_id.notna()
    correct = covered & scored.candidate_id.eq(scored.reference_arc_id)
    return {
        "aligned_observations": int(len(eligible)),
        "candidate_table_observations": int(covered.sum()),
        "candidate_hit_rate_from_coverage": float(eligible.ref_in_candidates.mean()) if len(eligible) else None,
        "top1_accuracy_conditional_on_candidate_table": float(winners.label.mean()) if len(winners) else None,
        "top1_accuracy_end_to_end_over_aligned_observations": float(correct.sum() / len(eligible)) if len(eligible) else None,
        "coverage_ref_in_candidates_count": int(eligible.ref_in_candidates.sum()),
    }


def main():
    for d in ["artifacts", "histories", "predictions"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train_candidates.csv")
    test = pd.read_csv(DATA / "student_test_candidates.csv")
    train_cov = pd.read_csv(DATA / "train_coverage.csv")
    test_cov = pd.read_csv(DATA / "student_test_coverage.csv")
    fold_cfg = json.loads((ROOT / "Hyperparameter_shared_pipeline" /
                           "fold_assignments.json").read_text(encoding="utf-8"))["folds"]
    x = train[FEATURES].to_numpy(np.float64)
    y = train.label.to_numpy(np.int64)
    folds = []
    oof = np.full(len(train), np.nan)
    for fold in range(5):
        spec = fold_cfg[str(fold)]
        val_ids = set(spec["val_trips"])
        trip_strings = train.trip_id.map(lambda z: f"{int(z):08d}")
        vm = trip_strings.isin(val_ids).to_numpy()
        scaler, model, history, elapsed = fit(x[~vm], y[~vm], f"fold_{fold+1}",
                                               validation=(x[vm], y[vm]))
        p = model.predict_proba(scaler.transform(x[vm]))[:, 1]
        oof[vm] = p
        val = train.loc[vm].copy()
        val_cov = train_cov[train_cov.trip_id.isin(val.trip_id.unique())]
        score = metrics(y[vm], p)
        score.update(rank_metrics(val, p, val_cov))
        score.update({"fold": fold + 1, "training_trip_count": len(spec["train_trips"]),
                      "validation_trip_count": len(spec["val_trips"]), "epochs": len(history),
                      "fit_seconds": elapsed})
        folds.append(score)
        pd.DataFrame({"trip_id": train.loc[vm, "trip_id"], "observation_id": train.loc[vm, "observation_id"],
                      "candidate_id": train.loc[vm, "candidate_id"], "label": y[vm], "probability": p}).to_csv(
                          OUT / "predictions" / f"fold_{fold+1}.csv", index=False)
        print(f"FOLD {fold+1}: macroF1={score['macro_f1']:.4f}, conditional Top1={score['top1_accuracy_conditional_on_candidate_table']:.4f}, end-to-end Top1={score['top1_accuracy_end_to_end_over_aligned_observations']:.4f}", flush=True)
    pd.DataFrame(folds).to_csv(OUT / "cross_validation.csv", index=False)
    cv_summary = {}
    for metric in ["candidate_accuracy", "precision", "recall", "macro_f1", "roc_auc", "log_loss",
                   "top1_accuracy_conditional_on_candidate_table", "top1_accuracy_end_to_end_over_aligned_observations"]:
        vals = np.array([f[metric] for f in folds], dtype=float)
        cv_summary[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

    scaler, model, history, final_seconds = fit(x, y, "final")
    xt = test[FEATURES].to_numpy(np.float64)
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        pt = model.predict_proba(scaler.transform(xt))[:, 1]
    infer_seconds = time.perf_counter() - start
    test_score = metrics(test.label.to_numpy(), pt)
    test_score.update(rank_metrics(test, pt, test_cov))
    test[["trip_id", "observation_id", "candidate_id", "label"]].assign(probability=pt).to_csv(
        OUT / "student_test_predictions.csv", index=False)
    joblib.dump({"scaler": scaler, "model": model, "features": FEATURES,
                 "architecture": [4, 32, 16, 1]}, OUT / "artifacts" / "pathsnap_mlp.joblib", compress=3)
    report = {
        "model": "MLPClassifier (4 inputs, hidden layers 32 and 16, sigmoid binary output)",
        "features": FEATURES,
        "training_rows": int(len(train)), "training_trips": int(train.trip_id.nunique()),
        "training_positive_rate": float(y.mean()), "student_test_rows": int(len(test)),
        "student_test_trips": int(test.trip_id.nunique()),
        "cv_strategy": "Provided frozen five folds, grouped by trip; StandardScaler fit on each training fold only.",
        "cross_validation_mean_and_sd": cv_summary, "folds": folds,
        "final_fit_epochs": len(history), "final_fit_seconds": final_seconds,
        "student_test_metrics": test_score,
        "student_test_inference_seconds": infer_seconds,
        "training_settings": {"activation": "relu", "solver": "adam", "alpha": 0.001,
                              "learning_rate_init": 0.001, "batch_size": 2048,
                              "random_state": SEED, "max_epochs": MAX_EPOCHS,
                              "early_stop": f"training loss plateau, {PATIENCE} epochs"},
        "known_data_limitations": [
            "Candidate CSVs contain only observation groups with a positive reference candidate. Groups with candidate negatives but no positive are absent, so candidate-level and conditional Top-1 metrics are conditional on a reference candidate being present.",
            "Coverage num_candidates/has_candidate disagree with the candidate CSV for observations whose reference arc is absent; coverage ref_in_candidates remains the provided measure for candidate hit rate.",
            "Arc endpoint connectivity is not included, so route continuity cannot be computed from this archive alone.",
            "Reference labels are weak pointwise alignments; reported scores measure agreement with these labels."
        ]
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(OUT / "histories" / "final.csv", index=False)
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2), flush=True)
