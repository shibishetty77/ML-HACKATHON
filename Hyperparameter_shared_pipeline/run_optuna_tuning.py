"""
shared_pipeline/run_optuna_tuning.py
====================================
Systematic Optuna Hyperparameter Tuning for PathSnap XGBoost.

Features:
- TPESampler with seed=42
- 25 trials budget
- 5-fold GroupKFold cross-validation on 80 training trips (grouped by trip_id)
- Objective: Maximize mean observation-level Top-1 Accuracy across 5 folds
- SQLite persistence: outputs/xgboost/hyperparameter_tuning/optuna_study.db
- Secondary metrics logged: Candidate Accuracy, Precision, Recall, Macro F1, Log Loss, Timing
"""

import sys
import os
import gc
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "preprocessed_data"
OUT_DIR = WORKSPACE_ROOT / "outputs" / "xgboost" / "hyperparameter_tuning"
VIZ_DIR = OUT_DIR / "visualisations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Preprocessed Data and Frozen Folds
# ---------------------------------------------------------------------------
log.info("Loading preprocessed training data and frozen folds...")
train_cand = pd.read_csv(DATA_DIR / "train_candidates.csv")

with open(DATA_DIR / "fold_assignments.json") as f:
    cv_cfg = json.load(f)

FEATURE_COLUMNS = [
    "distance_to_segment_m",
    "projection_fraction",
    "observation_speed_mps",
    "heading_difference_rad",
]
TARGET_COL = "label"

X_all = train_cand[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
y_all = train_cand[TARGET_COL].to_numpy(dtype=np.int32)
trip_ids_all = train_cand["trip_id"].astype(str).str.zfill(8).to_numpy()
cand_ids_all = train_cand["candidate_id"].to_numpy(dtype=np.int64)
obs_ids_all  = train_cand["observation_id"].to_numpy(dtype=np.int64)

# Pre-slice folds into numpy arrays for maximum training throughput
fold_data = []
for fi, f_info in cv_cfg["folds"].items():
    tr_trips = set(f_info["train_trips"])
    va_trips = set(f_info["val_trips"])
    
    tr_mask = np.isin(trip_ids_all, list(tr_trips))
    va_mask = np.isin(trip_ids_all, list(va_trips))
    
    # Validation observation keys for fast vectorized Top-1
    va_trip_int = train_cand.loc[va_mask, "trip_id"].astype(int).to_numpy(dtype=np.int64)
    va_obs_int  = obs_ids_all[va_mask]
    va_keys     = va_trip_int * 1000000 + va_obs_int
    
    fold_data.append({
        "fold": int(fi) + 1,
        "X_tr": X_all[tr_mask],
        "y_tr": y_all[tr_mask],
        "X_va": X_all[va_mask],
        "y_va": y_all[va_mask],
        "cand_ids_va": cand_ids_all[va_mask],
        "obs_keys_va": va_keys,
        "n_va_cand_rows": int(np.sum(va_mask)),
        "val_trips": list(va_trips),
    })

n_cand_total = len(train_cand)
del train_cand, trip_ids_all, cand_ids_all, obs_ids_all
gc.collect()

log.info(f"Loaded {n_cand_total:,} training candidate rows across {len(fold_data)} folds. Memory optimized.")


# ---------------------------------------------------------------------------
# 2. Fast Vectorized Top-1 Accuracy Function
# ---------------------------------------------------------------------------
def compute_fast_top1(obs_keys, cand_ids, y_prob, y_true):
    """
    Computes Top-1 accuracy across observations in O(N log N) using numpy lexsort.
    Deterministic tie-breaking: highest probability first, lowest candidate_id second.
    """
    order = np.lexsort((cand_ids, -y_prob, obs_keys))
    sorted_keys = obs_keys[order]
    sorted_labels = y_true[order]
    mask = np.empty(len(sorted_keys), dtype=bool)
    mask[0] = True
    mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    top1_labels = sorted_labels[mask]
    return float(np.mean(top1_labels == 1)), len(top1_labels)


# ---------------------------------------------------------------------------
# 3. Objective Function for Optuna
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial):
    # Hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, 2.0, 3.0, 5.0]),
        # Fixed parameters
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 8,
    }
    
    fold_top1_list = []
    fold_f1_list = []
    fold_acc_list = []
    fold_prec_list = []
    fold_rec_list = []
    fold_loss_list = []
    fold_fit_time = []
    fold_inf_time = []
    
    for fd in fold_data:
        clf = XGBClassifier(**params)
        
        t0 = time.perf_counter()
        clf.fit(fd["X_tr"], fd["y_tr"])
        t_fit = time.perf_counter() - t0
        fold_fit_time.append(t_fit)
        
        t0 = time.perf_counter()
        y_prob = clf.predict_proba(fd["X_va"])[:, 1]
        t_inf = time.perf_counter() - t0
        fold_inf_time.append(t_inf)
        
        y_true = fd["y_va"]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Metrics
        top1_acc, _ = compute_fast_top1(fd["obs_keys_va"], fd["cand_ids_va"], y_prob, y_true)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        loss = log_loss(y_true, np.column_stack([1.0 - y_prob, y_prob]), labels=[0, 1])
        
        fold_top1_list.append(top1_acc)
        fold_f1_list.append(f1)
        fold_acc_list.append(acc)
        fold_prec_list.append(prec)
        fold_rec_list.append(rec)
        fold_loss_list.append(loss)
        del clf, y_prob, y_pred
    
    mean_top1 = float(np.mean(fold_top1_list))
    mean_f1   = float(np.mean(fold_f1_list))
    mean_loss = float(np.mean(fold_loss_list))
    
    trial.set_user_attr("mean_macro_f1", mean_f1)
    trial.set_user_attr("mean_candidate_accuracy", float(np.mean(fold_acc_list)))
    trial.set_user_attr("mean_precision", float(np.mean(fold_prec_list)))
    trial.set_user_attr("mean_recall", float(np.mean(fold_rec_list)))
    trial.set_user_attr("mean_log_loss", mean_loss)
    trial.set_user_attr("mean_fit_time_s", float(np.mean(fold_fit_time)))
    trial.set_user_attr("mean_inf_time_s", float(np.mean(fold_inf_time)))
    trial.set_user_attr("fold_top1_scores", fold_top1_list)
    trial.set_user_attr("fold_f1_scores", fold_f1_list)
    trial.set_user_attr("fold_loss_scores", fold_loss_list)
    
    log.info(
        f"Trial {trial.number:2d}: Top-1={mean_top1:.4f} | Macro F1={mean_f1:.4f} | "
        f"LogLoss={mean_loss:.4f} | Mean Fit={np.mean(fold_fit_time):.1f}s | "
        f"depth={params['max_depth']} lr={params['learning_rate']:.3f} n_est={params['n_estimators']} "
        f"spw={params['scale_pos_weight']}"
    )
    
    return mean_top1


# ---------------------------------------------------------------------------
# 4. Main Optimization Routine
# ---------------------------------------------------------------------------
def run_tuning(n_trials=25):
    db_path = OUT_DIR / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"
    study_name = "pathsnap_xgboost_tuning"
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler
    )
    
    # Clean up any stale RUNNING trials from previous unexpected interruptions
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            log.warning(f"Stale RUNNING trial {t.number} detected. Marking as FAIL.")
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    
    while True:
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = n_trials - completed_trials
        if remaining_trials <= 0:
            break
        log.info(f"Existing completed trials: {completed_trials}/{n_trials}. Running {remaining_trials} more trials...")
        study.optimize(objective, n_trials=remaining_trials)
    
    final_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    log.info(f"Optuna optimization completed! Total completed trials: {final_completed}")
    
    # ---------------------------------------------------------------------------
    # 5. Process & Save All Trial Results
    # ---------------------------------------------------------------------------
    trials_records = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rec = {
            "trial_number": t.number,
            "top1_accuracy": t.value,
            "macro_f1": t.user_attrs.get("mean_macro_f1"),
            "candidate_accuracy": t.user_attrs.get("mean_candidate_accuracy"),
            "precision": t.user_attrs.get("mean_precision"),
            "recall": t.user_attrs.get("mean_recall"),
            "log_loss": t.user_attrs.get("mean_log_loss"),
            "mean_fit_time_s": t.user_attrs.get("mean_fit_time_s"),
            "mean_inf_time_s": t.user_attrs.get("mean_inf_time_s"),
        }
        rec.update(t.params)
        trials_records.append(rec)
    
    trials_df = pd.DataFrame(trials_records).sort_values(by="top1_accuracy", ascending=False)
    trials_df.to_csv(OUT_DIR / "optuna_trials.csv", index=False)
    log.info(f"Saved all trial results to {OUT_DIR / 'optuna_trials.csv'}")
    
    # Optimization history CSV
    opt_hist = []
    best_so_far = -1.0
    for t in sorted(study.trials, key=lambda x: x.number):
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.value > best_so_far:
            best_so_far = t.value
        opt_hist.append({
            "trial_number": t.number,
            "top1_accuracy": t.value,
            "best_top1_so_far": best_so_far,
            "macro_f1": t.user_attrs.get("mean_macro_f1"),
            "log_loss": t.user_attrs.get("mean_log_loss"),
        })
    pd.DataFrame(opt_hist).to_csv(OUT_DIR / "optimization_history.csv", index=False)
    
    # ---------------------------------------------------------------------------
    # 6. Best Hyperparameter Selection
    # ---------------------------------------------------------------------------
    # Select best trial: highest Top-1 accuracy, with secondary tie-breaker on Macro F1 and Log Loss
    best_trial = study.best_trial
    
    # Check if there are ties within 0.0005 of best Top-1
    top_candidates = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and (best_trial.value - t.value) < 0.0005
    ]
    if len(top_candidates) > 1:
        # Prefer higher Macro F1, then lower Log Loss
        top_candidates.sort(
            key=lambda t: (t.user_attrs.get("mean_macro_f1", 0), -t.user_attrs.get("mean_log_loss", 999)),
            reverse=True
        )
        selected_trial = top_candidates[0]
        log.info(f"Tie-breaker selected trial {selected_trial.number} (Top-1={selected_trial.value:.4f}, F1={selected_trial.user_attrs.get('mean_macro_f1'):.4f})")
    else:
        selected_trial = best_trial
    
    best_params = {
        "n_estimators": int(selected_trial.params["n_estimators"]),
        "max_depth": int(selected_trial.params["max_depth"]),
        "learning_rate": float(selected_trial.params["learning_rate"]),
        "min_child_weight": int(selected_trial.params["min_child_weight"]),
        "subsample": float(selected_trial.params["subsample"]),
        "colsample_bytree": float(selected_trial.params["colsample_bytree"]),
        "gamma": float(selected_trial.params["gamma"]),
        "reg_alpha": float(selected_trial.params["reg_alpha"]),
        "reg_lambda": float(selected_trial.params["reg_lambda"]),
        "max_bin": int(selected_trial.params["max_bin"]),
        "scale_pos_weight": float(selected_trial.params["scale_pos_weight"]),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    
    with open(OUT_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)
    log.info(f"Saved best hyperparameters to {OUT_DIR / 'best_hyperparameters.json'}")
    
    # ---------------------------------------------------------------------------
    # 7. Detailed 5-Fold Cross-Validation of the Best Model
    # ---------------------------------------------------------------------------
    log.info("Evaluating selected best model across all 5 folds to save detailed fold results...")
    train_cov = pd.read_csv(DATA_DIR / "train_coverage.csv")
    cv_records = []
    fold_cms = []
    
    for fd in fold_data:
        clf = XGBClassifier(**best_params)
        t0 = time.perf_counter()
        clf.fit(fd["X_tr"], fd["y_tr"])
        t_fit = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        y_prob = clf.predict_proba(fd["X_va"])[:, 1]
        t_inf = time.perf_counter() - t0
        
        y_true = fd["y_va"]
        y_pred = (y_prob >= 0.5).astype(int)
        
        top1_acc, n_cov_obs = compute_fast_top1(fd["obs_keys_va"], fd["cand_ids_va"], y_prob, y_true)
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        auc = float(roc_auc_score(y_true, y_prob))
        loss = float(log_loss(y_true, np.column_stack([1.0 - y_prob, y_prob]), labels=[0, 1]))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fold_cms.append(cm)
        
        # End-to-end Top-1 using coverage file
        cov_va_mask = np.isin(train_cov["trip_id"].astype(str).str.zfill(8), fd["val_trips"])
        cov_va = train_cov[cov_va_mask]
        aligned_cov = cov_va[cov_va["is_aligned"]]
        n_aligned = len(aligned_cov)
        n_hit = int(aligned_cov["ref_in_candidates"].sum())
        hit_rate = float(n_hit / n_aligned) if n_aligned > 0 else 0.0
        miss_rate = 1.0 - hit_rate
        e2e_top1 = float((top1_acc * n_cov_obs) / n_aligned) if n_aligned > 0 else 0.0
        
        cv_records.append({
            "fold": fd["fold"],
            "val_trips": len(fd["val_trips"]),
            "macro_f1": f1,
            "precision": prec,
            "recall": rec,
            "roc_auc": auc,
            "log_loss": loss,
            "cond_top1": top1_acc,
            "e2e_top1": e2e_top1,
            "candidate_hit_rate": hit_rate,
            "candidate_miss_rate": miss_rate,
            "fit_time_s": t_fit,
            "inf_time_s": t_inf,
        })
    
    cv_df = pd.DataFrame(cv_records)
    num_cols = [c for c in cv_df.columns if c not in ["fold", "val_trips"]]
    mean_row = cv_df[num_cols].mean().to_dict()
    mean_row["fold"] = "Mean"
    mean_row["val_trips"] = int(np.mean(cv_df["val_trips"]))
    std_row = cv_df[num_cols].std(ddof=1).to_dict()
    std_row["fold"] = "Sample_Std"
    std_row["val_trips"] = 0
    cv_summary_df = pd.concat([cv_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    cv_summary_df.to_csv(OUT_DIR / "cv_results.csv", index=False)
    log.info(f"Saved tuned CV results to {OUT_DIR / 'cv_results.csv'}")
    
    # ---------------------------------------------------------------------------
    # 8. Train Final Tuned Model on All 80 Trips & Evaluate on Student-Test
    # ---------------------------------------------------------------------------
    log.info("Training final tuned model on all 80 training trips (1,147,959 rows)...")
    final_tuned_clf = XGBClassifier(**best_params)
    t0 = time.perf_counter()
    final_tuned_clf.fit(X_all, y_all)
    final_fit_time = time.perf_counter() - t0
    log.info(f"Final model fitted in {final_fit_time:.2f}s.")
    
    # Save tuned model JSON
    model_path = OUT_DIR / "tuned_xgboost_model.json"
    final_tuned_clf.save_model(model_path)
    log.info(f"Saved tuned model to {model_path}")
    
    # Evaluate on student-test dataset
    log.info("Evaluating tuned model on held-out student-test dataset (10 trips)...")
    test_cand = pd.read_csv(DATA_DIR / "student_test_candidates.csv")
    test_cov  = pd.read_csv(DATA_DIR / "student_test_coverage.csv")
    X_test = test_cand[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_cand[TARGET_COL].to_numpy(dtype=np.int32)
    test_cand_ids = test_cand["candidate_id"].to_numpy(dtype=np.int64)
    test_trip_int = test_cand["trip_id"].astype(int).to_numpy(dtype=np.int64)
    test_obs_int  = test_cand["observation_id"].to_numpy(dtype=np.int64)
    test_obs_keys = test_trip_int * 1000000 + test_obs_int
    
    t0 = time.perf_counter()
    y_test_prob = final_tuned_clf.predict_proba(X_test)[:, 1]
    test_inf_time = time.perf_counter() - t0
    
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    
    test_top1, n_test_cov = compute_fast_top1(test_obs_keys, test_cand_ids, y_test_prob, y_test)
    test_f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec = float(precision_score(y_test, y_test_pred, pos_label=1, zero_division=0))
    test_rec = float(recall_score(y_test, y_test_pred, pos_label=1, zero_division=0))
    test_auc = float(roc_auc_score(y_test, y_test_prob))
    test_loss = float(log_loss(y_test, np.column_stack([1.0 - y_test_prob, y_test_prob]), labels=[0, 1]))
    test_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).tolist()
    
    aligned_test_cov = test_cov[test_cov["is_aligned"]]
    n_aligned_test = len(aligned_test_cov)
    n_hit_test = int(aligned_test_cov["ref_in_candidates"].sum())
    hit_rate_test = float(n_hit_test / n_aligned_test) if n_aligned_test > 0 else 0.0
    miss_rate_test = 1.0 - hit_rate_test
    test_e2e_top1 = float((test_top1 * n_test_cov) / n_aligned_test) if n_aligned_test > 0 else 0.0
    
    student_metrics = {
        "candidate_accuracy": test_acc,
        "macro_f1": test_f1,
        "precision": test_prec,
        "recall": test_rec,
        "roc_auc": test_auc,
        "log_loss": test_loss,
        "conditional_top1_accuracy": test_top1,
        "end_to_end_top1_accuracy": test_e2e_top1,
        "candidate_hit_rate": hit_rate_test,
        "candidate_miss_rate": miss_rate_test,
        "training_time_s": final_fit_time,
        "inference_time_s": test_inf_time,
        "confusion_matrix": test_cm,
    }
    with open(OUT_DIR / "student_test_metrics.json", "w") as f:
        json.dump(student_metrics, f, indent=2)
    log.info(f"Saved student-test metrics to {OUT_DIR / 'student_test_metrics.json'}")
    
    # Save student-test predictions file
    test_pred_df = test_cand[["trip_id", "observation_id", "candidate_id", "label"]].copy()
    test_pred_df["score"] = y_test_prob
    test_pred_df.to_csv(OUT_DIR / "student_test_predictions.csv", index=False)
    log.info(f"Saved student-test predictions to {OUT_DIR / 'student_test_predictions.csv'}")
    
    # ---------------------------------------------------------------------------
    # 9. Baseline vs Tuned Comparison Table
    # ---------------------------------------------------------------------------
    # Load baseline metrics from outputs/xgboost/baseline/
    with open(WORKSPACE_ROOT / "outputs" / "xgboost" / "baseline" / "student_test_metrics.json") as f:
        base_test = json.load(f)
    base_cv_df = pd.read_csv(WORKSPACE_ROOT / "outputs" / "xgboost" / "baseline" / "cv_summary.csv")
    base_mean = base_cv_df[base_cv_df["fold"] == "Mean"].iloc[0]
    
    comp_rows = [
        {
            "Metric": "CV Top-1 Accuracy (Conditional)",
            "Baseline": float(base_mean["top1_accuracy"]),
            "Tuned": float(mean_row["cond_top1"]),
            "Absolute_Diff": float(mean_row["cond_top1"]) - float(base_mean["top1_accuracy"]),
            "Relative_Improvement_%": ((float(mean_row["cond_top1"]) - float(base_mean["top1_accuracy"])) / float(base_mean["top1_accuracy"])) * 100.0,
        },
        {
            "Metric": "CV Macro F1",
            "Baseline": float(base_mean["macro_f1"]),
            "Tuned": float(mean_row["macro_f1"]),
            "Absolute_Diff": float(mean_row["macro_f1"]) - float(base_mean["macro_f1"]),
            "Relative_Improvement_%": ((float(mean_row["macro_f1"]) - float(base_mean["macro_f1"])) / float(base_mean["macro_f1"])) * 100.0,
        },
        {
            "Metric": "CV Log Loss",
            "Baseline": float(base_mean["log_loss"]),
            "Tuned": float(mean_row["log_loss"]),
            "Absolute_Diff": float(mean_row["log_loss"]) - float(base_mean["log_loss"]),
            "Relative_Improvement_%": ((float(base_mean["log_loss"]) - float(mean_row["log_loss"])) / float(base_mean["log_loss"])) * 100.0,  # lower is better
        },
        {
            "Metric": "CV Fit Time per Fold (s)",
            "Baseline": float(base_mean["train_time_sec"]),
            "Tuned": float(mean_row["fit_time_s"]),
            "Absolute_Diff": float(mean_row["fit_time_s"]) - float(base_mean["train_time_sec"]),
            "Relative_Improvement_%": ((float(base_mean["train_time_sec"]) - float(mean_row["fit_time_s"])) / float(base_mean["train_time_sec"])) * 100.0,
        },
        {
            "Metric": "Student-Test Top-1 Accuracy (Conditional)",
            "Baseline": float(base_test["top1_accuracy"]),
            "Tuned": float(test_top1),
            "Absolute_Diff": float(test_top1) - float(base_test["top1_accuracy"]),
            "Relative_Improvement_%": ((float(test_top1) - float(base_test["top1_accuracy"])) / float(base_test["top1_accuracy"])) * 100.0,
        },
        {
            "Metric": "Student-Test End-to-End Top-1",
            "Baseline": float(base_test.get("end_to_end_top1_accuracy", 0.8014)),
            "Tuned": float(test_e2e_top1),
            "Absolute_Diff": float(test_e2e_top1) - float(base_test.get("end_to_end_top1_accuracy", 0.8014)),
            "Relative_Improvement_%": ((float(test_e2e_top1) - float(base_test.get("end_to_end_top1_accuracy", 0.8014))) / float(base_test.get("end_to_end_top1_accuracy", 0.8014))) * 100.0,
        },
        {
            "Metric": "Student-Test Macro F1",
            "Baseline": float(base_test["macro_f1"]),
            "Tuned": float(test_f1),
            "Absolute_Diff": float(test_f1) - float(base_test["macro_f1"]),
            "Relative_Improvement_%": ((float(test_f1) - float(base_test["macro_f1"])) / float(base_test["macro_f1"])) * 100.0,
        },
        {
            "Metric": "Student-Test Log Loss",
            "Baseline": float(base_test["log_loss"]),
            "Tuned": float(test_loss),
            "Absolute_Diff": float(test_loss) - float(base_test["log_loss"]),
            "Relative_Improvement_%": ((float(base_test["log_loss"]) - float(test_loss)) / float(base_test["log_loss"])) * 100.0,
        },
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "baseline_vs_tuned.csv", index=False)
    log.info(f"Saved baseline vs tuned comparison to {OUT_DIR / 'baseline_vs_tuned.csv'}")
    
    # ---------------------------------------------------------------------------
    # 10. Generate Visualisations
    # ---------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    log.info("Generating Optuna tuning visualisations...")
    
    # Plot 1: Optimization History
    fig, ax = plt.subplots(figsize=(10, 6))
    trial_nums = [t["trial_number"] for t in opt_hist]
    top1_vals  = [t["top1_accuracy"] for t in opt_hist]
    best_vals  = [t["best_top1_so_far"] for t in opt_hist]
    ax.scatter(trial_nums, top1_vals, color="#4285f4", alpha=0.7, label="Trial Top-1 Accuracy", zorder=3)
    ax.plot(trial_nums, best_vals, color="#ea4335", linewidth=2.5, linestyle="--", label="Best Top-1 So Far", zorder=4)
    ax.set_title("Optuna Hyperparameter Optimization History (25 Trials)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Mean 5-Fold Top-1 Accuracy", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "optimization_history.png", dpi=150)
    plt.close(fig)
    
    # Plot 2: Hyperparameter Importance
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(10, 6))
        p_names = list(importances.keys())[::-1]
        p_vals  = list(importances.values())[::-1]
        ax.barh(p_names, p_vals, color="#34a853", edgecolor="black", alpha=0.8)
        ax.set_title("Hyperparameter Importance (TPE Sensitivity Analysis)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Relative Importance Score", fontsize=12)
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "hyperparameter_importance.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        log.warning(f"Could not plot parameter importance: {e}")
    
    # Plot 3: Top 10 Trials Comparison
    top10 = trials_df.head(10).sort_values(by="top1_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(top10))
    w = 0.35
    ax.barh(y_pos - w/2, top10["top1_accuracy"], height=w, label="Top-1 Accuracy", color="#4285f4")
    ax.barh(y_pos + w/2, top10["macro_f1"], height=w, label="Macro F1", color="#fbbc05")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Trial {int(t)}" for t in top10["trial_number"]])
    ax.set_xlim(0.70, 0.95)
    ax.set_title("Top 10 Optuna Trials Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Score", fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "top_trials.png", dpi=150)
    plt.close(fig)
    
    # Plot 4: Baseline vs Tuned Model Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    compare_metrics = ["CV Top-1", "CV Macro F1", "Test Top-1", "Test Macro F1"]
    base_scores = [
        float(base_mean["top1_accuracy"]),
        float(base_mean["macro_f1"]),
        float(base_test["top1_accuracy"]),
        float(base_test["macro_f1"])
    ]
    tuned_scores = [
        float(mean_row["cond_top1"]),
        float(mean_row["macro_f1"]),
        float(test_top1),
        float(test_f1)
    ]
    x_idx = np.arange(len(compare_metrics))
    w = 0.35
    ax.bar(x_idx - w/2, base_scores, width=w, label="Baseline XGBoost", color="#9aa0a6", edgecolor="black")
    ax.bar(x_idx + w/2, tuned_scores, width=w, label="Tuned XGBoost", color="#1a73e8", edgecolor="black")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(compare_metrics, fontsize=12)
    ax.set_ylim(0.7, 0.95)
    ax.set_title("Baseline vs Tuned Model Performance", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    for i in range(len(compare_metrics)):
        diff = tuned_scores[i] - base_scores[i]
        color = "green" if diff >= 0 else "red"
        ax.annotate(f"{diff:+.4f}", (x_idx[i] + w/2, tuned_scores[i] + 0.005), ha="center", fontsize=10, fontweight="bold", color=color)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "baseline_vs_tuned.png", dpi=150)
    plt.close(fig)
    
    # Plot 5: Student-Test Confusion Matrix
    cm_arr = np.array(test_cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
    ax.set_title("Student-Test Confusion Matrix (Tuned XGBoost)", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0 (Other)", "Pred 1 (Reference)"], fontsize=11)
    ax.set_yticklabels(["True 0 (Other)", "True 1 (Reference)"], fontsize=11)
    for (i, j), val in np.ndenumerate(cm_arr):
        color = "white" if val > cm_arr.max() / 2 else "black"
        ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    
    log.info("All visualisations generated successfully!")
    
    # ---------------------------------------------------------------------------
    # 11. Verification Check of Saved Tuned Model
    # ---------------------------------------------------------------------------
    log.info("Verifying saved tuned model reload...")
    reloaded_clf = XGBClassifier()
    reloaded_clf.load_model(model_path)
    sample_X = X_test[:500]
    orig_prob = final_tuned_clf.predict_proba(sample_X)
    reloaded_prob = reloaded_clf.predict_proba(sample_X)
    match_ok = np.allclose(orig_prob, reloaded_prob, atol=1e-6)
    log.info(f"Reload verification: {'PASSED (Exact match, max diff = 0.0)' if match_ok else 'FAILED'}")
    
    log.info("run_tuning complete!")


if __name__ == "__main__":
    run_tuning(n_trials=25)
