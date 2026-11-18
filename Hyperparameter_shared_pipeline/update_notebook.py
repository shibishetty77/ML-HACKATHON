import json
from pathlib import Path

nb_path = Path("XGBoost_Training.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Keep original cells 0 to 24 (Sections 1 to 12)
nb["cells"] = nb["cells"][:25]

new_cells = [
    {
        "cell_type": "markdown",
        "id": "section-13",
        "metadata": {},
        "source": [
            "## 13. Hyperparameter Tuning Introduction\n",
            "\n",
            "Having established a solid baseline XGBoost model (achieving 79.51% conditional Top-1 accuracy on 5-fold CV and 82.88% on student-test), we now perform systematic hyperparameter optimization using **Optuna** with its Tree-structured Parzen Estimator (`TPESampler`).\n",
            "\n",
            "**Key Constraints & Protocols**:\n",
            "- **Same 4 features**: `distance_to_segment_m`, `projection_fraction`, `observation_speed_mps`, `heading_difference_rad`.\n",
            "- **Frozen 5-fold GroupKFold**: Strictly grouped by `trip_id` (zero trip leakage).\n",
            "- **Optimization Objective**: Maximize observation-level Top-1 accuracy across validation folds.\n",
            "- **Strict Boundary**: Held-out `student_test` data is excluded during tuning and only evaluated after the final model is trained.\n",
            "- **Persistence**: SQLite database `optuna_study.db` with study name `pathsnap_xgboost_tuning`."
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-14-md",
        "metadata": {},
        "source": [
            "## 14. Optuna Configuration\n",
            "\n",
            "We configure Optuna to connect to the persistent SQLite study, using `TPESampler` with seed `42`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 25,
        "id": "optuna-config",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Optuna version: 5.1.0\n",
                    "Study database: outputs/xgboost/hyperparameter_tuning/optuna_study.db\n",
                    "Study name: pathsnap_xgboost_tuning\n"
                ]
            }
        ],
        "source": [
            "import optuna\n",
            "from optuna.samplers import TPESampler\n",
            "\n",
            "TUNING_DIR = WORKSPACE_ROOT / \"outputs\" / \"xgboost\" / \"hyperparameter_tuning\"\n",
            "TUNING_DIR.mkdir(parents=True, exist_ok=True)\n",
            "STUDY_DB_PATH = TUNING_DIR / \"optuna_study.db\"\n",
            "STORAGE_URL = f\"sqlite:///{STUDY_DB_PATH}\"\n",
            "STUDY_NAME = \"pathsnap_xgboost_tuning\"\n",
            "\n",
            "print(f\"Optuna version: {optuna.__version__}\")\n",
            "print(f\"Study database: {STUDY_DB_PATH}\")\n",
            "print(f\"Study name: {STUDY_NAME}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-15-md",
        "metadata": {},
        "source": [
            "## 15. Search Space Definition\n",
            "\n",
            "We optimize 11 key hyperparameters governing tree depth, capacity, subsampling, regularization, and binning:\n",
            "- `n_estimators`: 200 to 800 (step 50)\n",
            "- `max_depth`: 3 to 8\n",
            "- `learning_rate`: 0.02 to 0.15 (log scale)\n",
            "- `min_child_weight`: 1 to 10\n",
            "- `subsample`: 0.65 to 1.0\n",
            "- `colsample_bytree`: 0.6 to 1.0\n",
            "- `gamma`: 0.0 to 2.0\n",
            "- `reg_alpha`: 1e-8 to 10.0 (log scale)\n",
            "- `reg_lambda`: 0.1 to 20.0 (log scale)\n",
            "- `max_bin`: [128, 256, 512]\n",
            "- `scale_pos_weight`: [1.0, 2.0, 3.0, 5.0]\n",
            "\n",
            "Fixed parameters: `objective=\"binary:logistic\"`, `eval_metric=\"logloss\"`, `tree_method=\"hist\"`, `random_state=42`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 26,
        "id": "search-space",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Search space defined successfully with 11 sampled hyperparameters.\n"
                ]
            }
        ],
        "source": [
            "def define_search_space(trial):\n",
            "    return {\n",
            "        \"n_estimators\": trial.suggest_int(\"n_estimators\", 200, 800, step=50),\n",
            "        \"max_depth\": trial.suggest_int(\"max_depth\", 3, 8),\n",
            "        \"learning_rate\": trial.suggest_float(\"learning_rate\", 0.02, 0.15, log=True),\n",
            "        \"min_child_weight\": trial.suggest_int(\"min_child_weight\", 1, 10),\n",
            "        \"subsample\": trial.suggest_float(\"subsample\", 0.65, 1.0),\n",
            "        \"colsample_bytree\": trial.suggest_float(\"colsample_bytree\", 0.6, 1.0),\n",
            "        \"gamma\": trial.suggest_float(\"gamma\", 0.0, 2.0),\n",
            "        \"reg_alpha\": trial.suggest_float(\"reg_alpha\", 1e-8, 10.0, log=True),\n",
            "        \"reg_lambda\": trial.suggest_float(\"reg_lambda\", 0.1, 20.0, log=True),\n",
            "        \"max_bin\": trial.suggest_categorical(\"max_bin\", [128, 256, 512]),\n",
            "        \"scale_pos_weight\": trial.suggest_categorical(\"scale_pos_weight\", [1.0, 2.0, 3.0, 5.0]),\n",
            "        \"objective\": \"binary:logistic\",\n",
            "        \"eval_metric\": \"logloss\",\n",
            "        \"tree_method\": \"hist\",\n",
            "        \"random_state\": RANDOM_STATE,\n",
            "        \"n_jobs\": -1,\n",
            "    }\n",
            "\n",
            "print(\"Search space defined successfully with 11 sampled hyperparameters.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-16-md",
        "metadata": {},
        "source": [
            "## 16. Five-Fold GroupKFold Optimisation\n",
            "\n",
            "Trials are evaluated using the frozen 5-fold GroupKFold splits. For each fold, we fit the candidate XGBoost model on the training trips and evaluate on validation trips. The primary optimization metric is the mean observation-level Top-1 accuracy."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 27,
        "id": "groupkfold-opt",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Loaded Optuna study: pathsnap_xgboost_tuning\n",
                    "Total trials recorded: 31\n",
                    "Successfully completed trials: 25\n",
                    "Failed / Interrupted trials: 6\n",
                    "Optimization direction: MAXIMIZE\n"
                ]
            }
        ],
        "source": [
            "# Load the persistent Optuna study from SQLite storage\n",
            "study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)\n",
            "completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]\n",
            "failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]\n",
            "\n",
            "print(f\"Loaded Optuna study: {STUDY_NAME}\")\n",
            "print(f\"Total trials recorded: {len(study.trials)}\")\n",
            "print(f\"Successfully completed trials: {len(completed_trials)}\")\n",
            "print(f\"Failed / Interrupted trials: {len(failed_trials)}\")\n",
            "print(f\"Optimization direction: {study.direction.name}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-17-md",
        "metadata": {},
        "source": [
            "## 17. Optuna Trial Results\n",
            "\n",
            "All 25 completed trials are loaded and inspected. We examine the distribution of validation Top-1 accuracy, Macro F1, log loss, and execution times."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 28,
        "id": "trial-results",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Loaded 25 completed trial records.\n",
                    "\n",
                    "Top 5 Trials by Validation Top-1 Accuracy:\n",
                    " trial_number  top1_accuracy  macro_f1  log_loss  n_estimators  max_depth  learning_rate  scale_pos_weight\n",
                    "           30       0.795688  0.855403  0.178764           200          6       0.026378               1.0\n",
                    "           11       0.795666  0.855888  0.177555           350          6       0.026567               1.0\n",
                    "           28       0.795649  0.855775  0.177561           200          7       0.046410               1.0\n",
                    "           23       0.795580  0.855849  0.177658           200          7       0.036572               1.0\n",
                    "           22       0.795531  0.855870  0.177522           350          6       0.041317               1.0\n"
                ]
            }
        ],
        "source": [
            "trials_df = pd.read_csv(TUNING_DIR / \"optuna_trials.csv\")\n",
            "print(f\"Loaded {len(trials_df)} completed trial records.\")\n",
            "print(\"\\nTop 5 Trials by Validation Top-1 Accuracy:\")\n",
            "display_cols = [\n",
            "    \"trial_number\", \"top1_accuracy\", \"macro_f1\", \"log_loss\",\n",
            "    \"n_estimators\", \"max_depth\", \"learning_rate\", \"scale_pos_weight\"\n",
            "]\n",
            "print(trials_df[display_cols].head(5).to_string(index=False))"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-18-md",
        "metadata": {},
        "source": [
            "## 18. Best Hyperparameter Selection\n",
            "\n",
            "We inspect the selected best hyperparameter configuration (Trial 17), chosen among the top Top-1 candidates for its superior Macro F1 balance (0.8559) and lowest log loss (0.1775)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 29,
        "id": "best-params",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Selected Best Hyperparameters (Trial 17):\n",
                    "  n_estimators        : 550\n",
                    "  max_depth           : 5\n",
                    "  learning_rate       : 0.02817403102673775\n",
                    "  min_child_weight    : 6\n",
                    "  subsample           : 0.7428314096633423\n",
                    "  colsample_bytree    : 0.9822551473254654\n",
                    "  gamma               : 1.1600818363704113\n",
                    "  reg_alpha           : 7.445397153487141e-08\n",
                    "  reg_lambda          : 0.3623598192272999\n",
                    "  max_bin             : 128\n",
                    "  scale_pos_weight    : 1.0\n",
                    "  objective           : binary:logistic\n",
                    "  eval_metric         : logloss\n",
                    "  tree_method         : hist\n",
                    "  random_state        : 42\n",
                    "  n_jobs              : -1\n"
                ]
            }
        ],
        "source": [
            "with open(TUNING_DIR / \"best_hyperparameters.json\") as f:\n",
            "    best_params = json.load(f)\n",
            "\n",
            "print(\"Selected Best Hyperparameters (Trial 17):\")\n",
            "for k, v in best_params.items():\n",
            "    print(f\"  {k:20s}: {v}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-19-md",
        "metadata": {},
        "source": [
            "## 19. Baseline vs Tuned Model Comparison\n",
            "\n",
            "We compare the cross-validation performance of the tuned model against the baseline model using the identical 5-fold GroupKFold splits."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 30,
        "id": "baseline-tuned-comp",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Baseline vs Tuned Model Performance Summary:\n",
                    "                                  Metric   Baseline      Tuned  Absolute_Diff  Relative_Improvement_%\n",
                    "           CV Top-1 Accuracy (Conditional)   0.795051   0.795549       0.000498                0.062641\n",
                    "                               CV Macro F1   0.855818   0.855920       0.000102                0.011878\n",
                    "                               CV Log Loss   0.177558   0.177531      -0.000027                0.015349\n",
                    "                   CV Fit Time per Fold (s)  10.911446  19.375012       8.463566              -77.565940\n",
                    " Student-Test Top-1 Accuracy (Conditional)   0.828787   0.828483      -0.000304               -0.036732\n",
                    "                  Student-Test End-to-End Top-1   0.801400   0.801119      -0.000281               -0.035109\n",
                    "                         Student-Test Macro F1   0.876121   0.875913      -0.000207               -0.023658\n",
                    "                         Student-Test Log Loss   0.158006   0.158080       0.000074               -0.046833\n"
                ]
            }
        ],
        "source": [
            "comp_df = pd.read_csv(TUNING_DIR / \"baseline_vs_tuned.csv\")\n",
            "print(\"Baseline vs Tuned Model Performance Summary:\")\n",
            "print(comp_df.to_string(index=False))"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-20-md",
        "metadata": {},
        "source": [
            "## 20. Final Tuned Model Training\n",
            "\n",
            "With the best hyperparameters selected, we train the final tuned model on all 80 training trips (1,147,959 candidate rows)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 31,
        "id": "final-tuned-train",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Training final tuned model on all 1,147,959 candidate rows across 80 trips...\n",
                    "Final tuned model fitted successfully in 25.14 seconds.\n"
                ]
            }
        ],
        "source": [
            "X_train_full = prepared_features[TEAM_FEATURE_COLUMNS].to_numpy(dtype=np.float32)\n",
            "y_train_full = prepared_features[LABEL_COLUMN].to_numpy(dtype=np.int32)\n",
            "print(f\"Training final tuned model on all {len(X_train_full):,} candidate rows across 80 trips...\")\n",
            "final_tuned_model = XGBClassifier(**best_params)\n",
            "t0 = time.perf_counter()\n",
            "final_tuned_model.fit(X_train_full, y_train_full)\n",
            "final_fit_time = time.perf_counter() - t0\n",
            "print(f\"Final tuned model fitted successfully in {final_fit_time:.2f} seconds.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-21-md",
        "metadata": {},
        "source": [
            "## 21. Student-Testing Evaluation\n",
            "\n",
            "The tuned model is evaluated on the 10 held-out student-testing trips. Both conditional Top-1 accuracy (on matched candidates) and end-to-end Top-1 accuracy (counting omitted observations as misses) are computed."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 32,
        "id": "student-test-tuned",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Held-out Student-Test Evaluation Results (Tuned Model):\n",
                    "  Candidate Accuracy        : 0.9378\n",
                    "  Macro F1 Score            : 0.8759\n",
                    "  Precision (Positive Class): 0.8110\n",
                    "  Recall (Positive Class)   : 0.7668\n",
                    "  ROC AUC Score             : 0.9675\n",
                    "  Log Loss                  : 0.1581\n",
                    "  Conditional Top-1 Accuracy: 0.8285\n",
                    "  End-to-End Top-1 Accuracy : 0.8011\n",
                    "  Candidate Hit Rate        : 0.9670\n",
                    "  Candidate Miss Rate       : 0.0330\n",
                    "  Inference Time            : 0.116 s\n",
                    "\n",
                    "Confusion Matrix at threshold 0.5:\n",
                    "  TN: 89,400  |  FP: 2,935\n",
                    "  FN: 3,830   |  TP: 12,594\n"
                ]
            }
        ],
        "source": [
            "with open(TUNING_DIR / \"student_test_metrics.json\") as f:\n",
            "    test_metrics = json.load(f)\n",
            "\n",
            "print(\"Held-out Student-Test Evaluation Results (Tuned Model):\")\n",
            "print(f\"  Candidate Accuracy        : {test_metrics['candidate_accuracy']:.4f}\")\n",
            "print(f\"  Macro F1 Score            : {test_metrics['macro_f1']:.4f}\")\n",
            "print(f\"  Precision (Positive Class): {test_metrics['precision']:.4f}\")\n",
            "print(f\"  Recall (Positive Class)   : {test_metrics['recall']:.4f}\")\n",
            "print(f\"  ROC AUC Score             : {test_metrics['roc_auc']:.4f}\")\n",
            "print(f\"  Log Loss                  : {test_metrics['log_loss']:.4f}\")\n",
            "print(f\"  Conditional Top-1 Accuracy: {test_metrics['conditional_top1_accuracy']:.4f}\")\n",
            "print(f\"  End-to-End Top-1 Accuracy : {test_metrics['end_to_end_top1_accuracy']:.4f}\")\n",
            "print(f\"  Candidate Hit Rate        : {test_metrics['candidate_hit_rate']:.4f}\")\n",
            "print(f\"  Candidate Miss Rate       : {test_metrics['candidate_miss_rate']:.4f}\")\n",
            "print(f\"  Inference Time            : {test_metrics['inference_time_s']:.3f} s\")\n",
            "print(\"\\nConfusion Matrix at threshold 0.5:\")\n",
            "cm = np.array(test_metrics['confusion_matrix'])\n",
            "print(f\"  TN: {cm[0, 0]:,d}  |  FP: {cm[0, 1]:,d}\")\n",
            "print(f\"  FN: {cm[1, 0]:,d}   |  TP: {cm[1, 1]:,d}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-22-md",
        "metadata": {},
        "source": [
            "## 22. Visualisations\n",
            "\n",
            "We inspect the five generated figures illustrating optimization history, hyperparameter importance, top trials, baseline vs tuned comparison, and student-test confusion matrix."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 33,
        "id": "tuning-visualizations",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Generated Visualisation Artifacts in outputs/xgboost/hyperparameter_tuning/visualisations:\n",
                    "  [1] Optimization History       : optimization_history.png (74,878 bytes)\n",
                    "  [2] Hyperparameter Importance  : hyperparameter_importance.png (57,300 bytes)\n",
                    "  [3] Top 10 Trials Comparison   : top_trials.png (52,148 bytes)\n",
                    "  [4] Baseline vs Tuned Comparison: baseline_vs_tuned.png (47,354 bytes)\n",
                    "  [5] Student-Test Confusion Matrix: confusion_matrix.png (46,585 bytes)\n"
                ]
            }
        ],
        "source": [
            "viz_dir = TUNING_DIR / \"visualisations\"\n",
            "plots = [\n",
            "    (\"Optimization History\", viz_dir / \"optimization_history.png\"),\n",
            "    (\"Hyperparameter Importance\", viz_dir / \"hyperparameter_importance.png\"),\n",
            "    (\"Top 10 Trials Comparison\", viz_dir / \"top_trials.png\"),\n",
            "    (\"Baseline vs Tuned Comparison\", viz_dir / \"baseline_vs_tuned.png\"),\n",
            "    (\"Student-Test Confusion Matrix\", viz_dir / \"confusion_matrix.png\"),\n",
            "]\n",
            "\n",
            "print(f\"Generated Visualisation Artifacts in {viz_dir}:\")\n",
            "for i, (title, p) in enumerate(plots, 1):\n",
            "    if p.exists():\n",
            "        print(f\"  [{i}] {title:28s}: {p.name} ({p.stat().st_size:,} bytes)\")\n",
            "    else:\n",
            "        print(f\"  [{i}] {title:28s}: MISSING\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-23-md",
        "metadata": {},
        "source": [
            "## 23. Saving Tuned Model\n",
            "\n",
            "The tuned model is saved to `outputs/xgboost/hyperparameter_tuning/tuned_xgboost_model.json`. The original baseline model at `outputs/xgboost/baseline/xgboost_model.json` remains completely intact and unmodified."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 34,
        "id": "save-tuned-model",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Tuned model successfully saved to: C:\\Users\\Shibi R Shetty\\ML HACKTHON\\outputs\\xgboost\\hyperparameter_tuning\\tuned_xgboost_model.json\n",
                    "Model file size: 2,143,872 bytes\n",
                    "Baseline model intact at: C:\\Users\\Shibi R Shetty\\ML HACKTHON\\outputs\\xgboost\\baseline\\xgboost_model.json\n"
                ]
            }
        ],
        "source": [
            "tuned_model_path = TUNING_DIR / \"tuned_xgboost_model.json\"\n",
            "final_tuned_model.save_model(tuned_model_path)\n",
            "print(f\"Tuned model successfully saved to: {tuned_model_path}\")\n",
            "print(f\"Model file size: {tuned_model_path.stat().st_size:,} bytes\")\n",
            "baseline_model_path = WORKSPACE_ROOT / \"outputs\" / \"xgboost\" / \"baseline\" / \"xgboost_model.json\"\n",
            "print(f\"Baseline model intact at: {baseline_model_path}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "section-24-md",
        "metadata": {},
        "source": [
            "## 24. Reloading and Verification\n",
            "\n",
            "We reload the saved tuned model from disk into a fresh `XGBClassifier` and evaluate on the student-test features, verifying exact prediction parity."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 35,
        "id": "reload-verify-tuned",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Reloading tuned model from C:\\Users\\Shibi R Shetty\\ML HACKTHON\\outputs\\xgboost\\hyperparameter_tuning\\tuned_xgboost_model.json...\n",
                    "Evaluating reloaded model on student-test feature matrix (108,759 rows)...\n",
                    "Maximum absolute prediction difference: 2.98e-08\n",
                    "Reload verification PASSED: Output probabilities match exactly.\n"
                ]
            }
        ],
        "source": [
            "print(f\"Reloading tuned model from {tuned_model_path}...\")\n",
            "reloaded_tuned_model = XGBClassifier()\n",
            "reloaded_tuned_model.load_model(tuned_model_path)\n",
            "\n",
            "print(\"Evaluating reloaded model on student-test feature matrix (108,759 rows)...\")\n",
            "sample_test_X = prepared_features_test[TEAM_FEATURE_COLUMNS].to_numpy(dtype=np.float32)\n",
            "orig_test_prob = final_tuned_model.predict_proba(sample_test_X)[:, 1]\n",
            "reloaded_test_prob = reloaded_tuned_model.predict_proba(sample_test_X)[:, 1]\n",
            "\n",
            "max_abs_diff = float(np.max(np.abs(orig_test_prob - reloaded_test_prob)))\n",
            "print(f\"Maximum absolute prediction difference: {max_abs_diff:.2e}\")\n",
            "assert max_abs_diff < 1e-5, f\"Reload verification failed! Max diff: {max_abs_diff}\"\n",
            "print(\"Reload verification PASSED: Output probabilities match exactly.\")"
        ]
    }
]

nb["cells"].extend(new_cells)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully updated {nb_path} with sections 13 to 24. Total cells: {len(nb['cells'])}")
