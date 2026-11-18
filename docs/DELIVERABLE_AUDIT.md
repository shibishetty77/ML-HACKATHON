# PathSnap deliverable audit

This audit separates supplied evidence from claims in the original repository. No evaluator data was used to train or tune the model.

| Requirement | Evidence found | Status |
| --- | --- | --- |
| Tuned XGBoost model | Supplied `hyperparameter_tuning.zip` includes Trial 17 JSON, best parameters, Optuna DB, CV CSVs, student-test metrics and predictions. All 16 archive files were copied unchanged into `outputs/xgboost/hyperparameter_tuning`. | Present; integrated without retraining |
| Baseline model fallback | Referenced in tuning report; artifact absent from the supplied ZIP and checkout | Missing; backend uses it only if later provided |
| Shared candidate preprocessing source | Added on GitHub in `Hyperparameter_shared_pipeline` after the initial audit. The backend retains its independent candidate generator and was checked against saved student-test probabilities. | Present; backend alignment with the newly added source remains to be verified |
| Evaluator data | `evaluator_test.zip` contains ten trips, each with track, nodes, arcs and route | Present; ZIP is read directly |
| HMM and Viterbi | Original backend had simulated output; new directed graph and log-space decoder are implemented | Implemented and tested |
| Backend API | Original API returned fabricated route IDs, metrics and timings; new API computes them from model and raw files | Implemented and tested |
| Next.js evaluator UI | Present in repository; route rendering and unavailable metrics updated | Present; production build and six live API tests pass |
| Five-model comparison | The supplied Team 9 MLP folder contains a completed four-feature run, archived under `analysis/mlp_shared/`. `analysis/report_benchmark.py` re-evaluates Random Forest, linear SVM and KNN on the same four-feature tables and frozen trip folds used by XGBoost and MLP. Results are in `analysis/results/`. Original notebook runs used different contracts and remain separate. | Present; fixed-configuration common comparison, with measured caveats |
| Consolidated technical report PDF | `docs/PathSnap_Team9_Technical_Report.pdf` is built from archived results, common re-evaluations and supplied demo screenshots by `analysis/build_team9_report.py`. | Present; Team 9 should review and add member names if desired |
| Folium interactive map | `analysis/folium_map.py` generates `analysis/results/folium_trip_00000030.html` from evaluator GPS, network, post-inference reference and deployed route. A rendered screenshot is in `analysis/evidence/`. | Present |
| GitHub submission | Integrated backend, frontend, tests, and tuned model are included in this repository update | Included in this submission |

## Metric limitation

The evaluator `.route` sequence cannot be aligned one-to-one to `.track` rows without a separate alignment procedure. Candidate-level Top-1, macro F1, log loss and reference miss rate therefore cannot be reproduced honestly from the evaluator ZIP alone. The backend returns `null` for those fields and reports measured route arc overlap and directed continuity separately. The supplied student-test metrics in the tuning archive remain historical training-team results; they are not reused as evaluator-trip outputs.

## Feature compatibility check

On student-test trip `00000000`, the reconstructed 50 m candidate generator reproduced 10,688 of 10,692 candidate IDs in the saved prediction CSV. The score correlation on matched rows was 0.999987; the median absolute probability difference was about 2.0e-10 and the 99th percentile was about 0.0051. This is strong evidence for feature compatibility, but not exact equality for every row. The newly added `Hyperparameter_shared_pipeline` source should be compared with the backend before claiming exact reproduction.

## Tuning record integrity

The SQLite study contains 25 `COMPLETE` and 6 `FAIL` trials; `optuna_trials.csv` contains 25 rows. Trial 17 specifies 550 boosting rounds and the shipped model contains 550 rounds and four features. All 16 tuning archive files in the checkout are byte-identical to the supplied ZIP. The trials CSV gives Trial 17's original search score as 0.795488 Top-1 / 0.855889 macro F1, while the report's later final CV comparison lists 0.795549 / 0.855920. These appear to represent different evaluation runs; retain both labels and reconcile the run details before quoting a single CV figure.

## Remaining limitations

The baseline XGBoost model artifact is still absent, so the baseline-versus-tuned comparison relies on archived metrics. The backend shares the four feature names and 50 m radius with training, but its candidate generator can return more than the training table's ten arcs per observation. The student-test classifier table and evaluator route ablation are reported as separate evaluation units. Team 9 should review the PDF's analysis and add member details before final submission.
