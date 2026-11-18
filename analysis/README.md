# Team 9 report experiments

This directory contains the measured inputs for the Team 9 consolidated report.
The final service remains the tuned XGBoost classifier and HMM in `backend/`.

## Common model comparison

The five-model table uses the shared four-feature candidate and coverage CSVs
in `Hyperparameter_shared_pipeline/outputs/` and the frozen trip folds in
`Hyperparameter_shared_pipeline/fold_assignments.json`. The XGBoost results
come from `outputs/xgboost/hyperparameter_tuning/`. The MLP results and trainer
come from the supplied Team 9 MLP project and are archived in `mlp_shared/`.
The MLP source was adjusted only to read the same CSVs from this repository.

The Random Forest, linear SVM and KNN models were re-evaluated on the shared
CSV rows because their original notebooks did not contain a complete,
comparable set of metrics. Their fixed configurations are in
`report_benchmark.py`; no student-test labels were used for tuning. Run with
Python 3.11 after installing `backend/requirements.txt` and scikit-learn,
pandas, matplotlib, and folium:

```powershell
py -3.11 analysis\report_benchmark.py --model svm
py -3.11 analysis\report_benchmark.py --model random_forest
py -3.11 analysis\report_benchmark.py --model knn
py -3.11 analysis\score_existing.py
py -3.11 analysis\plot_report.py
py -3.11 analysis\build_team9_report.py
```

`score_existing.py` reads the archived MLP results in `mlp_shared/` by default.

The common candidate CSVs contain only observation groups where the reference
arc was retrieved. Candidate Macro F1 and conditional Top-1 are therefore
conditional on that event. End-to-end Top-1 uses all aligned observations in
the coverage file as its denominator. Route continuity in the five-model table
means the chosen arc is unchanged or directly adjacent to the next chosen arc,
for consecutive covered GPS rows with a time gap of at most 30 seconds.
Inference timings are local measurements from different run environments and
are not a controlled hardware benchmark.

## Deployed decoder and map

With `evaluator_test.zip` in Downloads or an explicit path, run:

```powershell
py -3.11 analysis\hmm_ablation.py --evaluator-zip C:\path\to\evaluator_test.zip
py -3.11 analysis\folium_map.py --evaluator-zip C:\path\to\evaluator_test.zip
```

The ablation computes route-arc set overlap with `.route` after predictions.
It does not turn `.route` into per-observation truth. The Folium HTML includes
raw GPS points, road geometry, reference route, and deployed XGBoost plus HMM
geometry. Its street tiles require an internet connection.

The screenshots in `evidence/` were supplied from the live two-laptop demo.
The report uses the supplied Route Explorer capture for the map figure; the
Folium HTML is a separate interactive artifact.
`results/figures/` contains plots generated only from archived CSV/JSON
results. No result is derived from another team's report.
