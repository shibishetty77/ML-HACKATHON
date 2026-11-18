"""Audit saved XGBoost and four-feature MLP student-test predictions.

Pass the MLP project folder supplied by Team 9. This never retrains on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from report_benchmark import CSV_TYPES, OUT, continuity, evaluate, load_data

ROOT = Path(__file__).resolve().parents[1]


def check(name: str, predictions: Path, expected: dict,
          table: pd.DataFrame, coverage: pd.DataFrame, raw_zip: Path) -> None:
    pred = pd.read_csv(predictions, dtype=CSV_TYPES)
    pred["trip_id"] = pred.trip_id.str.zfill(8)
    key = ["trip_id", "observation_id", "candidate_id", "label"]
    if not pred[key].equals(table[key]):
        raise ValueError(f"{name}: saved predictions do not align with shared candidate rows")
    score = pred["score" if name == "xgboost" else "probability"].to_numpy(float)
    result, winners = evaluate(table, coverage, score, (score >= 0.5).astype(np.int8),
                               probabilities=True)
    route_cont, connected, total = continuity(winners, raw_zip)
    result.update({"classifier_route_continuity": route_cont,
                   "continuity_connected": connected,
                   "continuity_transitions": total})
    for metric, key_name in expected.items():
        if abs(result[metric] - key_name) > 1e-4:
            raise ValueError(f"{name}: {metric} mismatch: {result[metric]} vs {key_name}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_verified.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(name, json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-folder", type=Path, default=ROOT / "analysis" / "mlp_shared")
    parser.add_argument("--raw-zip", type=Path,
                        default=Path.home() / "Downloads" / "Student folder.zip")
    args = parser.parse_args()
    table, coverage = load_data("student_test")
    xgb_path = ROOT / "outputs" / "xgboost" / "hyperparameter_tuning"
    xgb_stats = json.loads((xgb_path / "student_test_metrics.json").read_text())
    check("xgboost", xgb_path / "student_test_predictions.csv", {
        "candidate_macro_f1": xgb_stats["macro_f1"],
        "conditional_top1": xgb_stats["conditional_top1_accuracy"],
        "end_to_end_top1": xgb_stats["end_to_end_top1_accuracy"],
        "log_loss": xgb_stats["log_loss"],
    }, table, coverage, args.raw_zip)
    mlp_path = args.mlp_folder if (args.mlp_folder / "metrics.json").is_file() \
        else args.mlp_folder / "teammate_mlp_results"
    mlp_stats = json.loads((mlp_path / "metrics.json").read_text())["student_test_metrics"]
    check("mlp", mlp_path / "student_test_predictions.csv", {
        "candidate_macro_f1": mlp_stats["macro_f1"],
        "conditional_top1": mlp_stats["top1_accuracy_conditional_on_candidate_table"],
        "end_to_end_top1": mlp_stats["top1_accuracy_end_to_end_over_aligned_observations"],
        "log_loss": mlp_stats["log_loss"],
    }, table, coverage, args.raw_zip)


if __name__ == "__main__":
    main()
