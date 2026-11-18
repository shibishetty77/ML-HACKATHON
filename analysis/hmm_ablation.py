"""Compare independent XGBoost arc choices with the deployed Viterbi decoder.

The evaluator reference `.route` is an ordered arc sequence, not a per-GPS
label. This experiment reports route arc set overlap and transition feasibility,
not observation Top-1 or candidate F1.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.data import TripRepository
from backend.app.geometry import RoadNetwork, motion_features
from backend.app.hmm import decode
from backend.app.model import CandidateClassifier


def arc_scores(ids: list[int], reference: np.ndarray) -> tuple[float, float, float]:
    found, truth = set(ids), set(map(int, reference))
    shared = len(found & truth)
    precision = shared / len(found) if found else 0.0
    recall = shared / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-zip", type=Path,
                        default=Path.home() / "Downloads" / "evaluator_test.zip")
    args = parser.parse_args()
    repo = TripRepository(args.evaluator_zip)
    classifier = CandidateClassifier(ROOT)
    rows = []
    for trip_id in repo.trip_ids:
        trip = repo.load(trip_id)
        road = RoadNetwork(trip)
        xy, speed, heading = motion_features(trip.track, road)
        groups = [road.candidates(xy[i], speed[i], None if np.isnan(heading[i]) else float(heading[i]))
                  for i in range(len(trip.track))]
        classifier.score(groups)
        independent = [group[0] if group else None for group in groups]
        independent_ids = [candidate.arc_id for candidate in independent if candidate is not None]
        connected = 0
        for i in range(1, len(trip.track)):
            before, after = independent[i - 1], independent[i]
            if before is None or after is None:
                continue
            elapsed = float(trip.track[i, 2] - trip.track[i - 1, 2])
            distance = float(np.linalg.norm(xy[i] - xy[i - 1]))
            connected += road.transition(before, after, distance, elapsed) is not None
        decoded = decode(road, groups, trip.track, xy)
        decoded_ids = [arc for chunk in decoded.chunks for arc in chunk]
        reference = repo.reference_route(trip_id)
        base_p, base_r, base_f1 = arc_scores(independent_ids, reference)
        hmm_p, hmm_r, hmm_f1 = arc_scores(decoded_ids, reference)
        denominator = max(1, len(trip.track) - 1)
        row = {
            "trip_id": trip_id,
            "observations": len(trip.track),
            "candidate_rows": sum(map(len, groups)),
            "no_candidate_observations": sum(not group for group in groups),
            "classifier_arc_precision": base_p,
            "classifier_arc_recall": base_r,
            "classifier_arc_f1": base_f1,
            "classifier_transition_feasibility": connected / denominator,
            "hmm_arc_precision": hmm_p,
            "hmm_arc_recall": hmm_r,
            "hmm_arc_f1": hmm_f1,
            "hmm_transition_feasibility": decoded.connected_steps / denominator,
            "hmm_unresolved_steps": decoded.unresolved_steps,
            "hmm_minus_classifier_arc_f1": hmm_f1 - base_f1,
        }
        rows.append(row)
        print(trip_id, "base", f"{base_f1:.4f}", "hmm", f"{hmm_f1:.4f}",
              "continuity", f"{connected / denominator:.4f}",
              f"{decoded.connected_steps / denominator:.4f}", flush=True)
    repo.close()
    out = ROOT / "analysis" / "results" / "hmm_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
