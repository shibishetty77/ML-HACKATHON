"""PathSnap evaluator API. Reference routes are read only after inference."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data import TripRepository, default_source
from .geometry import FEATURE_NAMES, RoadNetwork, motion_features
from .hmm import decode
from .model import CandidateClassifier

ROOT = Path(__file__).resolve().parents[2]
VERSION = "2.0.0"
LOGGER = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
STARTED = time.monotonic()
REQUEST_COUNT = 0
INFERENCE_LATENCIES: list[float] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.repository = TripRepository(default_source())
    app.state.classifier = CandidateClassifier(ROOT)
    LOGGER.info("startup model=%s selection=%s trips=%d source=%s",
                app.state.classifier.path, app.state.classifier.selection,
                len(app.state.repository.trip_ids), app.state.repository.source)
    try:
        yield
    finally:
        app.state.repository.close()
        LOGGER.info("shutdown evaluator data closed")


app = FastAPI(title="PathSnap Map Matching API", version=VERSION, lifespan=lifespan)
configured_origins = [item.strip() for item in os.getenv("PATHSNAP_CORS_ORIGINS", "").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", *configured_origins],
    allow_origin_regex=r"^http://(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}):3000$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$")


def _components() -> tuple[TripRepository, CandidateClassifier]:
    return app.state.repository, app.state.classifier


@app.get("/health")
def health() -> dict[str, Any]:
    repo, classifier = _components()
    return {
        "status": "healthy", "version": VERSION,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "model_loaded": bool(classifier.booster), "model_algorithm": "XGBoost",
        "features_loaded": len(FEATURE_NAMES), "hmm_ready": True,
        "evaluator_trips_available": len(repo.trip_ids),
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    return health()


@app.get("/api/v1/trips")
def trips() -> dict[str, Any]:
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    repo, _ = _components()
    summaries = []
    for trip_id in repo.trip_ids:
        # Read the track only: listing never touches reference routes.
        track = repo._read(trip_id, "track", columns=3)
        summaries.append({
            "trip_id": trip_id, "status": "available", "num_gps_points": len(track),
            "approx_duration_s": float(track[-1, 2] - track[0, 2]),
            "bounds": {"min_lat": float(track[:, 1].min()), "max_lat": float(track[:, 1].max()),
                       "min_lon": float(track[:, 0].min()), "max_lon": float(track[:, 0].max())},
        })
    return {"trips": summaries, "total": len(summaries)}


def _route_geometry(road: RoadNetwork, chunks: list[list[int]]) -> list[list[list[float]]]:
    polylines = []
    for arc_ids in chunks:
        line: list[list[float]] = []
        for arc_id in arc_ids:
            u, v = road.arcs[arc_id]
            for node in (u, v):
                lon, lat = road.nodes[node]
                point = [float(lat), float(lon)]
                if not line or line[-1] != point:
                    line.append(point)
        if line:
            polylines.append(line)
    return polylines


def _reference_metrics(predicted: list[int], reference: np.ndarray) -> dict[str, float | None]:
    # .route is an arc sequence, not per-observation labels. Observation Top-1,
    # candidate F1, log loss, and reference-candidate miss rate are unidentifiable.
    pred_set, ref_set = set(predicted), set(map(int, reference))
    common = len(pred_set & ref_set)
    precision = common / len(pred_set) if pred_set else 0.0
    recall = common / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": None, "precision": None, "recall": None, "macro_f1": None,
            "top1_accuracy": None, "top1_accuracy_conditional": None,
            "candidate_miss_rate": None, "log_loss": None,
            "route_arc_precision": precision, "route_arc_recall": recall, "route_arc_f1": f1}


@app.post("/api/v1/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    repo, classifier = _components()
    if request.trip_id not in repo.trip_ids:
        raise HTTPException(404, f"Trip {request.trip_id} is not in the evaluator set")
    started = time.perf_counter()
    LOGGER.info("prediction start trip=%s", request.trip_id)
    try:
        trip = repo.load(request.trip_id)
        road = RoadNetwork(trip)
        xy, speeds, headings = motion_features(trip.track, road)
        groups = [road.candidates(xy[i], speeds[i], None if np.isnan(headings[i]) else float(headings[i]))
                  for i in range(len(trip.track))]
        preprocess_ms = (time.perf_counter() - started) * 1000
        inference_ms = classifier.score(groups)
        decode_started = time.perf_counter()
        result = decode(road, groups, trip.track, xy)
        viterbi_ms = (time.perf_counter() - decode_started) * 1000
        evaluation_started = time.perf_counter()
        route_ids = [arc_id for chunk in result.chunks for arc_id in chunk]
        reference = repo.reference_route(request.trip_id)
        if np.any(reference < 0) or np.any(reference >= len(trip.arcs)):
            raise ValueError("Reference route contains invalid arc IDs")
        metrics = _reference_metrics(route_ids, reference)
        metrics["route_continuity"] = result.connected_steps / max(1, len(trip.track) - 1)
        no_candidates = sum(not group for group in groups)
        metrics["no_candidate_rate"] = no_candidates / len(groups)
        evaluation_ms = (time.perf_counter() - evaluation_started) * 1000
        polylines = _route_geometry(road, result.chunks)
        observations = []
        for i, (lon, lat, timestamp) in enumerate(trip.track):
            matched = result.assignments[i]
            observations.append({"obs_id": f"{request.trip_id}_OBS_{i:04d}",
                                 "timestamp": float(timestamp), "lat": float(lat), "lon": float(lon),
                                 "speed_mps": float(speeds[i]),
                                 "matched_arc_id": matched.arc_id if matched else None,
                                 "confidence": matched.probability if matched else None})
        total_ms = (time.perf_counter() - started) * 1000
        INFERENCE_LATENCIES.append(total_ms)
        state = "failed" if not route_ids else ("success" if result.unresolved_steps == 0 else "partial")
        LOGGER.info("prediction finish trip=%s status=%s observations=%d candidates=%d matched=%d unresolved=%d total_ms=%.1f",
                    request.trip_id, state, len(groups), sum(map(len, groups)), len(route_ids),
                    result.unresolved_steps, total_ms)
        return {
            "trip_id": request.trip_id, "status": state,
            "timing": {"preprocessing_ms": preprocess_ms, "inference_ms": inference_ms,
                       "viterbi_ms": viterbi_ms, "evaluation_ms": evaluation_ms,
                       "total_pipeline_ms": total_ms},
            "metrics": metrics,
            "candidate_stats": {"total_observations": len(groups),
                                "total_candidates": sum(map(len, groups)),
                                "avg_candidates_per_obs": sum(map(len, groups)) / len(groups),
                                "candidate_misses": no_candidates,
                                "no_candidate_observations": no_candidates},
            "route": {"num_matched_segments": len(route_ids), "matched_arc_ids": route_ids,
                      "predicted_polyline": polylines[0] if polylines else [],
                      "polylines": polylines, "arc_chunks": result.chunks,
                      "unresolved_steps": result.unresolved_steps},
            "gps_observations": observations,
            "candidate_rankings": [{"obs_id": observations[i]["obs_id"],
                                    "candidates": [{"arc_id": c.arc_id, "probability": c.probability}
                                                   for c in group[:5]]}
                                   for i, group in enumerate(groups)],
        }
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("prediction failure trip=%s", request.trip_id)
        raise HTTPException(500, "Prediction failed; inspect backend logs for details")


@app.get("/api/v1/model/info")
def model_info() -> dict[str, Any]:
    _, classifier = _components()
    params_path = ROOT / "outputs" / "xgboost" / "hyperparameter_tuning" / "best_hyperparameters.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if classifier.selection == "tuned" and params_path.is_file() else {}
    return {"model_family": "Gradient Boosted Decision Trees", "algorithm": "XGBoost",
            "version": xgboost.__version__, "framework": "xgboost-python",
            "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
            "hyperparameters": params, "model_selection": classifier.selection,
            "model_file": str(classifier.path), "fallback_reason": classifier.fallback_reason,
            "decoder": {"type": "Hidden Markov Model with log-space Viterbi decoding",
                        "emission_source": "XGBoost positive-class probability",
                        "transition_metric": "Directed shortest path and GPS displacement penalty"}}


@app.get("/api/v1/system/metrics")
def system_metrics() -> dict[str, Any]:
    return {"uptime_seconds": time.monotonic() - STARTED, "total_requests": REQUEST_COUNT,
            "avg_inference_latency_ms": sum(INFERENCE_LATENCIES) / len(INFERENCE_LATENCIES) if INFERENCE_LATENCIES else 0.0}
