import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.data import Trip, TripRepository
from app.geometry import Candidate, FEATURE_NAMES, RoadNetwork
from app.hmm import decode
from app.main import app
from app.model import CandidateClassifier


def small_road():
    nodes = np.array([[0, 0], [0.0001, 0], [0.0002, 0], [0.001, 0], [0.0011, 0]], dtype=float)
    arcs = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int32)
    track = np.array([[0.00005, 0, 1], [0.00015, 0, 2]], dtype=float)
    return RoadNetwork(Trip("synthetic", track, nodes, arcs))


def candidate(arc_id, fraction, probability):
    c = Candidate(arc_id, 0, fraction, (0, fraction, 5, 0))
    c.probability = probability
    return c


def test_directed_transitions_and_stationary_jitter():
    road = small_road()
    assert road.transition(candidate(0, 0.8, .9), candidate(1, 0.2, .9), 5, 1) is not None
    assert road.transition(candidate(1, 0.8, .9), candidate(0, 0.2, .9), 5, 1) is None
    assert road.transition(candidate(0, 0.8, .9), candidate(2, 0.2, .9), 5, 1) is None
    assert road.transition(candidate(0, 0.8, .9), candidate(0, 0.7, .9), 0, 1) is not None


def test_viterbi_rejects_disconnected_high_score():
    road = small_road()
    groups = [[candidate(0, 0.5, .8)], [candidate(2, 0.5, .99), candidate(1, 0.5, .5)]]
    result = decode(road, groups, road.trip.track, road.project(road.trip.track[:, :2]))
    assert [c.arc_id if c else None for c in result.assignments] == [0, 1]
    assert result.chunks == [[0, 1]]
    assert result.unresolved_steps == 0


def test_viterbi_marks_missing_candidate():
    road = small_road()
    result = decode(road, [[candidate(0, .5, .9)], []], road.trip.track,
                    road.project(road.trip.track[:, :2]))
    assert result.assignments[1] is None
    assert result.unresolved_steps == 1


def test_model_feature_contract():
    model = CandidateClassifier(Path(__file__).resolve().parents[2])
    assert model.selection == "tuned"
    assert model.booster.num_features() == 4
    assert FEATURE_NAMES == ("distance_to_segment_m", "projection_fraction",
                             "observation_speed_mps", "heading_difference_rad")
    good = candidate(0, .5, 0)
    model.score([[good]])
    assert 0 <= good.probability <= 1
    bad = Candidate(0, 0, 0, (float("nan"), 0, 0, 0))
    with pytest.raises(ValueError):
        model.score([[bad]])


@pytest.fixture(scope="module")
def client():
    if not os.getenv("PATHSNAP_EVALUATOR_DATA"):
        pytest.skip("Set PATHSNAP_EVALUATOR_DATA to evaluator_test.zip")
    with TestClient(app) as test_client:
        yield test_client


def test_evaluator_discovery_and_api(client):
    expected = {"00000015", "00000017", "00000023", "00000030", "00000049",
                "00000059", "00000067", "00000073", "00000087", "00000094"}
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/docs").status_code == 200
    listed = client.get("/api/v1/trips").json()
    assert listed["total"] == 10
    assert {trip["trip_id"] for trip in listed["trips"]} == expected
    assert client.post("/api/v1/predict", json={"trip_id": "00000000"}).status_code == 404
    assert client.post("/api/v1/predict", json={"trip_id": "../bad"}).status_code == 422


def test_real_trip_prediction_and_reference_isolation(client, monkeypatch):
    repo = app.state.repository
    original = repo.reference_route
    def forbidden(_):
        raise AssertionError("reference route read during candidate generation")
    # The production handler calls reference_route only after decode. Check the
    # lower-level loader itself never opens .route during inference.
    monkeypatch.setattr(repo, "reference_route", forbidden)
    trip = repo.load("00000023")
    assert len(trip.track) == 458
    monkeypatch.setattr(repo, "reference_route", original)
    response = client.post("/api/v1/predict", json={"trip_id": "00000023"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] in {"success", "partial"}
    assert len(data["gps_observations"]) == 458
    assert data["candidate_stats"]["total_candidates"] > 0
    assert data["route"]["num_matched_segments"] > 0
    assert data["timing"]["total_pipeline_ms"] > 0
    assert data["metrics"]["top1_accuracy"] is None
    assert "reference_route" not in data and "reference_labels" not in data
    assert all(0 <= arc < len(trip.arcs) for arc in data["route"]["matched_arc_ids"])
    for chunk in data["route"]["arc_chunks"]:
        assert all(trip.arcs[before, 1] == trip.arcs[after, 0]
                   for before, after in zip(chunk, chunk[1:]))


def test_cors_private_frontend_origin(client):
    response = client.options("/api/v1/predict", headers={
        "Origin": "http://192.168.43.205:3000", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://192.168.43.205:3000"
    assert "access-control-allow-credentials" not in response.headers


@pytest.mark.parametrize("trip_id", ["00000015", "00000017", "00000023", "00000030",
                                     "00000049", "00000059", "00000067", "00000073",
                                     "00000087", "00000094"])
def test_every_evaluator_trip_decodes_without_false_graph_edges(client, trip_id):
    response = client.post("/api/v1/predict", json={"trip_id": trip_id})
    assert response.status_code == 200, response.text
    data = response.json()
    trip = app.state.repository.load(trip_id)
    assert data["status"] in {"success", "partial"}
    assert len(data["gps_observations"]) == len(trip.track)
    assert data["candidate_stats"]["total_candidates"] > 0
    assert len(data["route"]["arc_chunks"]) == len(data["route"]["polylines"])
    assert data["route"]["num_matched_segments"] == sum(map(len, data["route"]["arc_chunks"]))
    for chunk in data["route"]["arc_chunks"]:
        assert all(trip.arcs[a, 1] == trip.arcs[b, 0] for a, b in zip(chunk, chunk[1:]))
    assert "reference_route" not in data
