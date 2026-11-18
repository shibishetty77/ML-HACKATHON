"""Directed road graph and training-compatible candidate feature construction."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from .data import Trip

FEATURE_NAMES = (
    "distance_to_segment_m",
    "projection_fraction",
    "observation_speed_mps",
    "heading_difference_rad",
)
SEARCH_RADIUS_M = 50.0
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Candidate:
    arc_id: int
    distance_m: float
    fraction: float
    features: tuple[float, float, float, float]
    probability: float = 0.0


class RoadNetwork:
    def __init__(self, trip: Trip):
        self.trip = trip
        self.nodes = trip.nodes
        self.arcs = trip.arcs
        self.lat0 = math.radians(float(np.mean(self.nodes[:, 1])))
        self.lon0 = math.radians(float(np.mean(self.nodes[:, 0])))
        self.xy = self.project(self.nodes)
        self.starts = self.xy[self.arcs[:, 0]]
        self.ends = self.xy[self.arcs[:, 1]]
        self.vectors = self.ends - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.bearings = np.arctan2(self.vectors[:, 1], self.vectors[:, 0])
        self.adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(len(self.nodes))]
        for arc_id, (u, v) in enumerate(self.arcs):
            self.adjacency[int(u)].append((int(v), arc_id, float(self.lengths[arc_id])))

        # Sample long arcs so the index cannot miss an arc whose midpoint is far away.
        counts = np.maximum(1, np.ceil(self.lengths / 50.0).astype(int))
        sample_arc_ids = np.repeat(np.arange(len(self.arcs), dtype=np.int32), counts + 1)
        within = np.arange(len(sample_arc_ids)) - np.repeat(np.cumsum(counts + 1) - (counts + 1), counts + 1)
        fractions = within / (counts[sample_arc_ids])
        points = self.starts[sample_arc_ids] + fractions[:, None] * self.vectors[sample_arc_ids]
        self.sample_arc_ids = sample_arc_ids
        self.index = cKDTree(points)

    def project(self, lon_lat: np.ndarray) -> np.ndarray:
        lon_lat = np.asarray(lon_lat, dtype=float)
        radians = np.radians(lon_lat)
        return np.column_stack(((radians[:, 0] - self.lon0) * EARTH_RADIUS_M * math.cos(self.lat0),
                                (radians[:, 1] - self.lat0) * EARTH_RADIUS_M))

    def candidates(self, point_xy: np.ndarray, speed: float, heading: float | None) -> list[Candidate]:
        sample_indices = self.index.query_ball_point(point_xy, SEARCH_RADIUS_M + 25.0)
        if not sample_indices:
            return []
        ids = np.unique(self.sample_arc_ids[sample_indices])
        vectors = self.vectors[ids]
        lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
        delta = point_xy - self.starts[ids]
        fractions = np.divide(np.einsum("ij,ij->i", delta, vectors), lengths_sq,
                              out=np.zeros(len(ids)), where=lengths_sq > 0)
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = self.starts[ids] + fractions[:, None] * vectors
        distances = np.linalg.norm(point_xy - projections, axis=1)
        selected = np.flatnonzero(distances <= SEARCH_RADIUS_M)
        result = []
        for j in selected:
            difference = 0.0 if heading is None else abs(math.atan2(math.sin(heading - self.bearings[ids[j]]),
                                                                     math.cos(heading - self.bearings[ids[j]])))
            features = (float(distances[j]), float(fractions[j]), float(speed), float(difference))
            result.append(Candidate(int(ids[j]), float(distances[j]), float(fractions[j]), features))
        return result

    @lru_cache(maxsize=256)
    def _reachable(self, start: int) -> dict[int, tuple[float, int, int]]:
        # Distances are directed. The 1 km cap applies only to a single <=30 s step.
        distances: dict[int, tuple[float, int, int]] = {start: (0.0, -1, -1)}
        queue = [(0.0, start)]
        while queue:
            length, node = heapq.heappop(queue)
            if length > distances[node][0] + 1e-9:
                continue
            for successor, arc_id, arc_length in self.adjacency[node]:
                new_length = length + arc_length
                if new_length <= 1000.0 and (successor not in distances or new_length < distances[successor][0]):
                    distances[successor] = (new_length, node, arc_id)
                    heapq.heappush(queue, (new_length, successor))
        return distances

    def directed_path(self, start: int, end: int) -> tuple[float, list[int]] | None:
        if start == end:
            return 0.0, []
        reachable = self._reachable(start)
        if end not in reachable:
            return None
        length = reachable[end][0]
        arcs = []
        node = end
        while node != start:
            _, prior, arc_id = reachable[node]
            arcs.append(arc_id)
            node = prior
        arcs.reverse()
        return length, arcs

    def transition(self, before: Candidate, after: Candidate, gps_distance: float, elapsed_s: float) -> tuple[float, list[int]] | None:
        if elapsed_s <= 0 or elapsed_s > 30:
            return None
        if before.arc_id == after.arc_id:
            # The hidden state is the arc, not a sub-arc position. GPS jitter can
            # shift a projection backward while the vehicle remains on that arc.
            network_distance = max(0.0, after.fraction - before.fraction) * self.lengths[before.arc_id]
            connectors: list[int] = []
        else:
            exit_node = int(self.arcs[before.arc_id, 1])
            enter_node = int(self.arcs[after.arc_id, 0])
            path = self.directed_path(exit_node, enter_node)
            if path is None:
                return None
            middle_length, connectors = path
            network_distance = ((1 - before.fraction) * self.lengths[before.arc_id]
                                + middle_length + after.fraction * self.lengths[after.arc_id])
        max_travel = min(1000.0, max(150.0, gps_distance * 3 + 80.0, elapsed_s * 40.0 + 80.0))
        if network_distance > max_travel:
            return None
        scale = max(20.0, gps_distance * 0.5 + 15.0)
        return -abs(float(network_distance) - gps_distance) / scale, connectors


def motion_features(track: np.ndarray, road: RoadNetwork) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = road.project(track[:, :2])
    delta = np.diff(xy, axis=0)
    dt = np.diff(track[:, 2])
    distances = np.linalg.norm(delta, axis=1)
    speeds = np.zeros(len(track), dtype=float)
    headings = np.full(len(track), np.nan)
    valid = (distances >= 1.0) & (dt <= 30.0) & (dt > 0)
    speeds[1:][valid] = distances[valid] / dt[valid]
    headings[1:][valid] = np.arctan2(delta[valid, 1], delta[valid, 0])
    return xy, speeds, headings
