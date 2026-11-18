"""Log-space Viterbi over directed road arcs, retaining unresolved gaps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Candidate, RoadNetwork


@dataclass
class DecodedRoute:
    assignments: list[Candidate | None]
    chunks: list[list[int]]
    connected_steps: int
    unresolved_steps: int


def decode(road: RoadNetwork, groups: list[list[Candidate]], track: np.ndarray, xy: np.ndarray,
           *, max_states: int = 8) -> DecodedRoute:
    if max_states < 1:
        raise ValueError("max_states must be positive")
    selected = [group[:max_states] for group in groups]
    assignments: list[Candidate | None] = [None] * len(selected)
    chunks: list[list[int]] = []
    connected_steps = 0
    active_positions: list[int] = []
    scores: list[float] = []
    backpointers: list[list[int]] = []

    def flush() -> None:
        nonlocal active_positions, scores, backpointers
        if not active_positions:
            return
        choice = int(np.argmax(scores))
        for offset in range(len(active_positions) - 1, -1, -1):
            position = active_positions[offset]
            assignments[position] = selected[position][choice]
            choice = backpointers[offset][choice]
        active_positions = []
        scores = []
        backpointers = []

    for position, current in enumerate(selected):
        if not current:
            flush()
            continue
        emissions = [math.log(max(1e-9, min(1.0, candidate.probability))) for candidate in current]
        if not active_positions:
            active_positions = [position]
            scores = emissions
            backpointers = [[-1] * len(current)]
            continue
        previous_position = active_positions[-1]
        elapsed = float(track[position, 2] - track[previous_position, 2])
        gps_distance = float(np.linalg.norm(xy[position] - xy[previous_position]))
        prior = selected[previous_position]
        next_scores = [-math.inf] * len(current)
        pointers = [-1] * len(current)
        for i, before in enumerate(prior):
            transitions = [road.transition(before, after, gps_distance, elapsed) for after in current]
            finite = [(j, value[0]) for j, value in enumerate(transitions) if value is not None]
            if not finite:
                continue
            weights = [value for _, value in finite]
            peak = max(weights)
            log_normalizer = peak + math.log(sum(math.exp(value - peak) for value in weights))
            for j, weight in finite:
                score = scores[i] + weight - log_normalizer + emissions[j]
                if score > next_scores[j]:
                    next_scores[j] = score
                    pointers[j] = i
        if all(not math.isfinite(score) for score in next_scores):
            flush()
            active_positions = [position]
            scores = emissions
            backpointers = [[-1] * len(current)]
        else:
            active_positions.append(position)
            scores = next_scores
            backpointers.append(pointers)
    flush()

    # Build geometry only from transitions the decoder actually accepted.
    current_chunk: list[int] = []
    previous: Candidate | None = None
    previous_position = -1
    for position, candidate in enumerate(assignments):
        if candidate is None:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = []
            previous = None
            continue
        if previous is not None and position == previous_position + 1:
            elapsed = float(track[position, 2] - track[previous_position, 2])
            gps_distance = float(np.linalg.norm(xy[position] - xy[previous_position]))
            transition = road.transition(previous, candidate, gps_distance, elapsed)
            if transition is not None:
                connected_steps += 1
                for connector in transition[1]:
                    if not current_chunk or current_chunk[-1] != connector:
                        current_chunk.append(connector)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = []
        elif current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
        if not current_chunk or current_chunk[-1] != candidate.arc_id:
            current_chunk.append(candidate.arc_id)
        previous = candidate
        previous_position = position
    if current_chunk:
        chunks.append(current_chunk)
    unresolved = max(0, len(assignments) - 1 - connected_steps)
    return DecodedRoute(assignments, chunks, connected_steps, unresolved)
