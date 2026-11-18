"""Strict readers for the evaluator's per-trip files, in a directory or ZIP."""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: np.ndarray  # longitude, latitude, timestamp
    nodes: np.ndarray  # longitude, latitude
    arcs: np.ndarray  # directed from_node, to_node; row number is arc_id


class TripRepository:
    def __init__(self, source: Path):
        self.source = source.resolve()
        self._zip = zipfile.ZipFile(self.source) if self.source.is_file() else None
        if not self.source.exists():
            raise FileNotFoundError(f"Evaluator data not found: {self.source}")
        if self._zip:
            names = self._zip.namelist()
            self._prefix = next((p for p in ("evaluator_test/", "") if any(n.startswith(p) and n.endswith(".track") for n in names)), "")
            ids = {m.group(1) for n in names if (m := re.fullmatch(re.escape(self._prefix) + r"(\d{8})/\1\.track", n))}
        else:
            base = self.source / "evaluator_test" if (self.source / "evaluator_test").is_dir() else self.source
            self.source = base
            ids = {p.name for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name) and (p / f"{p.name}.track").is_file()}
        self.trip_ids = sorted(ids)
        if not self.trip_ids:
            raise ValueError(f"No evaluator trips found in {self.source}")
        for trip_id in self.trip_ids:
            for extension in ("track", "nodes", "arcs", "route"):
                if not self._exists(trip_id, extension):
                    raise ValueError(f"Missing {trip_id}.{extension}")

    def _name(self, trip_id: str, extension: str) -> str:
        return f"{self._prefix}{trip_id}/{trip_id}.{extension}" if self._zip else str(self.source / trip_id / f"{trip_id}.{extension}")

    def _exists(self, trip_id: str, extension: str) -> bool:
        name = self._name(trip_id, extension)
        return name in self._zip.namelist() if self._zip else Path(name).is_file()

    def _read(self, trip_id: str, extension: str, *, columns: int, dtype=float) -> np.ndarray:
        if trip_id not in self.trip_ids:
            raise KeyError(trip_id)
        name = self._name(trip_id, extension)
        if self._zip:
            with self._zip.open(name) as file:
                data = np.loadtxt(io.BytesIO(file.read()), dtype=dtype)
        else:
            data = np.loadtxt(name, dtype=dtype)
        data = np.asarray(data).reshape(-1, columns)
        if data.size == 0 or not np.isfinite(data).all():
            raise ValueError(f"Empty or non-finite {name}")
        return data

    def load(self, trip_id: str) -> Trip:
        # Deliberately does not read .route. Reference data cannot enter inference.
        track = self._read(trip_id, "track", columns=3)
        nodes = self._read(trip_id, "nodes", columns=2)
        arcs = self._read(trip_id, "arcs", columns=2, dtype=np.int32)
        if np.any(np.diff(track[:, 2]) <= 0):
            raise ValueError(f"Trip {trip_id} timestamps are not strictly increasing")
        if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
            raise ValueError(f"Trip {trip_id} contains invalid node references")
        if np.any(np.abs(track[:, 1]) > 90) or np.any(np.abs(track[:, 0]) > 180):
            raise ValueError(f"Trip {trip_id} has invalid GPS coordinates")
        return Trip(trip_id, track, nodes, arcs)

    def reference_route(self, trip_id: str) -> np.ndarray:
        return self._read(trip_id, "route", columns=1, dtype=np.int32).ravel()

    def close(self) -> None:
        if self._zip:
            self._zip.close()


def default_source() -> Path:
    configured = os.getenv("PATHSNAP_EVALUATOR_DATA")
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[2]
    for candidate in (root / "evaluator_test", root / "evaluator_test.zip", Path.home() / "Downloads" / "evaluator_test.zip"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Set PATHSNAP_EVALUATOR_DATA to evaluator_test.zip or its extracted directory")
