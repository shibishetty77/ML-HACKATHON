"""Load the trained classifier once and reject incompatible feature matrices."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from .geometry import FEATURE_NAMES, Candidate


class CandidateClassifier:
    def __init__(self, root: Path):
        configured = os.getenv("PATHSNAP_MODEL_PATH")
        tuned = Path(configured) if configured else root / "outputs" / "xgboost" / "hyperparameter_tuning" / "tuned_xgboost_model.json"
        baseline = root / "outputs" / "xgboost" / "baseline" / "xgboost_model.json"
        errors = []
        for name, path in (("tuned", tuned), ("baseline", baseline)):
            if not path.is_file():
                errors.append(f"{name}: missing {path}")
                continue
            try:
                booster = xgb.Booster()
                booster.load_model(str(path))
                if booster.num_features() != len(FEATURE_NAMES):
                    raise ValueError(f"expected {len(FEATURE_NAMES)} features, found {booster.num_features()}")
                if booster.feature_names and booster.feature_names != list(FEATURE_NAMES):
                    raise ValueError(f"feature order mismatch: {booster.feature_names}")
                probe = np.array([[0, 0, 0, 0]], dtype=np.float32)
                result = booster.predict(xgb.DMatrix(probe, feature_names=list(FEATURE_NAMES)))
                if result.shape != (1,) or not np.isfinite(result).all() or not (0 <= result[0] <= 1):
                    raise ValueError("invalid probability from model probe")
                self.booster = booster
                self.path = path
                self.selection = name
                self.fallback_reason = "; ".join(errors) if name == "baseline" else None
                return
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError("No compatible XGBoost model: " + "; ".join(errors))

    def score(self, groups: list[list[Candidate]]) -> float:
        flat = [candidate for group in groups for candidate in group]
        if not flat:
            return 0.0
        matrix = np.asarray([candidate.features for candidate in flat], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
            raise ValueError("Candidate feature matrix has missing, non-finite, or incorrectly ordered values")
        if np.any(matrix[:, 0] < 0) or np.any((matrix[:, 1] < 0) | (matrix[:, 1] > 1)) or np.any(matrix[:, 2] < 0) or np.any((matrix[:, 3] < 0) | (matrix[:, 3] > math.pi + 1e-6)):
            raise ValueError("Candidate feature values are outside their valid ranges")
        started = time.perf_counter()
        probabilities = self.booster.predict(xgb.DMatrix(matrix, feature_names=list(FEATURE_NAMES)))
        elapsed_ms = (time.perf_counter() - started) * 1000
        if len(probabilities) != len(flat) or not np.isfinite(probabilities).all():
            raise RuntimeError("XGBoost returned invalid probabilities")
        for candidate, probability in zip(flat, probabilities):
            candidate.probability = float(probability)
        for group in groups:
            group.sort(key=lambda candidate: (-candidate.probability, candidate.arc_id))
        return elapsed_ms
