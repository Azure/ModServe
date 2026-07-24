from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

FEATURES = ("parallelism", "batch", "load_unit")


def _throughput(batch: float, load_unit: float, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return batch * load_unit * 1000.0 / latency_ms


class StageModel:

    def __init__(self, rf: RandomForestRegressor, stage: str, points: list[dict]):
        self._rf = rf
        self.stage = stage
        self.points = points

    @classmethod
    def fit(
        cls,
        points: list[dict],
        stage: str = "",
        n_estimators: int = 300,
        random_state: int = 0,
    ) -> "StageModel":
        if not points:
            raise ValueError("no points to fit")
        X = np.array(
            [[p["parallelism"], p["batch"], p["load_unit"]] for p in points], float
        )
        y = np.array([p["latency_ms"] for p in points], float)
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, y)
        return cls(rf, stage, points)

    def predict_latency(
        self, parallelism: float, batch: float, load_unit: float
    ) -> float:
        return float(self._rf.predict([[parallelism, batch, load_unit]])[0])

    def max_capacity(
        self,
        parallelism: int,
        slo_ms: float,
        batch_grid: list[int],
        load_grid: list[int],
    ) -> dict:
        best = {
            "m_c_per_s": 0.0,
            "at_batch": None,
            "at_load_unit": None,
            "at_latency_ms": None,
        }
        for b in batch_grid:
            for u in load_grid:
                lat = self.predict_latency(parallelism, b, u)
                if lat <= slo_ms:
                    thr = _throughput(b, u, lat)
                    if thr > best["m_c_per_s"]:
                        best = {
                            "m_c_per_s": thr,
                            "at_batch": b,
                            "at_load_unit": u,
                            "at_latency_ms": round(lat, 3),
                        }
        return best

    def r2(self) -> float:
        X = np.array(
            [[p["parallelism"], p["batch"], p["load_unit"]] for p in self.points], float
        )
        y = np.array([p["latency_ms"] for p in self.points], float)
        return float(self._rf.score(X, y))
