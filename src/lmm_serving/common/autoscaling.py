from __future__ import annotations

import math
import os
import threading
import time
from collections import deque


class LoadWindow:

    def __init__(self, window_s: float):
        self._window_s = float(window_s)
        self._events: deque = deque()  # (ts, tokens)
        self._lock = threading.Lock()

    def add(self, tokens: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._events.append((now, float(tokens)))
            self._evict(now)

    def rate_per_s(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._evict(now)
            total = sum(t for _, t in self._events)
        return total / self._window_s

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


def desired_active(load_per_s: float, m_c: float, n_max: int) -> int:
    if load_per_s <= 0:
        return 1
    if m_c <= 0:
        return n_max
    return max(1, min(math.ceil(load_per_s / m_c), n_max))


class Autoscaler:

    def __init__(self, pools, interval_s, logger, metrics_path=None):
        self._pools = pools
        self._interval_s = float(interval_s)
        self._logger = logger
        self._metrics_path = metrics_path
        self._stop = threading.Event()
        self._thread = None

    def decide(self, now=None) -> dict:
        out = {}
        for p in self._pools:
            load = p["window"].rate_per_s(now)
            out[p["name"]] = desired_active(load, p["m_c"], p["n_max"])
        return out

    def apply(self, decision: dict) -> None:
        for p in self._pools:
            p["scheduler"].set_active(decision[p["name"]])

    def _tick(self) -> None:
        now = time.monotonic()
        decision = self.decide(now)
        self.apply(decision)
        loads = {p["name"]: round(p["window"].rate_per_s(now), 1) for p in self._pools}
        self._logger.info("autoscale: active=%s load_tok_s=%s", decision, loads)
        if self._metrics_path:
            header = not os.path.exists(self._metrics_path)
            with open(self._metrics_path, "a") as f:
                if header:
                    cols = (
                        ["wall_time"]
                        + [f"{p['name']}_active" for p in self._pools]
                        + [f"{p['name']}_load_tok_s" for p in self._pools]
                    )
                    f.write(",".join(cols) + "\n")
                row = (
                    [f"{time.time():.0f}"]
                    + [str(decision[p["name"]]) for p in self._pools]
                    + [str(loads[p["name"]]) for p in self._pools]
                )
                f.write(",".join(row) + "\n")

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._tick()
            except Exception:
                self._logger.exception("autoscaler tick failed")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._logger.info(
            "Autoscaler started: interval=%ss pools=%s",
            self._interval_s,
            {p["name"]: f"M_C={p['m_c']} N={p['n_max']}" for p in self._pools},
        )

    def stop(self) -> None:
        self._stop.set()
