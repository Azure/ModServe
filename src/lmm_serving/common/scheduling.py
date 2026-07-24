from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import pandas as pd
import requests

# A node identifier — in practice a downstream "host:port" address string.
Node = str

# Per-request metrics schema, shared by both cluster managers (keyed by req_id).
REQUEST_METRICS_COLUMNS = [
    "req_id",
    "vllm_req_id",
    "timestamp",
    "num_images",
    "n_context_tokens",
    "n_generated_tokens",
    "num_image_tokens",
    "cluster_manager_waiting_time",
    "total_image_processing_time",
    "total_llm_backend_processing_time",
    "e2e_time",
    "http_code",
]


def request_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUEST_METRICS_COLUMNS).set_index("req_id")


def record_request_metrics(
    df: pd.DataFrame,
    lock: threading.Lock,
    req_id: str,
    vllm_req_id: str,
    timestamp: str | float = "",
    num_images: int = 0,
    n_context_tokens: int = 0,
    n_generated_tokens: int = 0,
    num_image_tokens: int = 0,
    cluster_manager_waiting_time: float = 0,
    total_image_processing_time: float = 0,
    total_llm_backend_processing_time: float = 0,
    e2e_time: float = 0,
    http_code: int = 200,
) -> None:
    row = {
        "vllm_req_id": vllm_req_id,
        "timestamp": timestamp,
        "num_images": num_images,
        "n_context_tokens": n_context_tokens,
        "n_generated_tokens": n_generated_tokens,
        "num_image_tokens": num_image_tokens,
        "cluster_manager_waiting_time": cluster_manager_waiting_time,
        "total_image_processing_time": total_image_processing_time,
        "total_llm_backend_processing_time": total_llm_backend_processing_time,
        "e2e_time": e2e_time,
        "http_code": http_code,
    }
    with lock:
        df.loc[req_id] = row


class Scheduler(ABC):

    def __init__(self, nodes: Iterable[Node]) -> None:
        self._nodes = list(nodes)
        self._active = len(self._nodes)
        self._lock = threading.Lock()

    def set_active(self, k: int) -> None:
        with self._lock:
            self._active = max(1, min(int(k), len(self._nodes)))

    @abstractmethod
    def acquire(self, load: int = 0) -> Node:
        pass

    @abstractmethod
    def release(self, node: Node, load: int = 0) -> None:
        pass


class RoundRobinScheduler(Scheduler):

    def __init__(self, nodes: Iterable[Node]) -> None:
        super().__init__(nodes)
        self._i = 0

    def acquire(self, load: int = 0) -> Node:
        with self._lock:
            node = self._nodes[self._i % self._active]
            self._i += 1
        return node

    def release(self, node: Node, load: int = 0) -> None:
        pass


class LeastLoadedScheduler(Scheduler):

    def __init__(self, nodes: Iterable[Node]) -> None:
        super().__init__(nodes)
        self._load: dict[Node, int] = {n: 0 for n in self._nodes}
        self._rr = 0

    def acquire(self, load: int = 0) -> Node:
        with self._lock:
            pool = self._nodes[: self._active]
            low = min(self._load[n] for n in pool)
            tied = [n for n in pool if self._load[n] == low]
            node = tied[self._rr % len(tied)]  # rotate among equally-loaded nodes
            self._rr += 1
            self._load[node] += load
        return node

    def release(self, node: Node, load: int = 0) -> None:
        with self._lock:
            if node in self._load:
                self._load[node] = max(0, self._load[node] - load)


def make_scheduler(name: str | None, nodes: Iterable[Node]) -> Scheduler:
    name = (name or "round_robin").lower()
    if name == "least_loaded":
        return LeastLoadedScheduler(nodes)
    if name == "round_robin":
        return RoundRobinScheduler(nodes)
    raise ValueError(
        f"Unknown scheduler={name!r} (expected round_robin or least_loaded)"
    )


def num_pending_tokens(
    n_text_tokens: int | str, n_image_tokens: int | str, *, decoder_only: bool
) -> int:
    return int(n_text_tokens) + (int(n_image_tokens) if decoder_only else 0)


def guarded_post(
    logger: logging.Logger, url: str, payload: dict[str, Any]
) -> requests.Response:
    try:
        return requests.post(url, json=payload)
    except requests.RequestException:
        logger.exception("Connection to %s failed", url)
        resp = requests.Response()
        resp.status_code = 503
        resp._content = b"{}"
        return resp
