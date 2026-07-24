from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lmm_serving import config as cfg

from . import env

VALID_TP = {"TP1", "TP2", "TP4", "TP8"}


@dataclass(frozen=True)
class Node:
    address: str
    tp: str = "TP1"
    local_ranks: list[int] = field(default_factory=list)
    rank: int = 0  # image node only: global NCCL rank
    gpu_id: str = ""  # image node only
    numa_physcpu: str = ""  # image node only

    @property
    def host(self) -> str:
        return self.address.split(":")[0]

    @property
    def port(self) -> str:
        return self.address.split(":")[1]


@dataclass(frozen=True)
class Topology:
    setup: str
    monolith_nodes: list[Node] = field(default_factory=list)
    text_nodes: list[Node] = field(default_factory=list)
    image_nodes: list[Node] = field(default_factory=list)

    @property
    def hosts(self) -> set[str]:
        return {
            n.host for n in (*self.monolith_nodes, *self.text_nodes, *self.image_nodes)
        }


def tp_size(tp: str) -> int:
    return int(tp.removeprefix("TP"))


def load(setup: str, nodes_config: str | None = None) -> Topology:
    path = Path(cfg.cluster_config_path(setup, nodes_config))
    raw = yaml.safe_load(path.read_text()) or {}
    if setup == "monolith":
        nodes = [
            Node(
                n["node_address"],
                n["node_tp"],
                [int(r) for r in n.get("local_ranks", [])],
            )
            for n in raw.get("monolith_nodes", [])
        ]
        return Topology(setup, monolith_nodes=nodes)
    text = [
        Node(
            n["text_node_address"],
            n["text_node_tp"],
            [int(r) for r in n["local_ranks"]],
        )
        for n in raw.get("modserve_text_nodes", [])
    ]
    image = [
        Node(
            n["image_node_address"],
            n["image_node_tp"],
            rank=int(n["image_node_rank"]),
            gpu_id=str(n["image_node_gpu_id"]),
            numa_physcpu=str(n["numa_physcpu"]),
        )
        for n in raw.get("modserve_image_nodes", [])
    ]
    return Topology(setup, text_nodes=text, image_nodes=image)


def local_gpu_count() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return len([line for line in out.splitlines() if line.strip()])


def _all_nodes(t: Topology) -> list[tuple[str, Node]]:
    return (
        [("monolith", n) for n in t.monolith_nodes]
        + [("text", n) for n in t.text_nodes]
        + [("image", n) for n in t.image_nodes]
    )


def _gpu_id(n: Node) -> int | None:
    try:
        return int(n.gpu_id)
    except (TypeError, ValueError):
        return None


def validate(t: Topology) -> list[str]:
    problems: list[str] = []
    tagged = _all_nodes(t)

    if t.setup == "monolith" and not t.monolith_nodes:
        problems.append("monolith config has no monolith_nodes")
    if t.setup == "modserve":
        if not t.text_nodes:
            problems.append("modserve config has no modserve_text_nodes")
        if not t.image_nodes:
            problems.append("modserve config has no modserve_image_nodes")

    for kind, n in tagged:
        host, sep, port = n.address.partition(":")
        if not sep or not port.isdigit():
            problems.append(f"{kind} {n.address}: address must be host:port")
        if kind == "image":
            if n.tp != "TP1":
                problems.append(
                    f"image {n.address}: image encoders always run TP1, got {n.tp}"
                )
        elif n.tp not in VALID_TP:
            problems.append(
                f"{kind} {n.address}: unsupported {n.tp} (supported: {sorted(VALID_TP)})"
            )
    for n in t.text_nodes:
        if n.tp in VALID_TP and tp_size(n.tp) != len(n.local_ranks):
            problems.append(
                f"text {n.address}: {n.tp} but {len(n.local_ranks)} local_ranks "
                f"{n.local_ranks} — TP size must equal the number of GPUs"
            )

    # Monolith replicas may pin GPUs; the pin must match TP and not overlap.
    if t.setup == "monolith":
        for n in t.monolith_nodes:
            if (
                n.local_ranks
                and n.tp in VALID_TP
                and tp_size(n.tp) != len(n.local_ranks)
            ):
                problems.append(
                    f"monolith {n.address}: {n.tp} but {len(n.local_ranks)} local_ranks "
                    f"{n.local_ranks} — TP size must equal the number of GPUs"
                )
        for host in t.hosts:
            used = [
                r for n in t.monolith_nodes if n.host == host for r in n.local_ranks
            ]
            clashes = sorted(g for g, c in Counter(used).items() if c > 1)
            if clashes:
                problems.append(
                    f"monolith host {host}: GPU id(s) {clashes} assigned to more than one replica"
                )

    # Modserve NCCL ranks must be 0..N-1, and GPUs can't be shared on a host.
    if t.setup == "modserve":
        for n in t.image_nodes:
            if _gpu_id(n) is None:
                problems.append(
                    f"image {n.address}: image_node_gpu_id {n.gpu_id!r} must be a single GPU id"
                )
        ranks = sorted(n.rank for n in t.image_nodes)
        if ranks != list(range(len(ranks))):
            problems.append(f"image_node_rank must be contiguous 0..N-1, got {ranks}")
        for host in t.hosts:
            used = [
                g
                for n in t.image_nodes
                if n.host == host and (g := _gpu_id(n)) is not None
            ]
            used += [r for n in t.text_nodes if n.host == host for r in n.local_ranks]
            clashes = sorted(g for g, c in Counter(used).items() if c > 1)
            if clashes:
                problems.append(
                    f"host {host}: GPU id(s) {clashes} assigned to more than one node"
                )

    for host in t.hosts:
        need = sum(
            1 if kind == "image" else tp_size(n.tp)
            for kind, n in tagged
            if n.host == host and (kind == "image" or n.tp in VALID_TP)
        )
        if env.is_local(host):
            have = local_gpu_count()
            if have is not None and need > have:
                problems.append(f"host {host}: needs {need} GPUs but has {have}")

    pairs = []
    for _kind, n in tagged:
        host, sep, port = n.address.partition(":")
        if sep and port.isdigit():
            pairs.append((host, port))
    for (host, port), count in Counter(pairs).items():
        if count > 1:
            problems.append(f"host {host}: port {port} used by {count} nodes")

    return problems
