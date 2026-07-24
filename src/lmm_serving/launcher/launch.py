from __future__ import annotations

import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from lmm_serving import config as cfg

from . import env
from . import topology as topo

RECV_PORT = "5051"
READY_TIMEOUT_S = 1800
CLUSTER_MANAGER_TIMEOUT_S = 120


@dataclass
class Node:
    name: str
    host: str
    argv: list[str]
    env: dict[str, str]
    log_file: Path
    ready_port: int | None = (
        None  # text/vLLM nodes only; image nodes are fire-and-forget
    )


def _component(setup: str, name: str) -> str:
    return str(env.code_dir(setup) / "components" / name)


def _metrics_env(setup: str, run_id: str, tag: str = "") -> dict[str, str]:
    m = env.metrics_dir(setup) / run_id
    sfx = f"_{tag}" if tag else ""
    return {
        "VISION_MODEL_METRICS_PATH": str(
            m / f"image_encoding_metrics_monolith{sfx}.csv"
        ),
        "TTFT_METRICS": str(m / f"ttft_metrics{sfx}.csv"),
        "TBT_METRICS": str(m / f"tbt_metrics{sfx}.csv"),
        "TRANSFER_STATS_FILE": str(m / f"transfer_stats_metrics{sfx}.csv"),
    }


def _monolith_nodes(t: topo.Topology, run_id: str) -> list[Node]:
    setup, logs = t.setup, env.log_dir(t.setup, run_id)
    nodes = []
    for n in t.monolith_nodes:
        e = {"LOG_DIR": str(logs), **_metrics_env(setup, run_id, n.port)}
        if n.local_ranks:
            e["CUDA_VISIBLE_DEVICES"] = ",".join(str(r) for r in n.local_ranks)
        nodes.append(
            Node(
                name=f"text {n.address}",
                host=n.host,
                argv=[
                    env.python_exe(),
                    _component(setup, "text_node_manager.py"),
                    n.host,
                    n.port,
                    n.tp,
                    "all",
                    cfg.MODEL,
                ],
                env=e,
                log_file=logs / f"text_node-{n.port}.log",
                ready_port=int(n.port),
            )
        )
    return nodes


def _modserve_nodes(t: topo.Topology, run_id: str) -> list[Node]:
    setup, logs = t.setup, env.log_dir(t.setup, run_id)
    nodes: list[Node] = []

    rank0_host = min(t.image_nodes, key=lambda n: n.rank).host
    n_image = len(t.image_nodes)
    n_text = len(t.text_nodes)
    world_size = n_image + n_text
    multi_host = len(t.hosts) > 1

    nccl = {
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "NCCL_DEBUG_SUBSYS": os.environ.get("NCCL_DEBUG_SUBSYS", "COLL"),
        "WORLD_SIZE": str(world_size),
        "N_IMAGE_NODES": str(n_image),
        "MASTER_PORT": str(cfg.NCCL_MASTER_PORT),
    }
    if multi_host:
        nccl["MASTER_ADDR"] = rank0_host
        nccl["NCCL_SOCKET_IFNAME"] = cfg.NCCL_SOCKET_IFNAME
        if getattr(cfg, "NCCL_IB_DISABLE", False):
            nccl["NCCL_IB_DISABLE"] = "1"
        else:
            nccl["NCCL_IB_HCA"] = cfg.NCCL_IB_HCA
    else:
        # Single box: rendezvous over loopback, transfer over NVLink (no IB).
        nccl["MASTER_ADDR"] = cfg.NCCL_MASTER_ADDR
        nccl["NCCL_IB_DISABLE"] = "1"
        nccl["NCCL_SOCKET_IFNAME"] = "lo"

    for n in t.image_nodes:
        nodes.append(
            Node(
                name=f"image {n.address} (gpu {n.gpu_id})",
                host=n.host,
                argv=[
                    "numactl",
                    "--cpunodebind=0",
                    f"--physcpubind={n.numa_physcpu}",
                    "--membind=0",
                    env.python_exe(),
                    _component(setup, "image_node_manager.py"),
                    n.host,
                    n.port,
                    "TP1",
                    n.gpu_id,
                    cfg.MODEL,
                ],
                env={**nccl, "IMAGE_NODE_RANK": str(n.rank)},
                log_file=logs / f"image_node-{n.host}-{n.port}.log",
                ready_port=int(n.port),
            )
        )

    for j, text in enumerate(t.text_nodes):
        recv_rank = n_image + j
        recv_port = str(int(RECV_PORT) + j)
        nodes.append(
            Node(
                name=f"transfer-recv {text.address} (rank {recv_rank})",
                host=text.host,
                argv=[
                    env.python_exe(),
                    _component(setup, "tensor_transfer_text_node.py"),
                ],
                env={
                    **nccl,
                    "CUDA_VISIBLE_DEVICES": ",".join(str(r) for r in text.local_ranks),
                    "TRANSFER_RANKS": f"{recv_rank}:0",
                    "TENSOR_TRANSFER_PORT": recv_port,
                },
                log_file=logs / f"transfer-recv-{text.host}-{text.port}.log",
            )
        )

    for j, text in enumerate(t.text_nodes):
        recv_port = str(int(RECV_PORT) + j)
        nodes.append(
            Node(
                name=f"text {text.address}",
                host=text.host,
                argv=[
                    env.python_exe(),
                    _component(setup, "text_node_manager.py"),
                    text.host,
                    text.port,
                    text.tp,
                    "all",
                    cfg.MODEL,
                ],
                env={
                    "LOG_DIR": str(logs),
                    **_metrics_env(setup, run_id, text.port),
                    "CUDA_VISIBLE_DEVICES": ",".join(str(r) for r in text.local_ranks),
                    "TENSOR_TRANSFER_RECV_PORT": recv_port,
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                },
                log_file=logs / f"text_node-{text.host}-{text.port}.log",
                ready_port=int(text.port),
            )
        )
    return nodes


def _cluster_manager_node(setup: str, run_id: str, nodes_config: str) -> Node:
    metrics_file = env.metrics_dir(setup) / run_id / "cluster_manager_metrics.csv"
    return Node(
        name=f"cluster-manager {cfg.CLUSTER_MANAGER_IP}:{cfg.CLUSTER_MANAGER_PORT}",
        host=cfg.CLUSTER_MANAGER_IP,
        argv=[
            env.python_exe(),
            _component(setup, "cluster_manager.py"),
            cfg.CLUSTER_MANAGER_IP,
            str(cfg.CLUSTER_MANAGER_PORT),
            str(metrics_file),
            nodes_config,
        ],
        env={
            "CLUSTER_CONFIG_DIR": str(env.configs_dir(setup)),
            "METRICS_FILE": str(metrics_file),
        },
        log_file=env.log_dir(setup, run_id) / "cluster_manager.log",
    )


def _build_nodes(t: topo.Topology, run_id: str) -> list[Node]:
    if t.setup == "monolith":
        return _monolith_nodes(t, run_id)
    return _modserve_nodes(t, run_id)


def _load_validated(setup: str, nodes_config: str) -> topo.Topology:
    t = topo.load(setup, nodes_config)
    problems = topo.validate(t)
    if problems:
        sys.exit(
            "Invalid topology ({}):\n  - {}".format(
                nodes_config, "\n  - ".join(problems)
            )
        )
    return t


def _ssh_argv(node: Node) -> list[str]:
    forward = {
        "PATH": f"{env.bin_dir()}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HF_HOME": str(cfg.HF_HOME),
        "HF_HUB_OFFLINE": str(cfg.HF_HUB_OFFLINE),
        **node.env,
    }
    envset = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(forward.items()))
    cmd = " ".join(shlex.quote(a) for a in node.argv)
    remote = f"cd {shlex.quote(str(env.REPO_ROOT))} && exec env {envset} {cmd}"
    return ["ssh", "-T", "-o", "StrictHostKeyChecking=accept-new", node.host, remote]


def _spawn(node: Node):
    node.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(node.log_file, "w")
    if env.is_local(node.host):
        argv = node.argv
        kw = {"env": env.activated_env(node.env), "cwd": str(env.REPO_ROOT)}
    else:
        argv, kw = _ssh_argv(node), {}
    proc = subprocess.Popen(
        argv, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True, **kw
    )
    print(f"  started  {node.name}")
    return proc, log_fh


def _terminate(proc, log_fh) -> None:
    def kill(sig):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    kill(signal.SIGTERM)
    for _ in range(30):
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    if proc.poll() is None:
        kill(signal.SIGKILL)
    try:
        log_fh.close()
    except OSError:
        pass


def _force_cleanup(t: topo.Topology) -> None:
    subprocess.run(
        ["bash", str(env.REPO_ROOT / "utils" / "cluster_cleanup.sh"), *sorted(t.hosts)],
        check=False,
    )


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex((host, port)) == 0


def _wait_ready(node: Node, proc) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{node.name} exited ({proc.returncode}) before ready — see {node.log_file}"
            )
        if _port_open(node.host, node.ready_port):
            return
        time.sleep(5)
    raise TimeoutError(
        f"{node.name} not ready after {READY_TIMEOUT_S}s — see {node.log_file}"
    )


def _wait_cluster_manager() -> None:
    deadline = time.monotonic() + CLUSTER_MANAGER_TIMEOUT_S
    while time.monotonic() < deadline:
        if _port_open(cfg.CLUSTER_MANAGER_IP, int(cfg.CLUSTER_MANAGER_PORT)):
            return
        time.sleep(2)
    print(
        f"  cluster-manager port {cfg.CLUSTER_MANAGER_PORT} not open after {CLUSTER_MANAGER_TIMEOUT_S}s; proceeding anyway"
    )


def _replay_trace(scale: str) -> None:
    # LMM_CLIENT swaps the workload generator while reusing startup/teardown.
    custom = os.environ.get("LMM_CLIENT")
    if custom:
        print(f"  running client {custom} ...")
        subprocess.run(
            [env.python_exe(), str(env.REPO_ROOT / custom)],
            env=env.activated_env(),
            cwd=str(env.REPO_ROOT),
            check=False,
        )
        return
    print(f"  replaying trace (scale {scale}) ...")
    subprocess.run(
        [
            env.python_exe(),
            str(env.REPO_ROOT / "client" / "azure_lmm_trace_client.py"),
            scale,
        ],
        env=env.activated_env(),
        cwd=str(env.REPO_ROOT),
        check=False,
    )


def _prepare_logs(setup: str, run_id: str) -> None:
    logs = env.log_dir(setup, run_id)
    if logs.exists():
        shutil.rmtree(logs)
    logs.mkdir(parents=True)


def _archive_logs(setup: str, run_id: str) -> None:
    src, dst = env.log_dir(setup, run_id), env.metrics_dir(setup) / run_id / "logs"
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for log in src.glob("*.log"):
        shutil.copy(log, dst / log.name)


def _combine_metrics(run_id: str, setup: str, nodes_config: str) -> None:
    subprocess.run(
        [
            env.python_exe(),
            str(env.REPO_ROOT / "utils" / "combine_metrics.py"),
            run_id,
            setup,
            nodes_config,
        ],
        env=env.activated_env(),
        cwd=str(env.REPO_ROOT),
        check=False,
    )


def _pull_metrics(t: topo.Topology, run_id: str) -> None:
    run_dir = env.metrics_dir(t.setup) / run_id
    text_nodes = t.monolith_nodes if t.setup == "monolith" else t.text_nodes
    for host in {n.host for n in text_nodes if not env.is_local(n.host)}:
        dest = run_dir / host
        dest.mkdir(parents=True, exist_ok=True)
        # only the top-level *.csv (skip the combined/ subdir); missing files are fine.
        subprocess.run(
            [
                "rsync",
                "-a",
                "-e",
                "ssh -o StrictHostKeyChecking=accept-new",
                "--include=*.csv",
                "--exclude=*",
                f"{host}:{env.REPO_ROOT}/metrics/{t.setup}/{run_id}/",
                f"{dest}/",
            ],
            check=False,
        )


def _check_ports_free(setup: str, nodes: list[Node]) -> None:
    ports = [
        (n.host, n.ready_port) for n in nodes if n.ready_port and env.is_local(n.host)
    ]
    ports.append((cfg.CLUSTER_MANAGER_IP, int(cfg.CLUSTER_MANAGER_PORT)))
    busy = [f"{h}:{p}" for h, p in ports if _port_open(h, p)]
    if busy:
        sys.exit(
            f"ports already in use: {', '.join(busy)} — a previous run may still be live; "
            f"run: bash utils/cluster_cleanup.sh"
        )


def _clean_run_dirs(t: topo.Topology, run_id: str) -> None:
    rel = env.metrics_dir(t.setup) / run_id
    shutil.rmtree(rel, ignore_errors=True)
    text_nodes = t.monolith_nodes if t.setup == "monolith" else t.text_nodes
    for host in {n.host for n in text_nodes if not env.is_local(n.host)}:
        subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                host,
                f"rm -rf {shlex.quote(str(rel))}",
            ],
            check=False,
        )


def run(setup: str, run_id: str, scale: str, nodes_config: str) -> None:
    t = _load_validated(setup, nodes_config)
    _clean_run_dirs(t, run_id)
    nodes = _build_nodes(t, run_id)
    cluster_manager = _cluster_manager_node(setup, run_id, nodes_config)
    _check_ports_free(setup, nodes)
    _prepare_logs(setup, run_id)
    logs = env.log_dir(setup, run_id)
    print(f"  logs     {logs}  (tail -f {logs}/*.log)")

    running = []
    try:
        for node in nodes:
            proc, log_fh = _spawn(node)
            running.append((proc, log_fh))
        for node, (proc, log_fh) in zip(nodes, running):
            if node.ready_port is not None:
                _wait_ready(node, proc)
                print(f"  ready    {node.name}")
        proc, log_fh = _spawn(cluster_manager)
        running.append((proc, log_fh))
        _wait_cluster_manager()
        _replay_trace(scale)
    finally:
        _archive_logs(setup, run_id)
        for proc, log_fh in reversed(running):
            _terminate(proc, log_fh)
        _force_cleanup(t)
        print("  torn down")
    _pull_metrics(t, run_id)
    _combine_metrics(run_id, setup, nodes_config)
    print(f"  metrics  {env.metrics_dir(setup) / run_id}")


def dry_run(setup: str, run_id: str, scale: str, nodes_config: str) -> None:
    t = _load_validated(setup, nodes_config)
    nodes = _build_nodes(t, run_id) + [
        _cluster_manager_node(setup, run_id, nodes_config)
    ]
    print(f"OK: topology valid ({setup}/{nodes_config})\n")
    for node in nodes:
        where = "local" if env.is_local(node.host) else f"ssh {node.host}"
        print(f"[{node.name}]  host={node.host} ({where})")
        print(f"  argv: {' '.join(node.argv)}")
        if node.env:
            print(
                f"  env : {', '.join(f'{k}={v}' for k, v in sorted(node.env.items()))}"
            )
        print(f"  log : {node.log_file}")
        if node.ready_port is not None:
            print(f"  ready when port {node.ready_port} is open")
        print()
    print(f"client: {env.python_exe()} client/azure_lmm_trace_client.py {scale}")
