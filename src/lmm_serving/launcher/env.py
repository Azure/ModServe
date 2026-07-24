from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

from lmm_serving import config as cfg

REPO_ROOT = Path(cfg.REPO_ROOT)
PKG_DIR = Path(cfg.PKG_DIR)


def code_dir(setup: str) -> Path:
    return PKG_DIR / setup


def configs_dir(setup: str) -> Path:
    return REPO_ROOT / "configs" / setup


def metrics_dir(setup: str) -> Path:
    return REPO_ROOT / "metrics" / setup


def log_dir(setup: str, run_id: str | None = None) -> Path:
    base = REPO_ROOT / "logs" / setup
    return base / run_id if run_id else base


def _env_root() -> Path:
    # CONDA_PREFIX (set by Pixi and conda) wins; else the repo Pixi env, else config.
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        return Path(prefix)
    pixi_env = REPO_ROOT / ".pixi" / "envs" / "default"
    if pixi_env.exists():
        return pixi_env
    return Path(cfg.CONDA_HOME) / "envs" / cfg.CONDA_ENV


def python_exe() -> str:
    return str(_env_root() / "bin" / "python")


def bin_dir() -> str:
    return str(_env_root() / "bin")


def activated_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    e = dict(os.environ)
    e["PATH"] = str(_env_root() / "bin") + os.pathsep + e.get("PATH", "")
    e["CONDA_PREFIX"] = str(_env_root())
    e["CONDA_DEFAULT_ENV"] = cfg.CONDA_ENV
    for key, value in (extra or {}).items():
        e[key] = str(value)
    return e


def _own_ips() -> set[str]:
    ips = {"127.0.0.1", "localhost"}
    try:
        out = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        ips.update(line.split()[3].split("/")[0] for line in out.splitlines())
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    try:
        ips.add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    return ips


def is_local(host: str) -> bool:
    return host in _own_ips()
