import os
import sys
from pathlib import Path

import yaml

_CFG_DIR = Path(__file__).resolve().parent  # src/lmm_serving/config
PKG_DIR = _CFG_DIR.parent  # src/lmm_serving (the package root)
REPO_ROOT = _CFG_DIR.parents[2]  # repo root (…/src/lmm_serving/config → repo)
# Make ${REPO_ROOT}/${PKG_DIR} expandable in config.yaml values (e.g. IMAGE_DIR).
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))
os.environ.setdefault("PKG_DIR", str(PKG_DIR))

with open(_CFG_DIR / "config.yaml") as _f:
    _CFG = yaml.safe_load(_f) or {}


def _resolve(key, default):
    if key in os.environ:
        raw = os.environ[key]
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, str):
            return os.path.expandvars(raw)
        return type(default)(raw)  # int / float
    return os.path.expandvars(default) if isinstance(default, str) else default


for _k, _v in _CFG.items():
    _val = _resolve(_k, _v)
    setattr(sys.modules[__name__], _k, _val)
    os.environ.setdefault(_k, str(_val))


def subsystem_dir(setup_type: str) -> Path:
    return PKG_DIR / setup_type


def cluster_config_path(setup_type: str, nodes_config: str | None = None) -> Path:
    return (
        REPO_ROOT
        / "configs"
        / setup_type
        / (nodes_config or os.environ["NODES_CONFIG"])
    )


def text_node_ips(setup_type: str, nodes_config: str | None = None) -> list[str]:
    cfg = yaml.safe_load(cluster_config_path(setup_type, nodes_config).read_text())
    if "monolith_nodes" in cfg:
        nodes = [n["node_address"] for n in cfg["monolith_nodes"]]
    else:
        nodes = [n["text_node_address"] for n in cfg["modserve_text_nodes"]]
    return list(dict.fromkeys(addr.split(":")[0] for addr in nodes))
