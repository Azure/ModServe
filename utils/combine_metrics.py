import pandas as pd
import glob
import os
import sys

from lmm_serving import config as cfg

run_id = sys.argv[1]
setup_type = sys.argv[2]
nodes_config = sys.argv[3] if len(sys.argv) > 3 else None

# Node IPs come from the active topology (deduped to unique hosts).
node_ips = cfg.text_node_ips(setup_type, nodes_config)
RUN_DIR = f"{cfg.REPO_ROOT}/metrics/{setup_type}/{run_id}"


def node_dir(ip):
    per_ip = os.path.join(RUN_DIR, ip)
    return per_ip if os.path.isdir(per_ip) else RUN_DIR


def read_across_nodes(filename):
    stem = filename[:-4] if filename.endswith(".csv") else filename
    frames = []
    for ip in node_ips:
        d = node_dir(ip)
        matches = sorted(glob.glob(os.path.join(d, f"{stem}*.csv")))
        if matches:
            frames.extend(pd.read_csv(p) for p in matches)
        else:
            print(f"WARN: no {stem}*.csv in {d} — skipping this node", file=sys.stderr)
    if not frames:
        raise FileNotFoundError(f"no {filename} found for any node in run {run_id}")
    return pd.concat(frames, ignore_index=True)


metrics = {
    "ttft_metrics.csv": read_across_nodes("ttft_metrics.csv"),
    "tbt_metrics.csv": read_across_nodes("tbt_metrics.csv"),
    "transfer_stats_metrics.csv": read_across_nodes("transfer_stats_metrics.csv"),
}

# Monolith encodes images inline on the text node, so it also emits this file.
if setup_type == "monolith":
    metrics["image_encoding_metrics_monolith.csv"] = read_across_nodes(
        "image_encoding_metrics_monolith.csv"
    )

out_dir = os.path.join(RUN_DIR, "combined")
os.makedirs(out_dir, exist_ok=True)
for filename, df in metrics.items():
    df.to_csv(os.path.join(out_dir, filename), index=False)
    print(f"wrote {os.path.join(out_dir, filename)} ({len(df)} rows)")
