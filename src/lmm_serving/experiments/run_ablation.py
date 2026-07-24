import argparse
import os
import subprocess
import sys

from lmm_serving import config as cfg

# Each variant adds one ModServe mechanism; vllm is the monolith baseline.
CONFIGS = {
    "vllm": ("monolith", {}),
    "decoup": ("modserve", {"SCHEDULER": "round_robin", "TEXT_SCHEDULER": "round_robin", "INSTANCE_SCHEDULING": "fcfs"}),
    "sched": ("modserve", {"SCHEDULER": "round_robin", "TEXT_SCHEDULER": "round_robin", "INSTANCE_SCHEDULING": "priority"}),
    "modserve": ("modserve", {"SCHEDULER": "least_loaded", "TEXT_SCHEDULER": "least_loaded", "INSTANCE_SCHEDULING": "priority"}),
}
MODELS = {
    "internvl": "OpenGVLab/InternVL2_5-26B",
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}


def served_rows(setup, run_id):
    f = cfg.REPO_ROOT / "metrics" / setup / run_id / "cluster_manager_metrics.csv"
    return sum(1 for _ in f.open()) - 1 if f.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--configs", default="vllm,decoup,sched,modserve")
    ap.add_argument("--models", default="internvl")
    ap.add_argument("--scales", default="7.77,3.89,2.59,1.94")
    ap.add_argument("--monolith-nodes", default="local.yaml")
    ap.add_argument("--modserve-nodes", default="local.yaml")
    ap.add_argument("--timeout", type=int, default=0, help="per-run seconds (0 = none)")
    ap.add_argument("--cleanup-hosts", default="", help="hosts to sweep between runs")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    nodes = {"monolith": a.monolith_nodes, "modserve": a.modserve_nodes}
    hosts = a.cleanup_hosts.replace(",", " ").split()
    cleanup = str(cfg.REPO_ROOT / "utils" / "cluster_cleanup.sh")

    for config in a.configs.split(","):
        setup, knobs = CONFIGS[config]
        nodes_config = nodes[setup]
        for model in a.models.split(","):
            for scale in a.scales.split(","):
                rid = f"{a.exp}_{config}_{model}_s{scale}"
                if not a.no_resume and served_rows(setup, rid) > 0:
                    print(f"[{rid}] skip (done)", flush=True)
                    continue
                env = {**os.environ, "MODEL": MODELS[model], "NODES_CONFIG": nodes_config, **knobs}
                cmd = [sys.executable, "-m", "lmm_serving.launcher", "run", setup,
                       scale, "--run-id", rid, "--nodes-config", nodes_config]
                if a.timeout:
                    cmd = ["timeout", "-s", "INT", "-k", "60", str(a.timeout)] + cmd
                print(f"[{rid}] {setup}/{nodes_config} {knobs}", flush=True)
                if a.dry_run:
                    print("    " + " ".join(cmd))
                    continue
                subprocess.run(cmd, env=env)
                if hosts:
                    subprocess.run(["bash", cleanup, *hosts], check=False)
                print(f"[{rid}] {served_rows(setup, rid)} rows", flush=True)


if __name__ == "__main__":
    main()
