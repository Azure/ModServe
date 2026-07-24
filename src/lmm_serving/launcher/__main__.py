from __future__ import annotations

import argparse
import sys
from datetime import datetime

from lmm_serving import config as cfg

from . import launch
from . import topology as topo

SETUPS = ("monolith", "modserve")


def _cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    nodes_config = args.nodes_config or cfg.NODES_CONFIG
    scales = [s.strip() for s in args.scales.split(",")]
    if args.run_id:
        run_ids = [r.strip() for r in args.run_id.split(",")]
        if len(run_ids) != len(scales):
            parser.error(
                f"--run-id ({len(run_ids)}) and scales ({len(scales)}) must pair up"
            )
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_ids = [f"{args.setup}_{ts}_scale{scale}" for scale in scales]
    for run_id, scale in zip(run_ids, scales):
        print(f"=== {args.setup}: {run_id} @ scale {scale} ===")
        if args.dry_run:
            launch.dry_run(args.setup, run_id, scale, nodes_config)
        else:
            launch.run(args.setup, run_id, scale, nodes_config)
        print(f"=== done: {run_id} ===\n")
    return 0


def _cmd_validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    nodes_config = args.nodes_config or cfg.NODES_CONFIG
    problems = topo.validate(topo.load(args.setup, nodes_config))
    if problems:
        print(f"INVALID: {args.setup}/{nodes_config}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: {args.setup}/{nodes_config} is valid")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="launcher", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="launch an experiment")
    p_run.add_argument("setup", choices=SETUPS)
    p_run.add_argument("scales", help="comma list, e.g. 8.0,4.0")
    p_run.add_argument(
        "--run-id",
        dest="run_id",
        help="comma list pairing with scales (default: auto timestamp)",
    )
    p_run.add_argument(
        "--nodes-config", help="topology file (default: config NODES_CONFIG)"
    )
    p_run.add_argument(
        "--dry-run", action="store_true", help="print the plan, don't launch"
    )
    p_run.set_defaults(func=_cmd_run)

    p_val = sub.add_parser("validate", help="check a topology without launching")
    p_val.add_argument("setup", choices=SETUPS)
    p_val.add_argument(
        "--nodes-config", help="topology file (default: config NODES_CONFIG)"
    )
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args()
    return args.func(args, parser)


if __name__ == "__main__":
    sys.exit(main())
