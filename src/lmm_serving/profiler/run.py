from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from lmm_serving import config as cfg

from . import model as rfmodel


def _gpu_list(tp: int) -> str:
    return ",".join(str(i) for i in range(tp))


def _run(
    module: str, args: list[str], out: str, cuda_devices: str | None = None
) -> dict:
    env = dict(os.environ)
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    subprocess.run(
        [sys.executable, "-m", module, *args, "--out", out],
        cwd=str(cfg.REPO_ROOT),
        env=env,
        check=True,
    )
    with open(out) as f:
        return json.load(f)


def _sweep_encoder(tps, batches, iters, tmp) -> list[dict]:
    points = []
    for tp in tps:
        r = _run(
            "lmm_serving.profiler.profile_encoder",
            ["--batches", batches, "--iters", str(iters)],
            os.path.join(tmp, f"enc_tp{tp}.json"),
            cuda_devices=_gpu_list(tp),
        )
        points.extend(r["points"])
    return points


def _sweep_backend(tps, contexts, batches, tmp) -> list[dict]:
    points = []
    for tp in tps:
        r = _run(
            "lmm_serving.profiler.profile_backend",
            ["--tp", str(tp), "--contexts", contexts, "--batches", batches],
            os.path.join(tmp, f"bak_tp{tp}.json"),
            cuda_devices=_gpu_list(tp),
        )
        points.extend(r["points"])
    return points


def _mc_table(sm: rfmodel.StageModel, tps, slo_ms, batch_grid, load_grid) -> dict:
    return {tp: sm.max_capacity(tp, slo_ms, batch_grid, load_grid) for tp in tps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-tps", default="1")  # image nodes deploy at TP1
    ap.add_argument("--backend-tps", default="2,4")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--contexts", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--backend-batches", default="1,2,4")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--slo-encode-ms", type=float, default=2000.0)
    ap.add_argument("--slo-prefill-ms", type=float, default=5000.0)
    ap.add_argument("--deploy-encoder-tp", type=int, default=1)
    ap.add_argument(
        "--deploy-backend-tp", type=int, default=None
    )  # default: from TEXT_NODE_TP
    args = ap.parse_args()

    enc_tps = [int(t) for t in args.encoder_tps.split(",")]
    bak_tps = [int(t) for t in args.backend_tps.split(",")]
    enc_batches = [int(b) for b in args.batches.split(",")]
    bak_batches = [int(b) for b in args.backend_batches.split(",")]
    contexts = [int(c) for c in args.contexts.split(",")]
    deploy_bak_tp = args.deploy_backend_tp or int(
        str(getattr(cfg, "TEXT_NODE_TP", "TP2")).removeprefix("TP")
    )

    model_tag = cfg.MODEL.split("/")[-1]
    out_dir = os.path.join(str(cfg.REPO_ROOT), "profiles", model_tag)
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        enc_points = _sweep_encoder(enc_tps, args.batches, args.iters, tmp)
        bak_points = _sweep_backend(bak_tps, args.contexts, args.backend_batches, tmp)

    enc_model = rfmodel.StageModel.fit(enc_points, stage="encoder")
    bak_model = rfmodel.StageModel.fit(bak_points, stage="backend")

    enc_load_grid = [256]  # InternVL tokens/tile (constant)
    enc_mc = _mc_table(
        enc_model, enc_tps, args.slo_encode_ms, enc_batches, enc_load_grid
    )
    max_ctx = max(contexts)
    bak_load_grid = list(range(256, max_ctx + 1, 256))
    bak_mc = _mc_table(
        bak_model, bak_tps, args.slo_prefill_ms, bak_batches, bak_load_grid
    )

    # Scalars the autoscaler reads: M_C at each stage's deployed TP.
    enc_deploy = enc_mc.get(args.deploy_encoder_tp) or next(iter(enc_mc.values()))
    bak_deploy = (
        bak_mc.get(deploy_bak_tp)
        or bak_mc[min(bak_mc, key=lambda t: abs(t - deploy_bak_tp))]
    )

    profile = {
        "model": cfg.MODEL,
        "encoder": {
            "stage": "encoder",
            "tps": enc_tps,
            "points": enc_points,
            "fit_r2": round(enc_model.r2(), 4),
            "slo_ms": args.slo_encode_ms,
            "m_c_by_tp": {str(k): v for k, v in enc_mc.items()},
            "deploy_tp": args.deploy_encoder_tp,
            "m_c_per_s": enc_deploy["m_c_per_s"],
        },
        "backend": {
            "stage": "backend",
            "tps": bak_tps,
            "points": bak_points,
            "fit_r2": round(bak_model.r2(), 4),
            "slo_ms": args.slo_prefill_ms,
            "m_c_by_tp": {str(k): v for k, v in bak_mc.items()},
            "deploy_tp": deploy_bak_tp,
            "m_c_per_s": bak_deploy["m_c_per_s"],
        },
    }
    out = os.path.join(out_dir, "profile.json")
    with open(out, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nwrote {out}")
    print(
        f"  encoder: {len(enc_points)} pts, R^2={profile['encoder']['fit_r2']}, "
        f"M_C@TP{args.deploy_encoder_tp}={enc_deploy['m_c_per_s']:.0f} img-tok/s "
        f"(batch {enc_deploy['at_batch']})"
    )
    for tp in enc_tps:
        print(
            f"    TP{tp}: M_C={enc_mc[tp]['m_c_per_s']:.0f} img-tok/s (batch {enc_mc[tp]['at_batch']})"
        )
    print(
        f"  backend: {len(bak_points)} pts, R^2={profile['backend']['fit_r2']}, "
        f"M_C@TP{deploy_bak_tp}={bak_deploy['m_c_per_s']:.0f} prompt-tok/s "
        f"(batch {bak_deploy['at_batch']}, ctx {bak_deploy['at_load_unit']})"
    )
    for tp in bak_tps:
        print(
            f"    TP{tp}: M_C={bak_mc[tp]['m_c_per_s']:.0f} prompt-tok/s "
            f"(batch {bak_mc[tp]['at_batch']}, ctx {bak_mc[tp]['at_load_unit']})"
        )


if __name__ == "__main__":
    main()
