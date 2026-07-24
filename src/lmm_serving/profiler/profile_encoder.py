from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from transformers import AutoModel

from lmm_serving.common.vision_tokens import INTERNVL_TOKENS_PER_TILE

ENCODER_ID = "OpenGVLab/InternViT-6B-448px-V2_5"
TILE = 448


def load_encoder():
    # bf16, sharded across visible GPUs via device_map (mirrors the image node).
    return AutoModel.from_pretrained(
        ENCODER_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()


def time_batch(model, batch: int, iters: int) -> float:
    x = torch.randn(batch, 3, TILE, TILE, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        for _ in range(2):  # warmup
            model(x)
    torch.cuda.synchronize()
    lats = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(x)
        end.record()
        torch.cuda.synchronize()
        lats.append(start.elapsed_time(end))
    return statistics.median(lats)


def profile(batches: list[int], iters: int) -> dict:
    model = load_encoder()
    gpus = torch.cuda.device_count()
    points = []
    for b in batches:
        lat = time_batch(model, b, iters)
        # load_unit = image-tokens/tile (constant for InternVL); batch-load = b*unit.
        points.append(
            {
                "parallelism": gpus,
                "batch": b,
                "load_unit": INTERNVL_TOKENS_PER_TILE,
                "latency_ms": round(lat, 3),
                "tokens_per_s": round(b * INTERNVL_TOKENS_PER_TILE * 1000.0 / lat, 1),
            }
        )
        print(
            f"  tp={gpus} batch={b:3d}  latency={lat:8.2f} ms  "
            f"{b * INTERNVL_TOKENS_PER_TILE * 1000.0 / lat:8.1f} tok/s",
            flush=True,
        )
    return {
        "stage": "encoder",
        "model": ENCODER_ID,
        "tp_gpus": gpus,
        "tokens_per_tile": INTERNVL_TOKENS_PER_TILE,
        "points": points,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    print(
        f"Encoder profile: {ENCODER_ID} on {torch.cuda.device_count()} GPU(s), "
        f"batches={batches} iters={args.iters}",
        flush=True,
    )
    t0 = time.time()
    result = profile(batches, args.iters)
    result["wall_s"] = round(time.time() - t0, 1)

    if args.out:
        import os

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
