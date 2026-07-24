from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import requests
from openai import OpenAI

from lmm_serving import config as cfg

VLLM_PORT = 8001


def _launch_vllm(model: str, tp: int, log_path: str) -> subprocess.Popen:
    # Use this interpreter (the mLLM env), not bare python3 (base conda, no vLLM).
    # Overridable so multimodal backends (e.g. Llama-3.2-Vision) whose profiling
    # forward reserves extra activation memory don't OOM at the default budget.
    gpu_util = os.environ.get("PROFILE_GPU_UTIL", "0.85")
    max_len = os.environ.get("PROFILE_MAX_MODEL_LEN", "55296")
    cmd = [
        sys.executable,
        "-m",
        "vllm.scripts",
        "serve",
        model,
        "--device",
        "cuda",
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--trust-remote-code",
        "--disable-log-requests",
        "--max-model-len",
        max_len,
        "--max_num_batched_tokens",
        max_len,
        "--gpu-memory-utilization",
        gpu_util,
        "--port",
        str(VLLM_PORT),
    ]
    # vLLM's memory-profiling forward runs a dummy batch of `max_num_seqs`
    # multimodal prompts through the vision encoder at once. For image models
    # (Llama-3.2-Vision) the default (256) reserves >70 GiB of CLIP activations
    # and OOMs. Cap it via env when profiling multimodal backends.
    max_num_seqs = os.environ.get("PROFILE_MAX_NUM_SEQS")
    if max_num_seqs:
        cmd += ["--max-num-seqs", max_num_seqs]
    return subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


def _wait_healthy(proc, timeout_s=900) -> None:
    url = f"http://localhost:{VLLM_PORT}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited ({proc.returncode}) before ready")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("vLLM not healthy in time")


def _ttft_ms(client, model: str, n_tokens: int) -> float:
    # max_tokens=1 isolates prefill (TTFT).
    prompt = "word " * n_tokens
    start = time.perf_counter()
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        extra_body={"ignore_eos": True},
    )
    return (time.perf_counter() - start) * 1000.0


def _batched_ttft_ms(client, model: str, n_tokens: int, concurrency: int) -> float:
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(_ttft_ms, client, model, n_tokens) for _ in range(concurrency)
        ]
        lats = [f.result() for f in futs]
    return statistics.median(lats)


def profile(
    model: str, tp: int, contexts: list[int], batches: list[int], log_path: str
) -> dict:
    proc = _launch_vllm(model, tp, log_path)
    try:
        _wait_healthy(proc)
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")
        _ttft_ms(client, model, 64)  # warmup
        points = []
        for b in batches:
            for n in contexts:
                ms = (
                    _batched_ttft_ms(client, model, n, b)
                    if b > 1
                    else _ttft_ms(client, model, n)
                )
                # load_unit = per-request context length; batch-load = b*n prompt tokens.
                points.append(
                    {
                        "parallelism": tp,
                        "batch": b,
                        "load_unit": n,
                        "latency_ms": round(ms, 3),
                        "tokens_per_s": round(b * n * 1000.0 / ms, 1),
                    }
                )
                print(
                    f"  tp={tp} batch={b:2d} ctx={n:6d}  TTFT={ms:8.2f} ms  "
                    f"{b * n * 1000.0 / ms:9.1f} tok/s",
                    flush=True,
                )
        return {"stage": "backend", "model": model, "tp": tp, "points": points}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--contexts", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--model", default=cfg.MODEL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default="/tmp/profiler_vllm.log")
    args = ap.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    batches = [int(b) for b in args.batches.split(",")]

    print(
        f"Backend profile: {args.model} TP{args.tp} contexts={contexts} batches={batches}",
        flush=True,
    )
    t0 = time.time()
    result = profile(args.model, args.tp, contexts, batches, args.log)
    result["wall_s"] = round(time.time() - t0, 1)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
