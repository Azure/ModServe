import json
from pathlib import Path

import pandas as pd

TTFT_PARTS = [
    "total_image_processing_time",
    "image_transfer_latency",
    "waiting_time",
    "prefill_ms",
]


def load_run(run_dir):
    run_dir = Path(run_dir)
    cm = pd.read_csv(run_dir / "cluster_manager_metrics.csv")
    ttft = pd.read_csv(run_dir / "combined" / "ttft_metrics.csv")
    for col in ("waiting_time", "prefill_ms"):
        if col in ttft:
            ttft[col] = pd.to_numeric(ttft[col], errors="coerce")
    ttft = ttft.dropna(subset=["request_id"])
    # Chunked prefill emits one row per chunk: waiting_time repeats, prefill_ms is per-chunk.
    ttft = ttft.groupby("request_id", as_index=False).agg(
        waiting_time=("waiting_time", "max"),
        prefill_ms=("prefill_ms", "sum"),
    )
    df = cm.merge(ttft, left_on="vllm_req_id", right_on="request_id", how="left")

    transfer = run_dir / "combined" / "transfer_stats_metrics.csv"
    if transfer.exists():
        df = df.merge(pd.read_csv(transfer), on="vllm_req_id", how="left")

    # Monolith encodes images inline; fold that cost in (ModServe has no such file -> no-op).
    enc_file = run_dir / "combined" / "image_encoding_metrics_monolith.csv"
    if enc_file.exists():
        enc = pd.read_csv(enc_file)
        for col in ("preprocessing_time", "vision_encoding_time_ms"):
            enc[col] = pd.to_numeric(enc.get(col), errors="coerce")
        enc = (
            enc.dropna(subset=["request_id"])
            .groupby("request_id", as_index=False)
            .agg(
                preprocessing_time=("preprocessing_time", "max"),
                vision_encoding_time_ms=("vision_encoding_time_ms", "max"),
            )
        )
        enc["inline_image_processing_ms"] = enc["preprocessing_time"].fillna(0.0) + enc["vision_encoding_time_ms"].fillna(0.0)
        df = df.merge(
            enc[["request_id", "inline_image_processing_ms"]].rename(columns={"request_id": "vllm_req_id"}),
            on="vllm_req_id",
            how="left",
        )

    for col in TTFT_PARTS + ["inline_image_processing_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df else 0.0
    df["ttft_ms"] = sum(df[col] for col in TTFT_PARTS) + df["inline_image_processing_ms"]
    df["is_image"] = df["num_images"] > 0
    df["ok"] = df["http_code"] == 200
    return df


def summarize(df):
    ok = df[df["ok"]]
    out = {"n": len(df), "n_ok": len(ok), "n_failed": int((~df["ok"]).sum())}
    for label, sub in [("all", ok), ("text", ok[~ok["is_image"]]), ("image", ok[ok["is_image"]])]:
        if len(sub):
            out[f"p50_ttft_{label}"] = round(sub["ttft_ms"].quantile(0.50), 2)
            out[f"p99_ttft_{label}"] = round(sub["ttft_ms"].quantile(0.99), 2)
            out[f"mean_ttft_{label}"] = round(sub["ttft_ms"].mean(), 2)
    return out


if __name__ == "__main__":
    import sys

    print(json.dumps(summarize(load_run(sys.argv[1])), indent=2))
