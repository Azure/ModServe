import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import requests
from transformers import AutoTokenizer

from lmm_serving import config as cfg
from lmm_serving.common.logging_config import get_logger

logger = get_logger("client.azure_lmm_trace")

QPM_LABEL = sys.argv[1]
TRACE_PATH = (
    f"{cfg.REPO_ROOT}/traces/experiment_traces/{cfg.TRACE}_scale{QPM_LABEL}.csv"
)
CLUSTER_MANAGER_URL = (
    f"http://{cfg.CLUSTER_MANAGER_IP}:{cfg.CLUSTER_MANAGER_PORT}/generate"
)
MAX_IMAGES = 8 if "Llama" in cfg.MODEL else 16
_req_timeout = os.environ.get("REQUEST_TIMEOUT_S")
REQUEST_TIMEOUT_S = float(_req_timeout) if _req_timeout else None

df = pd.read_csv(TRACE_PATH).fillna({"text_prompt": "", "image_ids": ""})
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL, trust_remote_code=True)

_STATS_LOCK = threading.Lock()
_stats = {"sent": 0, "succeeded": 0, "failed": 0}


def _record(success):
    with _STATS_LOCK:
        _stats["sent"] += 1
        _stats["succeeded" if success else "failed"] += 1


def send_request(prompt, n_text_tokens, image_ids, n_generated_tokens):
    payload = {
        "timestamp": str(datetime.now(timezone.utc)),
        "perf_time_start": time.perf_counter(),
        "req_id": f"req-{uuid.uuid4()}",
        "prompt": prompt,
        "num_text_tokens": int(n_text_tokens),
        "num_output_tokens": int(n_generated_tokens),
        "image_ids": image_ids,
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(n_generated_tokens),
        "ignore_eos": True,
        "stream": False,
    }
    try:
        r = requests.post(
            CLUSTER_MANAGER_URL,
            headers={"User-Agent": "Benchmark Client"},
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
        )
        _record(r.status_code == 200)
    except requests.RequestException:
        logger.exception("Request to %s failed", CLUSTER_MANAGER_URL)
        _record(False)


if __name__ == "__main__":
    prebuilt = []
    for idx in range(len(df)):
        prompt = df["text_prompt"][idx]
        image_ids = df["image_ids"][idx].split()[:MAX_IMAGES]
        n_text_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prebuilt.append(
            (prompt, n_text_tokens, image_ids, int(df["n_generated_tokens"][idx]))
        )

    logger.info(
        "Replaying %s: %d requests, cluster_manager=%s, model=%s, max_images=%d",
        TRACE_PATH,
        len(df),
        CLUSTER_MANAGER_URL,
        cfg.MODEL,
        MAX_IMAGES,
    )

    timestamps = df["Timestamp"].tolist()
    run_start = time.perf_counter()
    threads = []
    for idx in range(len(df)):
        wait = timestamps[idx] - (time.perf_counter() - run_start)
        if wait > 0:
            time.sleep(wait)
        t = threading.Thread(target=send_request, args=prebuilt[idx])
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    elapsed = time.perf_counter() - run_start
    logger.info(
        "Replay complete: %d sent, %d ok, %d failed in %.1f s",
        _stats["sent"],
        _stats["succeeded"],
        _stats["failed"],
        elapsed,
    )
