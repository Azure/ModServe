import os
import socket
import sys

from lmm_serving import config as cfg
from lmm_serving.common.logging_config import get_logger
from lmm_serving.common import text_node

logger = get_logger("modserve.text_node")

MAX_NUM_IMAGES_PER_PROMPT = 400

HOSTNAME = sys.argv[1]
PORT = sys.argv[2]
TENSOR_PARALLEL = sys.argv[3]
# sys.argv[4] is a legacy GPU placeholder ("all"), ignored — GPUs come from CUDA_VISIBLE_DEVICES.
MODEL_ID = sys.argv[5] if len(sys.argv) > 5 else cfg.MODEL
# Architecture tag for the decoupled-image protocol (consumed by the vLLM backend).
MODEL_ARCH = "MLLAMA" if "Llama" in MODEL_ID else "INTERNVL"

INSTANCE_PRIORITY = getattr(cfg, "INSTANCE_SCHEDULING", "fcfs").lower() == "priority"


def _vllm_commands() -> dict:
    base = ["python3", "-m", "vllm.scripts", "serve", MODEL_ID, "--device", "cuda"]
    img_limit = 8 if "Llama" in MODEL_ID else MAX_NUM_IMAGES_PER_PROMPT
    default_seqs = 8 if "Llama" in MODEL_ID else 32
    max_num_seqs = str(getattr(cfg, "TEXT_MAX_NUM_SEQS", "") or default_seqs)
    mm = [
        "--limit-mm-per-prompt",
        f"image={img_limit}",
        "--max-num-seqs",
        max_num_seqs,
        "--disable-log-requests",
        "--enforce-eager",
    ]
    # SLO-driven scheduling: shorter (lower-priority-value) requests first.
    if INSTANCE_PRIORITY:
        mm += ["--scheduling-policy", "priority"]

    def cmd(tp, gpu_util, tail):
        util = (
            str(cfg.TEXT_GPU_UTIL)
            if str(getattr(cfg, "TEXT_GPU_UTIL", "")).strip()
            else gpu_util
        )
        pre = (
            ["--preemption-mode", str(cfg.TEXT_PREEMPTION_MODE)]
            if str(getattr(cfg, "TEXT_PREEMPTION_MODE", "")).strip()
            else []
        )
        return (
            base
            + [
                "--tensor-parallel-size",
                tp,
                "--dtype",
                "bfloat16",
                "--enable-chunked-prefill",
                "False",
                "--gpu-memory-utilization",
                util,
            ]
            + pre
            + mm
            + tail
        )

    if "Llama" in MODEL_ID:
        max_len = str(getattr(cfg, "TEXT_MAX_MODEL_LEN", "") or "32768")
        ll = ["--max-model-len", max_len, "--max_num_batched_tokens", max_len]
        return {
            "TP8": cmd("8", "0.7", ll),
            "TP4": cmd("4", "0.7", ll),
            "TP2": cmd("2", "0.80", ll),
            "TP1": cmd("1", "0.70", ll),
        }
    # InternVL
    trc = ["--trust-remote-code"]
    ctx = ["--max-model-len", "55296", "--max_num_batched_tokens", "55296"]
    return {
        "TP8": cmd("8", "0.70", trc + ctx),
        "TP4": cmd("4", "0.85", trc + ctx),
        "TP2": cmd("2", "0.85", trc + ctx),
        "TP1": cmd("1", "0.85", trc + ctx),
    }


def _image_content(data: dict) -> list:
    image_ids = data.get("images", [])
    shapes = data.get("image_id_shapes", [])
    src_ranks = data.get("src_ranks", [])
    dest_ranks = data.get("dest_ranks", [])
    source_urls = data.get("source_node_ips", [])
    return [
        {
            "type": "decoupled-multimodal-images",
            "image": {
                "model": MODEL_ARCH,
                "image_id": image_id,
                "source_url": source_urls[idx],
                "image_tensor_shape": shapes[idx],
                "src_rank": src_ranks[idx],
                "dest_rank": dest_ranks[idx],
            },
        }
        for idx, image_id in enumerate(image_ids)
    ]


if __name__ == "__main__":
    assert (
        TENSOR_PARALLEL in text_node.VALID_TP
    ), f"Invalid parallelism {TENSOR_PARALLEL}"
    command = _vllm_commands()[TENSOR_PARALLEL] + ["--port", text_node.vllm_port(PORT)]

    _visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _gpu_desc = (
        _visible
        if _visible
        else f"auto (first {TENSOR_PARALLEL.lstrip('TP')} device(s))"
    )
    logger.info(
        "Starting modserve text node on %s:%s | host=%s | model=%s | parallel=%s | GPUs=%s | image-protocol=%s",
        HOSTNAME,
        PORT,
        socket.gethostname(),
        MODEL_ID,
        TENSOR_PARALLEL,
        _gpu_desc,
        MODEL_ARCH,
    )

    logdir = os.environ["LOG_DIR"]
    text_node.serve(
        logger=logger,
        hostname=HOSTNAME,
        port=PORT,
        model_id=MODEL_ID,
        command=command,
        out_log=f"{logdir}/vllm-output-{HOSTNAME}-{PORT}.log",
        err_log=f"{logdir}/vllm-error-{HOSTNAME}-{PORT}.log",
        image_content=_image_content,
        priority_enabled=INSTANCE_PRIORITY,
    )
