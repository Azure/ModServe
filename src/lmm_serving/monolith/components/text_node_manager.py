import os
import socket
import sys

from lmm_serving import config as cfg
from lmm_serving.common.logging_config import get_logger
from lmm_serving.common import text_node

logger = get_logger("monolith.text_node")

MAX_NUM_IMAGES_PER_PROMPT = 16

HOSTNAME = sys.argv[1]
PORT = sys.argv[2]
TENSOR_PARALLEL = sys.argv[3]
# sys.argv[4] is a legacy GPU placeholder ("all"), ignored — GPUs come from CUDA_VISIBLE_DEVICES.
MODEL_ID = sys.argv[5] if len(sys.argv) > 5 else cfg.MODEL
IMAGE_DIR = cfg.IMAGE_DIR


def _vllm_command(parallel: str) -> list:
    tp = parallel.lstrip("TP")
    img_limit = 8 if "Llama" in MODEL_ID else MAX_NUM_IMAGES_PER_PROMPT
    max_num_seqs = str(getattr(cfg, "MONOLITH_MAX_NUM_SEQS", 16))
    default_util = "0.85" if "Llama" in MODEL_ID else "0.70"
    gpu_util = str(getattr(cfg, "MONOLITH_GPU_UTIL", "") or default_util)
    common = [
        "python3",
        "-m",
        "vllm.scripts",
        "serve",
        MODEL_ID,
        "--device",
        "cuda",
        "--tensor-parallel-size",
        tp,
        "--dtype",
        "bfloat16",
        "--enable-chunked-prefill",
        "False",
        "--gpu-memory-utilization",
        gpu_util,
        "--limit-mm-per-prompt",
        f"image={img_limit}",
        "--max-num-seqs",
        max_num_seqs,
        "--disable-log-requests",
        "--enforce-eager",
    ]
    if "Llama" in MODEL_ID:
        max_len = str(getattr(cfg, "MONOLITH_MAX_MODEL_LEN", "") or "32768")
        cmd = common + [
            "--max-model-len",
            max_len,
            "--max_num_batched_tokens",
            max_len,
            "--allowed-local-media-path",
            IMAGE_DIR,
        ]
    else:  # InternVL
        cmd = common + [
            "--trust-remote-code",
            "--max-model-len",
            "55296",
            "--max_num_batched_tokens",
            "55296",
            "--allowed-local-media-path",
            IMAGE_DIR,
        ]
        if tp == "8":
            # Pin the 8-GPU InternVL case to NUMA node 1 (where those GPUs live).
            cmd = ["numactl", "--cpunodebind=1", "--membind=1"] + cmd
    return cmd + ["--port", text_node.vllm_port(PORT)]


def _image_content(data: dict) -> list:
    return [
        {"type": "image_url", "image_url": {"url": f"file://{IMAGE_DIR}/{img}"}}
        for img in data.get("image_ids", [])
    ]


if __name__ == "__main__":
    assert (
        TENSOR_PARALLEL in text_node.VALID_TP
    ), f"Invalid parallelism {TENSOR_PARALLEL}"
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _gpu_desc = (
        _visible
        if _visible
        else f"auto (first {TENSOR_PARALLEL.lstrip('TP')} device(s))"
    )
    logger.info(
        "Starting monolith text node on %s:%s | host=%s | model=%s | parallel=%s | GPUs=%s",
        HOSTNAME,
        PORT,
        socket.gethostname(),
        MODEL_ID,
        TENSOR_PARALLEL,
        _gpu_desc,
    )

    logdir = os.environ["LOG_DIR"]
    text_node.serve(
        logger=logger,
        hostname=HOSTNAME,
        port=PORT,
        model_id=MODEL_ID,
        command=_vllm_command(TENSOR_PARALLEL),
        out_log=f"{logdir}/output-{PORT}.log",
        err_log=f"{logdir}/error-{PORT}.log",
        image_content=_image_content,
    )
