import base64
import os
import pickle
import queue
import signal
import sys
import threading

import torch
from flask import Flask, request, jsonify
from torch.multiprocessing.reductions import reduce_tensor

from lmm_serving.common.logging_config import get_logger

logger = get_logger("modserve.tensor_transfer_text")
app = Flask(__name__)


def parse_transfer_ranks(spec: str) -> dict:
    owned = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if pair:
            g, gpu = pair.split(":")
            owned[int(g)] = int(gpu)
    return owned


OWNED_RANKS = parse_transfer_ranks(os.environ["TRANSFER_RANKS"])
assert len(OWNED_RANKS) == 1, f"recv node owns exactly one rank, got {OWNED_RANKS}"
RANK, GPU = next(iter(OWNED_RANKS.items()))
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
# Image nodes are ranks 0..N_IMAGE-1; text recv nodes are N_IMAGE..WORLD_SIZE-1.
N_IMAGE = int(os.environ["N_IMAGE_NODES"])
IMAGE_RANKS = list(range(N_IMAGE))

_jobs: dict = {}  # src_rank -> queue of (image_id, shape, result, done)
_pair_group: dict = {}  # src_rank -> ProcessGroup for the (src_rank, RANK) pair
pending: dict = {}  # image_id -> recv'd GPU tensor (kept alive until released)
pending_lock = threading.Lock()
RECV_TIMEOUT_S = float(os.environ.get("TENSOR_TRANSFER_TIMEOUT_S", "60"))


def normalize_image_id(image_id: str) -> str:
    return image_id.split(".")[0] if "." in image_id else image_id


def _ipc_handle(tensor: torch.Tensor) -> str:
    _, args = reduce_tensor(tensor)
    return base64.b64encode(pickle.dumps(args)).decode()


def _build_pair_groups():
    groups = {}
    for img in IMAGE_RANKS:
        for txt in range(N_IMAGE, WORLD_SIZE):
            g = torch.distributed.new_group(ranks=[img, txt])
            if RANK in (img, txt):
                groups[(img, txt)] = g
    return groups


def _warmup_pairs(groups):
    dummy = torch.zeros(1, dtype=torch.bfloat16, device=f"cuda:{GPU}")
    for img in IMAGE_RANKS:
        for txt in range(N_IMAGE, WORLD_SIZE):
            g = groups.get((img, txt))
            if RANK == img:
                torch.distributed.send(dummy, dst=txt, group=g)
            elif RANK == txt:
                torch.distributed.recv(torch.empty_like(dummy), src=img, group=g)
            torch.distributed.barrier()


def recv_worker(src_rank: int, group):
    torch.cuda.set_device(GPU)
    while True:
        image_id, shape, result, done = _jobs[src_rank].get()
        try:
            tensor = torch.empty(shape, dtype=torch.bfloat16, device=f"cuda:{GPU}")
            torch.distributed.recv(tensor=tensor, src=src_rank, group=group)
            with pending_lock:
                pending[image_id] = tensor
            result["handle"] = _ipc_handle(tensor)
        except Exception as e:
            logger.exception(
                "Recv failed for image_id=%s from rank=%s", image_id, src_rank
            )
            result["error"] = str(e)
        finally:
            done.set()


def nccl_worker():
    torch.cuda.set_device(GPU)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=RANK,
        world_size=WORLD_SIZE,
    )  # NB: no device_id -- eager init uses ncclCommSplit for new_group, which
    torch.distributed.barrier()
    groups = _build_pair_groups()
    _warmup_pairs(groups)
    for src_rank in IMAGE_RANKS:
        _jobs[src_rank] = queue.Queue()
        _pair_group[src_rank] = groups[(src_rank, RANK)]
        threading.Thread(
            target=recv_worker, args=(src_rank, _pair_group[src_rank]), daemon=True
        ).start()
    logger.info(
        "Tensor-transfer RECV node ready: rank=%s gpu=%s world=%s peers=%s, listening on :%s",
        RANK,
        GPU,
        WORLD_SIZE,
        len(IMAGE_RANKS),
        os.environ.get("TENSOR_TRANSFER_PORT", "5050"),
    )
    _ready.set()


_ready = threading.Event()


@app.route("/initiate_tensor_transfer_text_node", methods=["POST"])
def initiate():
    data = request.get_json()
    image_id = normalize_image_id(data.get("image_id", ""))
    shape = data.get("tensor_shape", [])
    src_rank = data.get("src_rank", 0)

    if not shape or any(int(d) <= 0 for d in shape):
        return (
            jsonify({"image_id": image_id, "error": f"invalid tensor_shape={shape}"}),
            400,
        )

    result, done = {}, threading.Event()
    q = _jobs.get(src_rank)
    if q is None:
        return (
            jsonify(
                {
                    "image_id": image_id,
                    "error": f"no recv worker for src_rank={src_rank}",
                }
            ),
            500,
        )
    q.put((image_id, shape, result, done))
    if not done.wait(timeout=RECV_TIMEOUT_S):
        return jsonify({"image_id": image_id, "error": "recv timed out"}), 504
    if "error" in result:
        return jsonify({"image_id": image_id, "error": result["error"]}), 500
    logger.debug(
        "Received image_id=%s from rank=%s; returning IPC handle", image_id, src_rank
    )
    return jsonify({"image_id": image_id, "ipc_handle": result["handle"]}), 200


@app.route("/release_tensor", methods=["POST"])
def release():
    image_id = normalize_image_id(request.get_json().get("image_id", ""))
    with pending_lock:
        pending.pop(image_id, None)
    return jsonify({"image_id": image_id}), 200


def signal_handler(*_):
    logger.info("Shutting down tensor-transfer RECV node")
    try:
        torch.distributed.destroy_process_group()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    logger.info(
        "Starting tensor-transfer RECV node: rank=%s gpu=%s world=%s master=%s:%s CVD=%s",
        RANK,
        GPU,
        WORLD_SIZE,
        os.environ["MASTER_ADDR"],
        os.environ["MASTER_PORT"],
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    threading.Thread(target=nccl_worker, daemon=True).start()
    _ready.wait()  # block until NCCL is up so we don't serve before the world is formed
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    app.run(
        debug=False,
        port=int(os.environ.get("TENSOR_TRANSFER_PORT", "5050")),
        host="0.0.0.0",
        threaded=True,
    )
