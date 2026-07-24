from __future__ import annotations

from typing import Any

from flask import Flask, request, jsonify, Response
import requests
import threading
import time
import os
import json
import sys
import uuid
import yaml
import pandas as pd
import signal

from PIL import Image

from lmm_serving import config as cfg
from lmm_serving.common.logging_config import get_logger
from lmm_serving.common import scheduling
from lmm_serving.common import autoscaling
from lmm_serving.common.vision_tokens import vision_tokens

logger = get_logger("modserve.cluster_manager")

CLUSTER_MANAGER_IP = sys.argv[1]  # 127.0.0.1
CLUSTER_MANAGER_PORT = sys.argv[2]  # 7070
METRICS_FILE = sys.argv[3]
# Optional 4th arg: cluster config filename inside configs/<setup>/.
NODES_CONFIG = (
    sys.argv[4]
    if len(sys.argv) > 4
    else os.environ.get("NODES_CONFIG", "local.yaml")
)
IMAGE_METRICS_FILE = (
    METRICS_FILE[: METRICS_FILE.rfind("/") + 1] + "image_node_metrics.csv"
)

# create an empty file for image encoding metrics, overwrite if exists
metrics_df = scheduling.request_metrics_df()

image_metrics_df = pd.DataFrame(
    columns=[
        "req_id",
        "image_id",
        "num_image_tokens",
        "waiting_time",
        "preprocessing_time",
        "encoding_time",
        "e2e_time",
    ]
)
image_metrics_df = image_metrics_df.set_index("req_id")

# text node : [image_nodes]
modserve_text_nodes = []
modserve_image_nodes = []

image_node_ranks = {}

with open(f"{os.environ['CLUSTER_CONFIG_DIR']}/{NODES_CONFIG}") as f:
    modserve_config = yaml.safe_load(f)
    for modserve_text_node in modserve_config["modserve_text_nodes"]:
        modserve_text_nodes.append(modserve_text_node["text_node_address"])

    for modserve_image_node in modserve_config["modserve_image_nodes"]:
        modserve_image_nodes.append(modserve_image_node["image_node_address"])
        image_node_ranks[modserve_image_node["image_node_address"]] = (
            modserve_image_node["image_node_rank"]
        )

RECV_RANK = len(modserve_image_nodes)
text_recv_rank = {addr: RECV_RANK + j for j, addr in enumerate(modserve_text_nodes)}

logger.info(
    "Modserve cluster_manager starting: listen=%s:%s config=%s text_nodes=%d image_nodes=%d",
    CLUSTER_MANAGER_IP,
    CLUSTER_MANAGER_PORT,
    NODES_CONFIG,
    len(modserve_text_nodes),
    len(modserve_image_nodes),
)
logger.debug(
    "Loaded cluster config: text_nodes=%s image_nodes=%s recv_rank=%s image_node_ranks=%s",
    modserve_text_nodes,
    modserve_image_nodes,
    RECV_RANK,
    image_node_ranks,
)

DATA_LOCK = threading.Lock()
METRICS_FILE_LOCK = threading.Lock()
IMAGE_METRICS_DF_LOCK = threading.Lock()

app = Flask(__name__)


def _post(url: str, payload: dict[str, Any]) -> requests.Response:
    return scheduling.guarded_post(logger, url, payload)


image_status_tracker = {}

MODEL_ARCH = "MLLAMA" if "Llama" in cfg.MODEL else "INTERNVL"
SCHEDULER = getattr(cfg, "SCHEDULER", "round_robin")
TEXT_SCHEDULER = getattr(cfg, "TEXT_SCHEDULER", "round_robin")
INSTANCE_PRIORITY = getattr(cfg, "INSTANCE_SCHEDULING", "fcfs").lower() == "priority"
DECODER_ONLY = MODEL_ARCH == "INTERNVL"  # CroAttn (MLLAMA) vs DecOnly (InternVL)
TEXT_IMG_WEIGHT = float(getattr(cfg, "AUTOSCALE_TEXT_IMG_WEIGHT", 0.0) or 0.0)
if TEXT_IMG_WEIGHT <= 0.0:
    TEXT_IMG_WEIGHT = 1.0 if DECODER_ONLY else 0.35

text_node_scheduler = scheduling.make_scheduler(TEXT_SCHEDULER, modserve_text_nodes)
image_node_scheduler = scheduling.make_scheduler(SCHEDULER, modserve_image_nodes)
logger.info(
    "Routing schedulers: image=%s text=%s instance_priority=%s (model_arch=%s)",
    SCHEDULER,
    TEXT_SCHEDULER,
    INSTANCE_PRIORITY,
    MODEL_ARCH,
)


def _text_load(n_text_tokens: int, n_image_tokens: int) -> int:
    return int(n_text_tokens + TEXT_IMG_WEIGHT * n_image_tokens)


TEXT_INFLIGHT_IMAGE_TOKEN_BUDGET = int(
    getattr(cfg, "TEXT_INFLIGHT_IMAGE_TOKEN_BUDGET", 250000)
)
_inflight = {}
_inflight_cv = threading.Condition()


def _admit_image_tokens(node_ip: str, n_image_tokens: int) -> None:
    n = min(n_image_tokens, TEXT_INFLIGHT_IMAGE_TOKEN_BUDGET)
    with _inflight_cv:
        while _inflight.get(node_ip, 0) + n > TEXT_INFLIGHT_IMAGE_TOKEN_BUDGET:
            _inflight_cv.wait()
        _inflight[node_ip] = _inflight.get(node_ip, 0) + n


def _release_image_tokens(node_ip: str, n_image_tokens: int) -> None:
    n = min(n_image_tokens, TEXT_INFLIGHT_IMAGE_TOKEN_BUDGET)
    with _inflight_cv:
        _inflight[node_ip] = max(0, _inflight.get(node_ip, 0) - n)
        _inflight_cv.notify_all()


# Autoscaling stuff
AUTOSCALE = getattr(cfg, "AUTOSCALE", False)
AUTOSCALE_INTERVAL_S = getattr(cfg, "AUTOSCALE_INTERVAL_S", 30)
AUTOSCALE_WINDOW_S = getattr(cfg, "AUTOSCALE_WINDOW_S", 60)

image_load_window = autoscaling.LoadWindow(AUTOSCALE_WINDOW_S)
text_load_window = autoscaling.LoadWindow(AUTOSCALE_WINDOW_S)
autoscaler = None  # started in __main__ when AUTOSCALE is on


def _load_profile_mc() -> tuple[float, float]:
    path = f"{cfg.REPO_ROOT}/profiles/{cfg.MODEL.split('/')[-1]}/profile.json"
    enc_mc = bak_mc = 0.0
    try:
        with open(path) as f:
            prof = json.load(f)
        enc_mc = float(prof["encoder"]["m_c_per_s"])
        bak_mc = float(prof["backend"]["m_c_per_s"])
    except (OSError, KeyError, ValueError) as e:
        logger.warning(
            "No usable profile at %s (%s); autoscaler will keep all instances active",
            path,
            e,
        )
    enc_override = float(getattr(cfg, "AUTOSCALE_IMAGE_MC", 0) or 0)
    bak_override = float(getattr(cfg, "AUTOSCALE_TEXT_MC", 0) or 0)
    if enc_override > 0:
        enc_mc = enc_override
    if bak_override > 0:
        bak_mc = bak_override
    logger.info("autoscaler M_C: encoder=%.0f backend=%.0f tok/s", enc_mc, bak_mc)
    return enc_mc, bak_mc


def _image_weight(image_id: str) -> int:

    if MODEL_ARCH == "MLLAMA":
        return vision_tokens(MODEL_ARCH, 0, 0)
    try:
        with Image.open(f"{cfg.IMAGE_DIR}/{image_id}") as im:
            wh = im.size  # (w, h) from header, no decode
    except Exception as e:
        logger.warning(
            "Could not read dims for image_id=%s (%s); using single-tile weight",
            image_id,
            e,
        )
        return vision_tokens(MODEL_ARCH, 1, 1)
    return vision_tokens(MODEL_ARCH, wh[0], wh[1])


def populate_metrics_df(req_id: str, vllm_req_id: str, **kwargs: Any) -> None:
    scheduling.record_request_metrics(
        metrics_df, METRICS_FILE_LOCK, req_id, vllm_req_id, **kwargs
    )


def populate_image_metrics_df(
    req_id: str,
    image_id: str,
    num_image_tokens: int,
    waiting_time: float,
    preprocessing_time: float,
    encoding_time: float,
    e2e_time: float,
) -> None:
    with IMAGE_METRICS_DF_LOCK:
        image_metrics_df.loc[req_id] = [
            image_id,
            num_image_tokens,
            waiting_time,
            preprocessing_time,
            encoding_time,
            e2e_time,
        ]


@app.route("/generate", methods=["POST"])
def process_request() -> tuple[Response, int]:
    data = request.get_json()
    e2e_start_time = time.perf_counter()
    cluster_manager_waiting_time = (
        time.perf_counter() - data.get("perf_time_start", time.perf_counter())
    ) * 1000

    # request id
    req_id = data.get("req_id", "")
    # text prompt
    prompt = data.get("prompt", "")
    n_text_tokens = data.get("num_text_tokens", 0)
    n_generated_tokens = data.get("num_output_tokens", 1)
    timestamp = data.get("timestamp", 0)

    image_ids = data.get("image_ids", [])
    req_type = "IMAGE" if len(image_ids) > 0 else "TEXT"

    logger.debug(
        "Received %s request req_id=%s images=%s", req_type, req_id, len(image_ids)
    )
    if req_type == "TEXT":
        logger.debug(
            "Request ID: %s %s Number of text tokens: %s Text prompt: %s",
            req_id,
            req_type,
            n_text_tokens,
            prompt,
        )
    else:
        logger.debug(
            "Request ID: %s %s Number of text tokens: %s Text prompt: %s Number of images: %s Image IDs: %s",
            req_id,
            req_type,
            n_text_tokens,
            prompt,
            len(image_ids),
            image_ids,
        )

    populate_metrics_df(
        req_id=req_id,
        vllm_req_id="",
        timestamp=timestamp,
        num_images=len(image_ids),
        n_context_tokens=n_text_tokens,
        n_generated_tokens=n_generated_tokens,
    )

    # Text-only requests go straight to a text node.
    if req_type == "TEXT":
        text_load = _text_load(n_text_tokens, 0)
        text_load_window.add(text_load)
        node_ip_to_route = text_node_scheduler.acquire(text_load)
        node_api_url = "http://" + node_ip_to_route + "/generate"
        if INSTANCE_PRIORITY:
            data["priority"] = text_load
        logger.info(
            "Routing %s request req_id=%s to text node %s (load=%d)",
            req_type,
            req_id,
            node_api_url,
            text_load,
        )
        logger.debug("Request payload: %s", data)
        try:
            response = _post(node_api_url, data)
        finally:
            text_node_scheduler.release(node_ip_to_route, text_load)

        if response.status_code != 200:
            logger.error(
                "Text node returned error: req_id=%s status=%s url=%s",
                req_id,
                response.status_code,
                node_api_url,
            )

        try:
            response_json = response.json()
        except ValueError:
            logger.error(
                "Non-JSON response from text node: req_id=%s url=%s status=%s body=%.200s",
                req_id,
                node_api_url,
                response.status_code,
                response.text,
            )
            response_json = {}
        vllm_req_id = response_json.get("vllm_req_id", "")
        content = response_json.get("content", "")
        vllm_processing_time = response_json.get("request_latency_ms", 0.0)

        populate_metrics_df(
            req_id=req_id,
            vllm_req_id=vllm_req_id,
            timestamp=timestamp,
            num_images=0,
            n_context_tokens=n_text_tokens,
            n_generated_tokens=n_generated_tokens,
            num_image_tokens=0,
            cluster_manager_waiting_time=cluster_manager_waiting_time,
            total_image_processing_time=0.0,
            total_llm_backend_processing_time=vllm_processing_time,
            http_code=response.status_code,
            e2e_time=(time.perf_counter() - e2e_start_time) * 1000
            + cluster_manager_waiting_time,
        )

    # Image+text requests encode on image nodes first, then route to a text node.
    elif req_type == "IMAGE":
        arrival_image_tokens = sum(_image_weight(i) for i in image_ids)
        image_load_window.add(arrival_image_tokens)
        text_load_window.add(_text_load(n_text_tokens, arrival_image_tokens))

        track_id = uuid.uuid4().hex
        image_thread_list = []
        image_processing_start_time = time.perf_counter()
        for image_id in image_ids:
            payload = {
                "req_id": data["req_id"],
                "track_id": track_id,
                "image_id": image_id,
            }

            thread = threading.Thread(target=forward_image_request, args=(payload,))
            image_thread_list.append(thread)
            with DATA_LOCK:
                image_status_tracker.setdefault(track_id, []).append(
                    {
                        "image_id": image_id,
                        "num_image_tokens": 0,
                        "image_node_ip": None,
                        "tensor_shape": [],
                    }
                )
            thread.start()

        for thread in image_thread_list:
            thread.join()
        image_processing_end_time = time.perf_counter()
        total_image_processing_time = (
            image_processing_end_time - image_processing_start_time
        ) * 1000

        statuses = image_status_tracker[track_id]
        failed = [
            s for s in statuses if s["image_node_ip"] is None or not s["tensor_shape"]
        ]
        if failed:
            logger.error(
                "Image encoding failed req_id=%s (%d/%d images); not routing to text node",
                req_id,
                len(failed),
                len(statuses),
            )
            populate_metrics_df(
                req_id=req_id,
                vllm_req_id="",
                timestamp=timestamp,
                num_images=len(statuses),
                n_context_tokens=n_text_tokens,
                n_generated_tokens=n_generated_tokens,
                cluster_manager_waiting_time=cluster_manager_waiting_time,
                total_image_processing_time=total_image_processing_time,
                http_code=500,
                e2e_time=(time.perf_counter() - e2e_start_time) * 1000
                + cluster_manager_waiting_time,
            )
            with DATA_LOCK:
                del image_status_tracker[track_id]
            return jsonify({"req_id": req_id, "error": "image encoding failed"}), 500

        image_ids = []
        image_id_shapes = []
        num_image_tokens = 0
        src_ranks = []
        image_node_source_urls = []
        for image_status in statuses:
            image_ids.append(image_status["image_id"])
            image_id_shapes.append(image_status["tensor_shape"])
            num_image_tokens += image_status["num_image_tokens"]
            src_ranks.append(image_node_ranks[image_status["image_node_ip"]])
            # Full host:port — the image node hosts the send endpoint on its own port.
            image_node_source_urls.append(image_status["image_node_ip"])

        text_load = _text_load(n_text_tokens, num_image_tokens)
        node_ip_to_route = text_node_scheduler.acquire(text_load)
        api_url = "http://" + node_ip_to_route + "/generate"
        # Send every image to the recv rank co-located with the chosen text node.
        dest_ranks = [text_recv_rank[node_ip_to_route]] * len(image_ids)
        logger.info(
            "Routing IMAGE request req_id=%s to text node %s (images=%d load=%d)",
            req_id,
            api_url,
            len(statuses),
            text_load,
        )

        payload = {
            "req_id": req_id,
            "prompt": data["prompt"],
            "num_text_tokens": data["num_text_tokens"],
            "images": image_ids,
            "image_id_shapes": image_id_shapes,
            "source_node_ips": image_node_source_urls,
            "src_ranks": src_ranks,
            "dest_ranks": dest_ranks,
            "n": data["n"],
            "best_of": data["best_of"],
            "use_beam_search": data["use_beam_search"],
            "temperature": data["temperature"],
            "top_p": data["top_p"],
            "max_tokens": data["max_tokens"],
            "ignore_eos": data["ignore_eos"],
            "stream": data["stream"],
        }
        if INSTANCE_PRIORITY:
            payload["priority"] = text_load

        logger.debug(
            "Sending payload to text node: req_id=%s src_ranks=%s dest_ranks=%s payload=%s",
            req_id,
            src_ranks,
            dest_ranks,
            payload,
        )

        # pool manager sitting on the same host

        num_images = len(statuses)

        _admit_image_tokens(node_ip_to_route, num_image_tokens)
        # Charge the text node until it responds, then refund (least-loaded routing).
        try:
            response = forward_text_request(api_url, payload)
        finally:
            _release_image_tokens(node_ip_to_route, num_image_tokens)
            text_node_scheduler.release(node_ip_to_route, text_load)

        # Guard against a non-JSON body; keep the arrival req_id as the metric key.
        try:
            response_json = response.json()
        except ValueError:
            logger.error(
                "Non-JSON response from text node: req_id=%s url=%s status=%s body=%.200s",
                req_id,
                api_url,
                response.status_code,
                response.text,
            )
            response_json = {}
        vllm_req_id = response_json.get("vllm_req_id", 0)
        content = response_json.get("content", "")
        vllm_processing_time = response_json.get("request_latency_ms", 0.0)

        if response.status_code != 200:
            logger.error(
                "Text node returned error: req_id=%s status=%s url=%s num_images=%s",
                req_id,
                response.status_code,
                api_url,
                num_images,
            )
            populate_metrics_df(
                req_id=req_id,
                vllm_req_id=vllm_req_id,
                total_image_processing_time=vllm_processing_time,
                timestamp=timestamp,
                num_images=num_images,
                n_context_tokens=n_text_tokens,
                n_generated_tokens=n_generated_tokens,
                num_image_tokens=num_image_tokens,
                e2e_time=(time.perf_counter() - e2e_start_time) * 1000
                + cluster_manager_waiting_time,
                http_code=response.status_code,
            )

        else:
            populate_metrics_df(
                req_id=req_id,
                vllm_req_id=vllm_req_id,
                timestamp=timestamp,
                num_images=num_images,
                n_context_tokens=n_text_tokens,
                n_generated_tokens=n_generated_tokens,
                num_image_tokens=num_image_tokens,
                cluster_manager_waiting_time=cluster_manager_waiting_time,
                total_image_processing_time=total_image_processing_time,
                total_llm_backend_processing_time=vllm_processing_time,
                e2e_time=(time.perf_counter() - e2e_start_time) * 1000
                + cluster_manager_waiting_time,
                http_code=response.status_code,
            )

        with DATA_LOCK:
            del image_status_tracker[track_id]

    response_ = requests.Response()
    response_.status_code = response.status_code
    response_._content = bytes(
        json.dumps({"vllm_req_id": vllm_req_id, "content": content}).encode("utf8")
    )
    return jsonify(response_.json()), response_.status_code


def forward_image_request(payload: dict[str, Any]) -> None:
    image_load = _image_weight(payload.get("image_id", ""))
    image_node_ip = image_node_scheduler.acquire(image_load)
    logger.debug(
        "IMAGE routed: req_id=%s image_id=%s -> %s (load=%d tokens)",
        payload.get("req_id"),
        payload.get("image_id"),
        image_node_ip,
        image_load,
    )
    try:
        _forward_image_request(payload, image_node_ip)
    finally:
        image_node_scheduler.release(image_node_ip, image_load)


def _forward_image_request(payload: dict[str, Any], image_node_ip: str) -> None:
    api_url = "http://" + image_node_ip + "/generate"

    logger.debug(
        "Routing image to image node: req_id=%s image_id=%s url=%s",
        payload.get("req_id"),
        payload.get("image_id"),
        api_url,
    )
    start_request_time = time.perf_counter()
    response = _post(api_url, payload)
    end_request_time = time.perf_counter()

    if response.status_code != 200:
        logger.error(
            "Image node returned error: req_id=%s status=%s url=%s",
            payload.get("req_id"),
            response.status_code,
            api_url,
        )

    # get the json of response (guard against a non-JSON error body)
    try:
        response_json = response.json()
    except ValueError:
        logger.error(
            "Non-JSON response from image node: req_id=%s url=%s status=%s body=%.200s",
            payload.get("req_id"),
            api_url,
            response.status_code,
            response.text,
        )
        response_json = {}
    image_id = payload.get("image_id", "")
    num_image_tokens = response_json.get("num_image_tokens", 0)
    req_id = payload.get("req_id", "")
    track_id = payload.get("track_id", req_id)
    preprocessing_time = response_json.get("preprocessing_time", 0.0)
    encoding_time = response_json.get("image_encoding_time", 0.0)
    e2e_time = (end_request_time - start_request_time) * 1000
    tensor_shape = response_json.get("tensor_shape", [])
    waiting_time_ms = e2e_time - (preprocessing_time + encoding_time)

    populate_image_metrics_df(
        req_id=req_id,
        image_id=image_id,
        waiting_time=waiting_time_ms,
        num_image_tokens=num_image_tokens,
        preprocessing_time=preprocessing_time,
        encoding_time=encoding_time,
        e2e_time=e2e_time,
    )

    with DATA_LOCK:
        for image_status in image_status_tracker[track_id]:
            if (
                image_status["image_id"] == image_id
                and image_status["image_node_ip"] is None
            ):
                logger.debug(
                    "Image encoded: req_id=%s image_id=%s num_image_tokens=%s",
                    req_id,
                    image_id,
                    num_image_tokens,
                )
                image_status["num_image_tokens"] = num_image_tokens
                image_status["image_node_ip"] = image_node_ip
                image_status["tensor_shape"] = tensor_shape
                break


def forward_text_request(api_url: str, payload: dict[str, Any]) -> requests.Response:
    logger.debug(
        "Forwarding image-text request to text node: req_id=%s payload=%s",
        payload.get("req_id"),
        payload,
    )
    response = _post(api_url, payload)

    return response


def signal_handler(sig: int, frame: Any) -> None:
    logger.info(
        "Received signal %s; writing metrics to %s and %s, then shutting down",
        signal.Signals(sig).name,
        METRICS_FILE,
        IMAGE_METRICS_FILE,
    )
    global metrics_df
    global image_metrics_df

    os.makedirs(METRICS_FILE[: METRICS_FILE.rfind("/")], exist_ok=True)

    metrics_df = metrics_df.round(2)
    csv_string = metrics_df.to_csv()
    with open(METRICS_FILE, "w+") as f:
        f.write(csv_string)

    image_metrics_df = image_metrics_df.round(2)
    csv_string = image_metrics_df.to_csv()
    with open(IMAGE_METRICS_FILE, "w+") as f:
        f.write(csv_string)

    logger.info(
        "Metrics written: %s (%d rows), %s (%d rows); shutdown complete",
        METRICS_FILE,
        len(metrics_df),
        IMAGE_METRICS_FILE,
        len(image_metrics_df),
    )
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if AUTOSCALE:
        enc_mc, bak_mc = _load_profile_mc()
        autoscale_metrics = (
            METRICS_FILE[: METRICS_FILE.rfind("/") + 1] + "autoscale_metrics.csv"
        )
        autoscaler = autoscaling.Autoscaler(
            pools=[
                {
                    "name": "image",
                    "scheduler": image_node_scheduler,
                    "window": image_load_window,
                    "m_c": enc_mc,
                    "n_max": len(modserve_image_nodes),
                },
                {
                    "name": "text",
                    "scheduler": text_node_scheduler,
                    "window": text_load_window,
                    "m_c": bak_mc,
                    "n_max": len(modserve_text_nodes),
                },
            ],
            interval_s=AUTOSCALE_INTERVAL_S,
            logger=logger,
            metrics_path=autoscale_metrics,
        )
        autoscaler.start()

    logger.info(
        "Cluster manager ready, serving on http://%s:%s/generate (text_nodes=%d, image_nodes=%d, autoscale=%s)",
        CLUSTER_MANAGER_IP,
        CLUSTER_MANAGER_PORT,
        len(modserve_text_nodes),
        len(modserve_image_nodes),
        AUTOSCALE,
    )
    app.run(
        debug=False, host=CLUSTER_MANAGER_IP, port=CLUSTER_MANAGER_PORT, threaded=True
    )
