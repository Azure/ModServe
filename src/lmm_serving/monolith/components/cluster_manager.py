from flask import Flask, request, jsonify
import requests
import threading
import time
import os
import sys
import signal
import json
import yaml

from lmm_serving.common.logging_config import get_logger
from lmm_serving.common import scheduling
from lmm_serving.common import autoscaling
from lmm_serving import config as cfg

logger = get_logger("monolith.cluster_manager")

# Configuration
CLUSTER_MANAGER_IP = sys.argv[1]
CLUSTER_MANAGER_PORT = sys.argv[2]
METRICS_FILE = sys.argv[3]
# Optional 4th arg: cluster config filename inside configs/<setup>/.
NODES_CONFIG = (
    sys.argv[4]
    if len(sys.argv) > 4
    else os.environ.get("NODES_CONFIG", "local.yaml")
)

# text node : [image_nodes]
monolith_nodes = []
with open(f"{os.environ['CLUSTER_CONFIG_DIR']}/{NODES_CONFIG}") as f:
    monolith_config = yaml.safe_load(f)
    for monolith_node in monolith_config["monolith_nodes"]:
        monolith_nodes.append(monolith_node["node_address"])

metrics_df = scheduling.request_metrics_df()

DATA_LOCK = threading.Lock()
METRICS_FILE_LOCK = threading.Lock()

app = Flask(__name__)


def _post(url, payload):
    return scheduling.guarded_post(logger, url, payload)


def populate_metrics_df(req_id, vllm_req_id, **kwargs):
    scheduling.record_request_metrics(
        metrics_df, METRICS_FILE_LOCK, req_id, vllm_req_id, **kwargs
    )


scheduler = scheduling.RoundRobinScheduler(monolith_nodes)

AUTOSCALE = getattr(cfg, "AUTOSCALE", False)
AUTOSCALE_INTERVAL_S = getattr(cfg, "AUTOSCALE_INTERVAL_S", 30)
AUTOSCALE_WINDOW_S = getattr(cfg, "AUTOSCALE_WINDOW_S", 60)
TOK_PER_IMAGE = 6404 if "LLAMA" in str(cfg.MODEL).upper() else 2188  # mllama 4*1601; InternVL ~8.5 tiles*256
DECODER_ONLY = "INTERNVL" in str(cfg.MODEL).upper()
MONOLITH_COUNTS_IMAGE = True
load_window = autoscaling.LoadWindow(AUTOSCALE_WINDOW_S)
autoscaler = None


@app.route("/generate", methods=["POST"])
def process_request():
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

    img_tok = len(image_ids) * TOK_PER_IMAGE
    load_window.add(
        scheduling.num_pending_tokens(
            n_text_tokens, img_tok, decoder_only=MONOLITH_COUNTS_IMAGE
        )
    )

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

    # get the next node to send this request to:
    with DATA_LOCK:
        node_ip_to_route = scheduler.acquire()

    populate_metrics_df(
        req_id=req_id,
        vllm_req_id="",
        timestamp=timestamp,
        num_images=len(image_ids),
        n_context_tokens=n_text_tokens,
        n_generated_tokens=n_generated_tokens,
    )

    node_api_url = "http://" + node_ip_to_route + "/generate"
    logger.info(
        "Routing %s request req_id=%s to text node %s", req_type, req_id, node_api_url
    )
    logger.debug("Request payload: %s", data)
    response = _post(node_api_url, data)

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
        num_images=len(image_ids),
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

    response_ = requests.Response()
    response_.status_code = response.status_code
    response_._content = bytes(
        json.dumps({"vllm_req_id": vllm_req_id, "content": content}).encode("utf8")
    )
    return jsonify(response_.json()), response_.status_code


def signal_handler(sig, frame):
    logger.info(
        "Received signal %s; writing metrics to %s, then shutting down",
        signal.Signals(sig).name,
        METRICS_FILE,
    )
    global metrics_df

    metrics_df = metrics_df.round(2)
    csv_string = metrics_df.to_csv()
    with open(METRICS_FILE, "w+") as f:
        f.write(csv_string)

    logger.info(
        "Wrote %d request metrics to %s; shutdown complete",
        len(metrics_df),
        METRICS_FILE,
    )
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if AUTOSCALE:
        mc = float(getattr(cfg, "AUTOSCALE_MONOLITH_MC", 0) or 0)
        autoscale_metrics = (
            METRICS_FILE[: METRICS_FILE.rfind("/") + 1] + "autoscale_metrics.csv"
        )
        autoscaler = autoscaling.Autoscaler(
            pools=[
                {
                    "name": "monolith",
                    "scheduler": scheduler,
                    "window": load_window,
                    "m_c": mc,
                    "n_max": len(monolith_nodes),
                }
            ],
            interval_s=AUTOSCALE_INTERVAL_S,
            logger=logger,
            metrics_path=autoscale_metrics,
        )
        autoscaler.start()
        logger.info(
            "Monolith autoscaler started: M_C=%.0f tok/s N=%d (each replica = TP4 = 4 GPUs)",
            mc,
            len(monolith_nodes),
        )

    logger.info(
        "Monolith cluster_manager starting: listen=%s:%s config=%s monolith_nodes=%d metrics=%s autoscale=%s",
        CLUSTER_MANAGER_IP,
        CLUSTER_MANAGER_PORT,
        NODES_CONFIG,
        len(monolith_nodes),
        METRICS_FILE,
        AUTOSCALE,
    )
    logger.info(
        "Cluster manager ready, serving on http://%s:%s/generate",
        CLUSTER_MANAGER_IP,
        CLUSTER_MANAGER_PORT,
    )
    app.run(debug=False, host=CLUSTER_MANAGER_IP, port=CLUSTER_MANAGER_PORT)
