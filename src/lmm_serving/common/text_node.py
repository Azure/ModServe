from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
from typing import Callable

import requests
from flask import Flask, jsonify, request
from openai import OpenAI

VALID_TP = ("TP1", "TP2", "TP4", "TP8")


def vllm_port(public_port: str) -> str:
    if public_port == "9090":
        return "8000"
    return str(8001 + max(0, int(public_port) - 8080))


def _start_vllm(command, out_path, err_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return subprocess.Popen(
        command, stdout=open(out_path, "w"), stderr=open(err_path, "w")
    )


def _wait_for_vllm(process, base_url, err_path, logger, poll_interval=5):
    health_url = base_url.rsplit("/v1", 1)[0] + "/health"
    logger.info("Waiting for vLLM to become ready at %s", health_url)
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {process.returncode} before becoming ready — see {err_path}"
            )
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                logger.info("vLLM backend ready at %s", health_url)
                return
        except requests.RequestException:
            pass
        time.sleep(poll_interval)


def serve(
    *,
    logger,
    hostname,
    port,
    model_id,
    command,
    out_log,
    err_log,
    image_content: Callable[[dict], list],
    priority_enabled: bool = False,
):
    base_url = f"http://localhost:{vllm_port(port)}/v1"
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    app = Flask(__name__)

    def _generate(inputs, output_len, ignore_eos, kind, priority):
        start = time.perf_counter()
        extra_body = {"ignore_eos": ignore_eos}
        if priority_enabled:
            extra_body["priority"] = priority
        try:
            completion = client.chat.completions.create(
                messages=inputs,
                model=model_id,
                max_tokens=output_len,
                extra_body=extra_body,
            )
            req_id, content, status = (
                completion.id,
                completion.choices[0].message.content,
                200,
            )
        except Exception as e:
            # Degrade to a clean tuple so the handler never returns a non-JSON 500.
            logger.exception(
                "%s request to vLLM failed (model=%s, max_tokens=%s)",
                kind,
                model_id,
                output_len,
            )
            req_id, content, status = (
                getattr(e, "request_id", "failed_req"),
                "",
                getattr(e, "status_code", 500) or 500,
            )
        return req_id, status, content, (time.perf_counter() - start) * 1000

    @app.route("/generate", methods=["POST"])
    def process_request():
        data = request.get_json()
        prompt = data.get("prompt", "Describe this image in one sentence.")
        output_len = data.get("max_tokens", 128)
        ignore_eos = data.get("ignore_eos", False)
        priority = int(data.get("priority", 0))

        content = [{"type": "text", "text": prompt}]
        image_blocks = image_content(data)
        content.extend(image_blocks)
        kind = "Image" if image_blocks else "Text"
        if image_blocks:
            logger.debug(
                "Image request: images=%d output_len=%d", len(image_blocks), output_len
            )

        vllm_req_id, status_code, text, latency_ms = _generate(
            [{"role": "user", "content": content}],
            output_len,
            ignore_eos,
            kind,
            priority,
        )

        response = requests.Response()
        response.status_code = status_code
        response._content = json.dumps(
            {
                "req_id": data.get("req_id", 0),
                "vllm_req_id": vllm_req_id,
                "content": text,
                "request_latency_ms": latency_ms,
            }
        ).encode("utf-8")
        return jsonify(response.json()), response.status_code

    server_process = _start_vllm(command, out_log, err_log)
    logger.info("vLLM server started with PID: %s", server_process.pid)

    def _cleanup():
        logger.info(
            "Shutting down: terminating vLLM server (PID %s)", server_process.pid
        )
        server_process.terminate()

    atexit.register(_cleanup)

    _wait_for_vllm(server_process, base_url, err_log, logger)
    logger.info(
        "Text node ready, serving on http://%s:%s/generate (vLLM at %s)",
        hostname,
        port,
        base_url,
    )
    app.run(debug=False, port=int(port), host=hostname)
