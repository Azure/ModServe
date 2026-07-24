# ModServe

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![vLLM Backend](https://img.shields.io/badge/backend-vLLM-orange.svg)](https://github.com/vllm-project/vllm)

Serving infrastructure and production trace-replay benchmarks for large multimodal models
(LMMs), from the paper [*ModServe: Modality- and Stage-Aware Resource
Disaggregation for Scalable Multimodal Model Serving*](https://dl.acm.org/doi/10.1145/3698038.3698540)
(SoCC 2025).

It compares two serving architectures under realistic request traces:

| Architecture | Description |
|---|---|
| **Monolith** | The full vision-language model served by a single vLLM instance. |
| **ModServe** | *Disaggregated* serving: vision encoders run on separate "image nodes" and stream encoded image tensors over NCCL to the "text node" running the language model. |

A **cluster manager** fronts both architectures with configurable routing
(least-loaded / round-robin); a **trace client** replays timestamped request
traces from the [Azure LMM Inference Trace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md).

## Architecture Overview

```
                          ┌──────────────────────────────────────────┐
                          │ Cluster Manager (Flask)                  │
 ┌───────────────┐  HTTP  │ - least_loaded / round_robin routing     │
 │ Trace Client  │───────►│ - autoscaling (optional)                 │
 └───────────────┘        └──┬──────────────────────────────┬────────┘
                        (Monolith)                      (ModServe)
                             │                              │
                             │                              │
                             │                              │
                  ┌──────────▼─────────┐          ┌─────────▼────────┐   NCCL   ┌──────────────┐
                  │ Text Node (vLLM)   │          │ Image Node(s)    │─────────►│ Text Node    │
                  │ full VLM inline    │          │ vision encoder   │          │ (LLM only)   │
                  └────────────────────┘          └──────────────────┘          └──────────────┘
```

## Supported Models

| Model | Architecture | Config key |
|---|---|---|
| [Llama 3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | MLLAMA (cross-attention) | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| [InternVL2.5-26B](https://huggingface.co/OpenGVLab/InternVL2_5-26B) | INTERNVL (decoder-only) | `OpenGVLab/InternVL2_5-26B` |

## Hardware Requirements

- **GPU**: NVIDIA A100 (40 GB or 80 GB) or equivalent; minimum 4 GPUs for the
  default `local.yaml` topologies.
- **CUDA**: 12.4+ (driver ≥ 550).
- **RAM**: ≥ 64 GB recommended (model weights + KV cache).
- **Disk**: ~50 GB for model weights + ~6.5 GB for ShareGPT-4o images.
- **Multi-box** (optional): InfiniBand for cross-node NCCL tensor transfer.

## Installation

### Option A: Pixi (recommended)

[Pixi](https://pixi.sh) manages the full toolchain — CUDA toolkit, Python 3.10,
all dependencies, the patched vLLM backend, and flash-attn — from `pixi.toml`.

```bash
# 1. Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Build everything: CUDA toolkit + deps + editable package + patched vLLM + flash-attn
pixi run setup

# Alternatively, use the full provisioning script (also installs Pixi if needed):
bash utils/setup_repo.sh
```

### Option B: Conda

Use a standalone Conda environment if you prefer managing dependencies manually.

```bash
# 1. Create and activate the environment
conda create -n modserve python=3.10 -y
conda activate modserve

# 2. Install PyTorch with CUDA 12.4 support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install project dependencies
pip install -r requirements.txt
pip install huggingface-hub==0.27.1 openai

# 4. Install the package in editable mode
pip install -e .

# 5. Build the patched vLLM backend (requires CUDA toolkit / nvcc)
bash llm-backend/build_vllm.sh

# 6. Install flash-attn (optional but recommended; requires nvcc)
pip install --no-build-isolation flash-attn==2.7.4.post1
```

> **Note:** Steps 5–6 require a working `nvcc` (CUDA 12.4 toolkit). If you
> installed CUDA via your package manager, ensure `nvcc --version` works before
> running them.

## Quickstart

After installing with either option above:

```bash
# 1. Configure paths and model
#    Edit IMAGE_DIR, HF_HOME, MODEL, and cluster settings as needed.
#    Defaults assume ${HOME}/hf_cache for model weights and images.
$EDITOR src/lmm_serving/config/config.yaml

# 2. Authenticate with Hugging Face (to download model weights)
huggingface-cli login          # or: export HF_TOKEN=hf_...

# 3. Stage trace images (downloads ~6.5 GB of ShareGPT-4o images)
python utils/stage_sharegpt4o_images.py

# 4. Fetch and sample the Azure LMM trace (downloads + samples a 1-hour window)
python utils/fetch_trace.py

# 5. Create experiment traces at a given scale factor
#    Higher factor = lighter load (e.g., 2.0 halves the QPS).
python utils/create_experiment_qps_traces.py \
    AzureLMMInferenceTrace_img_heavy_1_hour.csv 2.0

# 6. Run an experiment
python -m lmm_serving.launcher run modserve 2.0   # or: monolith
```

> **Pixi users:** prefix commands with `pixi run`, e.g., `pixi run python -m lmm_serving.launcher run modserve 2.0`.

### Multi-box Setup

For disaggregated serving across multiple hosts (image nodes on separate boxes):

```bash
# Bootstrap remote boxes from the primary host
bash utils/remote_setup.sh <box2_ip> [box3_ip ...]
```

Then edit the topology file under `configs/modserve/` to describe your cluster
layout (see [docs/topology.md](docs/topology.md)).

### Validating a Topology

Check your topology file for errors before launching:

```bash
python -m lmm_serving.launcher validate modserve   # or: monolith
```

## Project Structure

```
├── src/lmm_serving/          # Main package
│   ├── config/               # config.yaml + loader
│   ├── common/               # Shared: scheduling, autoscaling, text_node, vision_tokens
│   ├── launcher/             # CLI entry point, topology, process orchestration
│   ├── modserve/             # Disaggregated serving components
│   │   └── components/       # cluster_manager, image/text node managers, tensor transfer
│   ├── monolith/             # Monolithic serving components
│   │   └── components/       # cluster_manager, text node manager
│   ├── profiler/             # Latency profiling + RF model fitting
│   └── experiments/          # Ablation runner + scoring
├── client/                   # Trace replay client
├── configs/                  # Topology YAML files (modserve/ and monolith/)
├── llm-backend/              # Patched vLLM build script + patch file
├── utils/                    # Setup, trace fetch/staging, cleanup scripts
├── tests/                    # Unit tests (topology, scheduling, autoscaling, profiler)
├── notebooks/                # Evaluation notebooks
├── docs/                     # Detailed documentation
├── pixi.toml                 # Pixi workspace definition
├── pyproject.toml            # Python package metadata
└── requirements.txt          # Pip dependencies
```

## Running Tests

The test suite validates topology wiring, scheduling logic, autoscaling, and the
profiler model — no GPUs required:

```bash
python tests/test_launcher.py      # 39 tests — topology, NCCL wiring, SSH transport
python tests/test_scheduling.py    #  9 tests — routing, load balancing
python tests/test_autoscaler.py    # 20 tests — active set, sliding window, autoscaler
python tests/test_profiler.py      #  2 tests — RF model capacity
```

## Documentation

| Document | Description |
|---|---|
| [docs/configuration.md](docs/configuration.md) | Every `config.yaml` parameter explained |
| [docs/topology.md](docs/topology.md) | Writing `configs/<setup>/nodes_*.yaml` cluster topologies |
| [docs/experiments.md](docs/experiments.md) | End-to-end pipeline: profiling → traces → runs → ablation → plots |

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'vllm'` | Run `bash llm-backend/build_vllm.sh` (Conda) or `pixi run build-vllm` (Pixi). |
| `ModuleNotFoundError: No module named 'flash_attn'` | Run `pip install --no-build-isolation flash-attn==2.7.4.post1`. Requires `nvcc`. |
| `huggingface_hub` version conflict | Pin to `0.27.1`: `pip install huggingface-hub==0.27.1`. |
| NCCL timeout on multi-box | Check `NCCL_SOCKET_IFNAME` matches your network interface; set `NCCL_IB_DISABLE=true` if no InfiniBand. |
| GPU out of memory | Lower `TEXT_GPU_UTIL` or `TEXT_MAX_MODEL_LEN` in `config.yaml`. |
| `nvidia-smi` not found | Install the NVIDIA driver (≥ 550 for CUDA 12.4). |
| Topology validation errors | Run `python -m lmm_serving.launcher validate <setup>` and fix the reported issues. |

## Citation

If you use ModServe in your research, please cite:

```bibtex
@inproceedings{qiu2025modserve,
  title     = {{ModServe}: Modality-and Stage-Aware Resource Disaggregation
               for Scalable Multimodal Model Serving},
  author    = {Qiu, Haoran and Biswas, Anish and Zhao, Zihan and Mohan, Jayashree
               and Khare, Alind and Choukse, Esha and Goiri, {\'I}{\~n}igo
               and Zhang, Zeyu and Shen, Haiying and Bansal, Chetan and others},
  booktitle = {Proceedings of the 2025 ACM Symposium on Cloud Computing (SoCC 2025)},
  pages     = {817--830},
  year      = {2025}
}
```