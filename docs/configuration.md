# Configuration

All settings live in `src/lmm_serving/config/config.yaml`. Each key is exported as an
env var (shell: `yq` via `load.sh`; Python: PyYAML). **A shell env var overrides the
file**, so you can override per-run: `MODEL=... TEXT_GPU_UTIL=0.8 pixi run ...`.
`${HOME}`/`${REPO_ROOT}` expand on both sides.

## Data
| Key | Default | Notes |
|---|---|---|
| `IMAGE_DIR` | `${HOME}/hf_cache/sharegpt_images` | trace image files (`file://` URLs) |
| `HF_HOME` | `${HOME}/hf_cache` | model weight cache |
| `HF_HUB_OFFLINE` | `1` | `0` only when regenerating traces (fetches gated dataset) |
| `SOURCE_TRACE` | `${HOME}/hf_cache/...multimodal.csv` | full Azure trace (`fetch_trace.py` downloads here, then samples) |
| `TRACE` | `AzureLMMInferenceTrace_img_heavy_1_hour` | base name; client replays `${TRACE}_scale<QPM>.csv` |

## Model
| Key | Default | Notes |
|---|---|---|
| `MODEL` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | or `OpenGVLab/InternVL2_5-26B` |

## Cluster manager
| Key | Default | Notes |
|---|---|---|
| `CLUSTER_MANAGER_IP` / `CLUSTER_MANAGER_PORT` | `127.0.0.1` / `7070` | front-end the client hits |
| `NODES_CONFIG` | `local.yaml` | topology under `configs/<setup>/` |

## Tensor transfer + NCCL (modserve, multi-box)
| Key | Default | Notes |
|---|---|---|
| `TENSOR_TRANSFER_PORT` | `5050` | base recv port |
| `NCCL_MASTER_ADDR` / `NCCL_MASTER_PORT` | `127.0.0.1` / `29500` | set ADDR to box-1's eth0 IP for multi-box |
| `NCCL_SOCKET_IFNAME` | `eth0` | bootstrap NIC |
| `NCCL_IB_HCA` | `mlx5_ib0..7` | IB devices |
| `NCCL_IB_DISABLE` | `true` | forces TCP (set when boxes can't RDMA cross-subnet) |

## Modserve routing / scheduling
| Key | Default | Choices / Notes |
|---|---|---|
| `SCHEDULER` | `least_loaded` | image-node routing: `round_robin` \| `least_loaded` |
| `TEXT_SCHEDULER` | `least_loaded` | text-node routing |
| `INSTANCE_SCHEDULING` | `priority` | in-node: `fcfs` \| `priority` (SLO-driven) |
| `TEXT_GPU_UTIL` | `0.90` | text KV-cache fraction |
| `TEXT_INFLIGHT_IMAGE_TOKEN_BUDGET` | `120000` | backpressure cap on admitted image tokens |
| `TEXT_MAX_MODEL_LEN` | `16384` | cap so KV fits at low TP (empty = 32768) |
| `TEXT_MAX_NUM_SEQS` | `` | empty = per-model default |
| `TEXT_PREEMPTION_MODE` | `` | empty = vLLM default; `swap` avoids recompute re-issuing transfers |

## Monolith
| Key | Default | Notes |
|---|---|---|
| `MONOLITH_MAX_NUM_SEQS` | `16` | max concurrent sequences |
| `MONOLITH_GPU_UTIL` | `` | empty = vLLM default |

## Autoscaling (modserve)
`M_C` (max tok/s under SLO) comes from `profiles/<model>/profile.json`. Off by default.
| Key | Default | Notes |
|---|---|---|
| `AUTOSCALE` | `false` | enable token-aware autoscaling |
| `AUTOSCALE_INTERVAL_S` / `AUTOSCALE_WINDOW_S` | `30` / `60` | decide every 30s on 60s load |
| `AUTOSCALE_IMAGE_MC` / `AUTOSCALE_TEXT_MC` | `0.0` | per-instance M_C override (0 = profiled). Set TEXT_MC to the measured knee — profiler overstates prefill-only |
| `AUTOSCALE_MONOLITH_MC` | `0.0` | per-replica M_C (0 = all replicas active) |
