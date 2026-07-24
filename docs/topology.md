# Cluster topology

Topology files live in `configs/<setup>/*.yaml` and are selected with `NODES_CONFIG`
or `--nodes-config`. Two schemas: `monolith` and `modserve`. Validate before running:

```bash
pixi run python -m lmm_serving.launcher validate modserve --nodes-config internvl_3box.yaml
```

## Monolith

Each entry is one full-VLM vLLM replica; the cluster manager round-robins across them.

```yaml
monolith_nodes:
  - node_address: 10.0.0.4:8080   # host:port
    node_tp: TP4                  # TP1|TP2|TP4|TP8
    local_ranks: [0, 1, 2, 3]     # GPU ids (len must == TP size)
```

## Modserve

Text nodes run the LLM; image nodes encode and stream tensors over NCCL.

```yaml
modserve_text_nodes:
  - text_node_address: 10.0.0.4:8080
    text_node_tp: TP4
    local_ranks: [0, 1, 2, 3]     # GPU ids (len must == TP size)

modserve_image_nodes:
  - {image_node_address: 10.0.0.4:8086, image_node_rank: 0, image_node_tp: TP1, image_node_gpu_id: "4", numa_physcpu: "8"}
```

| Field | Meaning |
|---|---|
| `image_node_rank` | global NCCL rank; **must be contiguous 0..N-1**. Rank 0's host is the NCCL master |
| `image_node_gpu_id` | GPU index on that host |
| `numa_physcpu` | NUMA-0 core (0-23) for `numactl` preprocessing pin |

## Rules (enforced by `validate`)
- text/monolith TP must be one of `TP1/TP2/TP4/TP8`; image encoders must be `TP1`.
- text/monolith `local_ranks` length must equal TP size.
- modserve needs ≥1 text and ≥1 image node; `image_node_rank` contiguous `0..N-1`.
- Every address is `host:port`; no GPU id or `host:port` reused on a host.
- On a local host, total GPUs needed must not exceed `nvidia-smi` count.

## Multi-box
Set `NCCL_MASTER_ADDR` to the eth0 IP of rank-0's host. If boxes can't RDMA across
subnets, set `NCCL_IB_DISABLE: true` (TCP fallback). See `docs/configuration.md`.
