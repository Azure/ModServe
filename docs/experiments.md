# Experiments

End-to-end pipeline to reproduce the eval figures. Assumes setup is done (see README).

## 1. Fetch the trace
Downloads the official Azure LMM trace ([CC-BY](https://github.com/Azure/AzurePublicDataset))
and samples a window to `traces/<TRACE>.csv`.

```bash
pixi run python utils/fetch_trace.py                  # first hour
pixi run python utils/fetch_trace.py --start-hour 5 --hours 1
```

## 2. Profile capacity (once per model)
Measures encoder + backend `M_C` → `profiles/<model>/profile.json` (used by autoscaling).

```bash
pixi run python -m lmm_serving.profiler.run
```

## 3. Generate per-QPM traces
Slices a window from a source trace and scales the rate (higher factor = lighter load).
Writes `traces/experiment_traces/<trace>_scale<factor>.csv` (gitignored).

```bash
pixi run python utils/create_experiment_qps_traces.py <source.csv> <factor> [window_min]
```

## 4. Run one experiment
`<setup>` = `monolith` | `modserve`; `<scales>` = comma list of trace factors.

```bash
pixi run python -m lmm_serving.launcher run modserve 2.0 --nodes-config internvl_3box.yaml
pixi run python -m lmm_serving.launcher run modserve 8.0,4.0,2.0 --dry-run   # plan only
```
Metrics land in `metrics/<setup>/<run_id>/` (auto-merged across nodes).

## 5. Run the ablation campaign (Fig 14)
Iterates configs × models × loads, one launcher call per cell. Resumable: a cell whose
`metrics/<setup>/<run_id>/` already has rows is skipped (use `--no-resume` to force).

```bash
pixi run python -m lmm_serving.experiments.run_ablation fig14 \
  --configs vllm,decoup,sched,modserve --models internvl --scales 7.77,3.89,2.59,1.94 \
  --modserve-nodes internvl_3box.yaml --monolith-nodes internvl_3box.yaml \
  --timeout 1800 --cleanup-hosts "10.0.0.4,10.0.0.5,10.0.0.6"
```
One model per run (each model uses different topologies); run again with `--models llama`.

| Config | Setup | Knobs |
|---|---|---|
| `vllm` | monolith | — |
| `decoup` | modserve | round_robin + fcfs |
| `sched` | modserve | round_robin + priority |
| `modserve` | modserve | least_loaded + priority |

## 6. Autoscaling run (Fig 16)
Same launcher with `AUTOSCALE=1`, replaying a bursty window for monolith and modserve.

```bash
AUTOSCALE=1 pixi run python -m lmm_serving.launcher run modserve <scale> --nodes-config llama_3box.yaml
```

## 7. Plot
`notebooks/evals.ipynb` scores the runs (`lmm_serving.experiments.scoring`) and draws
Fig 14 (TTFT vs load) and Fig 16 (GPU allocation).
