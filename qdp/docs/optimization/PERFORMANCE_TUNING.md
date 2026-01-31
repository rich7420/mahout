# QDP Performance and Tuning

Short guide for tuning QDP encode throughput and latency. For full bottleneck analysis and roadmap, see [QDP_OPTIMIZATION_PLAN_EN.md](QDP_OPTIMIZATION_PLAN_EN.md).

## Goals

- **Throughput ↑**: More vectors per second; fewer sync points, larger effective batches.
- **Latency ↓**: Shorter time from submit to result; overlap H2D/compute, defer sync.

## API choice

| Use case | API / path | Note |
|----------|------------|------|
| **Throughput** | `encode_stream` or `encode_list` with **larger batches** (e.g. 256–512 samples per batch) | One pipeline path; larger batch → more work per call, better GPU utilization. |
| **Latency** | `encode_stream` with default or small batch | Stream yields per batch quickly; observed lower median latency than coalesced in baseline (e.g. ~1.27 ms vs ~1.45 ms per vector). |
| **Batched list** | `encode_list` with a large chunk (e.g. up to 64 MB) | One FFI per chunk; avoids per-call Python overhead; use when you can collect many samples before calling. |

All encode APIs (`encode`, `encode_batch`, `encode_list`, `encode_stream`) use the **same pipeline** (`encode_batch_via_pipeline`); there is no separate "sync" vs "pipeline" path.

## Recommended batch sizes

- **Throughput**: Use **batch_size 256–512** (or larger) when running stream or coalesced benchmarks so each pipeline run has more chunks and longer GPU time per call.
- **Latency**: Default batch_size (e.g. 64) with `encode_stream` is fine for interactive use.

## Environment variables

| Variable | Meaning | Default / note |
|----------|---------|-----------------|
| `QDP_PINNED_POOL_SIZE` | Number of pinned buffers in the pipeline (2–16). | Default 2. Set to **4** for tuning when host memory allows; can improve H2D overlap within one pipeline run. |
| `QDP_CHUNK_SIZE_MB` | Chunk size in MB for the dual-stream pipeline (1–256). | Default 8. Keep ≥ 1 MB for efficient async H2D. |
| `QDP_PCIE_GEN` | PCIe generation (3, 4, 5 or gen3, gen4, gen5) for defaults. | Unset → Unknown; used for chunk default when set. |
| `QDP_BATCH_QUEUE_CAPACITY` | **Rust BatchPool** job queue capacity (2–8). | Default **4**. Bounded queue for backpressure; encode_batch and encode_stream use the pool on Linux. |
| `QDP_MASTER_CPU_ID` | (Optional) Pin the pool master thread to this CPU ID (Linux). | Requires build with feature **thread_affinity** (`cargo build --features thread_affinity`). Set to desired CPU index (e.g. `0`). Fewer OS context switches. |
| `QDP_ENABLE_POOL_METRICS` | Enable pool utilization metrics. | Optional; see [OBSERVABILITY_USAGE.md](../observability/OBSERVABILITY_USAGE.md). |
| `QDP_ENABLE_OVERLAP_TRACKING` | Enable H2D overlap tracking. | Optional; see observability guide. |
| `RUST_LOG` | Rust log level (e.g. `info`, `debug`). | Set before importing Python module for pipeline logs. |

## Observed GPU utilization (baseline)

With default settings (batch_size 64, stream or coalesced):

- **--use-coalesced**: GPU utilization ~**15–35%** (low).
- **--use-stream**: GPU utilization ~**25–40%** (low; slightly higher than coalesced).

With **batch 256 + QDP_PINNED_POOL_SIZE=4** (stream path): GPU utilization still only **~55% at best**. The pool is a **single GPU master thread**; bottleneck is feed cadence (registration tax, serial submission). See [QDP_OPTIMIZATION_PLAN_EN.md](QDP_OPTIMIZATION_PLAN_EN.md) Part Q/R (R.1 recycled pinning done; R.2/R.3 look-ahead planned).

To improve:

1. **Pool (Linux default)**: **encode_batch** and **encode_stream** use a **single master** + bounded queue (`QDP_BATCH_QUEUE_CAPACITY`, default 4). Zero-copy when caller reuses the same buffer (R.1). Tune `in_flight` and larger batch or coalesced for throughput.
2. **Coalesced path**: Use **`encode_list`** with a large chunk when you can batch many samples; one pipeline run, less Python overhead.
3. **Larger batch / pool**: e.g. `--batch-size 256` and `QDP_PINNED_POOL_SIZE=4` improve within-run overlap.

## Tuning run (2026-01-31): batch 256, pool 4

With `QDP_PINNED_POOL_SIZE=4` and `--batch-size 256` (stream path, 40 batches, 3 trials):

- **Throughput**: median 976.8 vec/s, P95 1028.6 vec/s
- **Latency**: median 1.122 ms/vec, P95 1.145 ms/vec
- **H2D overlap**: Chunk 0 often 58–62%, Chunk 10 often 52–56% (pool 4 in use)

Report: [results/pipeline_baseline_20260131_stream_inflight4_rep_config.md](results/pipeline_baseline_20260131_stream_inflight4_rep_config.md) (overwritten by this run). Compare with batch_size 64 runs for GPU utilization and throughput.

## Benchmarking

From the repo root or `qdp/qdp-python/benchmark`:

```bash
# Stream path (recommended for throughput in current baseline)
uv run python run_pipeline_baseline.py --qubits 16 --batch-size 64 --prefetch 16 --batches 100 --trials 5 --use-stream --in-flight 4

# With larger batch and pool (tuning)
QDP_PINNED_POOL_SIZE=4 uv run python run_pipeline_baseline.py --qubits 16 --batch-size 256 --prefetch 16 --batches 50 --trials 5 --use-stream --in-flight 4

# Coalesced path
uv run python run_pipeline_baseline.py --qubits 16 --batch-size 64 --prefetch 16 --batches 100 --trials 5 --use-coalesced
```

Results are written to `qdp/docs/optimization/results/`. See [QDP_OPTIMIZATION_PLAN_EN.md](QDP_OPTIMIZATION_PLAN_EN.md) for bottleneck analysis and Part Q/R for Phase 3.

## encode_stream in_flight (Linux)

On Linux, **encode_stream** uses **encode_batch_submit** + get in order: batches go to a **single master** pool. **in_flight** (default 4) limits how many batches are queued before the consumer calls get(); backpressure keeps the pool fed. Tune with larger batch or coalesced path.

## Related docs

- **Full plan**: [QDP_OPTIMIZATION_PLAN_EN.md](QDP_OPTIMIZATION_PLAN_EN.md) (goals, bottlenecks, single pipeline, coalescing, Part G/H/I).
- **Observability**: [OBSERVABILITY_USAGE.md](../observability/OBSERVABILITY_USAGE.md) (pool metrics, overlap tracking).
- **Results**: [results/](results/) (pipeline baseline reports and comparisons).
