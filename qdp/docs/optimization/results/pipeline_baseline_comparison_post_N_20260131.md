# Pipeline baseline comparison (post Part N.1–N.7, 2026-01-31)

Same hardware: RTX 3080, Driver 560.94, CUDA 12.8. Config: qubits=16, batch_size=64, prefetch=16, amplitude.

## Stack after Part N (single GPU)

- **N.1** Zero-copy ingress (`cudaHostRegister` when caller provides contiguous buffer).
- **N.2** Descriptors in queue (`BatchJobData`: Owned / Borrowed).
- **N.3** Bounded queue (`QDP_BATCH_QUEUE_CAPACITY`, default 4).
- **N.4** Single GPU master thread (one worker, no `QDP_CONCURRENT_GPU_SLOTS`).
- **N.5** Fused L2+amplitude kernel (one launch per chunk).
- **N.6** Pipeline final sync via `cudaStreamQuery` loop + `yield_now()` (no blocking `cudaStreamSynchronize`).
- **N.7** Optional thread affinity (feature `thread_affinity` + `QDP_MASTER_CPU_ID`; not used in this run).

All paths use the same pool: single master, bounded job queue, same pipeline (from_pinned, fused kernel).

## New runs (2026-01-31, post N.1–N.7)

| Path | batches | trials | Throughput (median) | Throughput (P95) | Latency median (ms/vec) | Latency P95 |
|------|--------|--------|--------------------|------------------|--------------------------|-------------|
| **Default** | 200 | 5 | **789.8** | 865.2 | 1.431 | 1.488 |
| **Stream** (in_flight=4) | 200 | 5 | **783.0** | 844.4 | 1.379 | 1.465 |
| **Coalesced** | 100 | 3 | **622.6** | 736.3 | 1.713 | 1.739 |

Reports: `pipeline_baseline_20260131_rep_config.md`, `pipeline_baseline_20260131_stream_inflight4_rep_config.md`, `pipeline_baseline_20260131_coalesced_rep_config.md`.
Run with minimal logging: `RUST_LOG=warn`, `QDP_ENABLE_OVERLAP_TRACKING=0`, `QDP_ENABLE_POOL_METRICS=0`.

## Comparison with previous baseline (2026-01-31, pre–single-master)

Previous comparison doc (`pipeline_baseline_comparison_20260131.md`) reported:

| Path | Throughput median (old) | Throughput median (new) | Δ |
|------|-------------------------|-------------------------|---|
| Default | 808.8 vec/s | 789.8 vec/s | **−19.0** (−2.4%) |
| Stream | 816.0 vec/s | 783.0 vec/s | **−33.0** (−4.0%) |

- Throughput is within normal run-to-run variance (same order of magnitude; default and stream both ~780–810 vec/s).
- **Latency**: Stream path **1.379 ms/vec** (new) vs **1.477 ms/vec** (old stream report) → **−6.6%**; N.6 (query sync + yield) may reduce wait in the driver.
- Coalesced (batches=100): 622.6 vec/s (new) vs 665.0 vec/s (old) → slightly lower; coalesced uses different batch count and workload shape.

## Summary

| Metric | Default (new) | Stream (new) | Coalesced (new) |
|--------|---------------|-------------|-----------------|
| Throughput median | 789.8 | 783.0 | 622.6 |
| Throughput P95 | 865.2 | 844.4 | 736.3 |
| Latency median (ms/vec) | 1.431 | **1.379** | 1.713 |
| Latency P95 (ms/vec) | 1.488 | **1.465** | 1.739 |

- **Default vs stream**: Throughput nearly the same; stream has slightly better latency (1.379 vs 1.431 ms/vec).
- **Coalesced**: Lower throughput (622.6) and higher latency (1.713) in this config (batches=100, trials=3); as in the plan, coalesced tends to trade latency for batching.
- Part N changes (single master, bounded queue, zero-copy, fused kernel, query sync) are in place; numbers are stable and comparable to the previous baseline. For higher throughput, the plan suggests larger batch (e.g. 256) or tuning `QDP_BATCH_QUEUE_CAPACITY` / `QDP_PINNED_POOL_SIZE`.

---

*Generated from run_pipeline_baseline.py (post N.1–N.7).*
