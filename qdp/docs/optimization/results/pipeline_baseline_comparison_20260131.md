# Pipeline baseline comparison (2026-01-31)

Same hardware: RTX 3080, Driver 560.94, CUDA 12.8. Config: qubits=16, batch_size=64, prefetch=16, batches=200, amplitude.

## Paths compared

| Path | Report | Throughput (median) | Throughput (P95) | Note |
|------|--------|---------------------|------------------|------|
| **Default** (no stream) | pipeline_baseline_20260131_rep_config.md | **808.8** vec/s | 848.2 | encode(batch) → pool (submit+get) |
| **Stream** (in_flight=4) | pipeline_baseline_20260131_stream_inflight4_rep_config.md | **816.0** vec/s | 864.5 | encode_batch_submit + get in order |

- Default and stream **both use the Rust BatchPool** on Linux (encode_batch_via_pipeline_from_pinned).
- Stream path is slightly higher (~+0.9% median) because it submits multiple batches then gets in order, keeping the pool queue fed.

## GPU utilization (observed)

- **Observation**: During the run, GPU utilization was **a bit higher** than before (single-batch path), but **still did not exceed ~55%**.
- **Interpretation**: The Rust pool (2 workers, default `QDP_CONCURRENT_GPU_SLOTS=2`) and from_pinned pipeline reduce CPU copy and keep 2 batches in flight on the GPU, so utilization improves somewhat; the cap around 55% suggests the workload (batch 64, 16 qubits) is still not enough to saturate the device, or memory/scheduler contention.

**QDP_CONCURRENT_GPU_SLOTS=4 on single GPU**: Tried with Rust pool; **GPU utilization decreased** (observed: 越來越低). More concurrent slots on a single GPU add contention/overhead; **keep default 2** for single GPU. Prefer larger batch or coalesced path to push throughput/GPU % instead of increasing slots.

## Historical reference (pre–single-path)

- **stream_inflight4_concurrent4_rep_config.md** (old run with `concurrent_slots=4`, Python thread pool): 1219.2 vec/s median. That path was removed (GIL serialized per-batch copy; GPU % did not improve). Throughput there was measured with 5 trials; not directly comparable. Current design uses Rust pool only; no Python thread pool.

## Summary

| Metric | Default | Stream | Δ (stream − default) |
|--------|---------|--------|------------------------|
| Throughput median | 808.8 | 816.0 | **+7.2** (+0.9%) |
| Throughput P95 | 848.2 | 864.5 | +16.3 (+1.9%) |
| GPU % (observed) | — | Slightly higher, **still &lt;55%** | — |

**Next steps to push GPU % higher**: **Do not** use `QDP_CONCURRENT_GPU_SLOTS=4` on single GPU (observed to lower utilization). Try `--batch-size 256` or `--use-coalesced` with a large batch; re-run and observe with nvitop. See QDP_OPTIMIZATION_PLAN_EN.md Part G.

---

*Generated from run_pipeline_baseline.py results and user observation (GPU % &lt;55%).*
