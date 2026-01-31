# Pipeline baseline comparison (2026-01-31, latest run)

Same hardware: RTX 3080, Driver 560.94, CUDA 12.8. Config: qubits=16, batch_size=64, prefetch=16, encoding=amplitude.

## Run parameters

- **Default** & **Stream**: batches=80, trials=3.
- **Coalesced**: batches=80, trials=3.
- Observability: `RUST_LOG=warn`, `QDP_ENABLE_OVERLAP_TRACKING=0`, `QDP_ENABLE_POOL_METRICS=0`.

## Results (three paths)

| Path | batches | trials | Throughput median (vec/s) | Throughput P95 | Latency median (ms/vec) | Latency P95 |
|------|--------|--------|---------------------------|----------------|--------------------------|-------------|
| **Default** (per-batch encode) | 80 | 3 | **852.4** | 898.0 | **1.328** | 1.359 |
| **Stream** (encode_stream, in_flight=4) | 80 | 3 | **868.4** | 980.7 | **1.336** | 1.425 |
| **Coalesced** (encode_list) | 80 | 3 | **646.3** | 754.3 | **1.706** | 1.739 |

## Comparison summary

| Metric | Default | Stream | Coalesced |
|--------|---------|--------|-----------|
| Throughput median | 852.4 | **868.4** (+1.9% vs default) | 646.3 |
| Latency median (ms/vec) | **1.328** | 1.336 | 1.706 |

- **Default vs Stream**: Throughput slightly higher on stream (868.4 vs 852.4 vec/s); latency nearly the same (1.328 vs 1.336 ms/vec). Stream path uses zero-copy ingress (cudaHostRegister), single master, fused kernel, and cudaStreamQuery sync.
- **Coalesced**: Lower throughput (646.3 vec/s) and higher latency (1.706 ms/vec), as expected for the coalesced workload shape (encode_list batching).
- All three paths completed successfully; Phase 2 / N.1–N.7 stack (zero-copy, single master, bounded queue, fused kernel, sync_stream_via_query) is in place and stable.

## Source reports

- Default: `pipeline_baseline_20260131_rep_config.md`
- Stream: `pipeline_baseline_20260131_stream_inflight4_rep_config.md`
- Coalesced: `pipeline_baseline_20260131_coalesced_rep_config.md` (earlier run with batches=100 may differ; above numbers from run with batches=80)

---

*Generated after run_pipeline_baseline.py (default, --use-stream, --use-coalesced).*
