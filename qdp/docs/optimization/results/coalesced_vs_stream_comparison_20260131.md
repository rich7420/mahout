# Coalesced vs Stream comparison (2026-01-31)

Same config for both: qubits=16, batch_size=64, prefetch=16, batches=100, trials=5, amplitude.
Coalesced: `run_pipeline_baseline.py --use-coalesced`.
Stream: `run_pipeline_baseline.py --use-stream --in-flight 4`.

---

## Throughput and latency

| Metric | Coalesced | Stream | Δ (stream − coalesced) |
|--------|-----------|--------|-------------------------|
| **Throughput median** (vec/s) | 665.0 | **812.2** | **+147.2 (+22.1%)** |
| **Throughput P95** (vec/s) | 755.0 | **960.7** | +205.7 (+27.3%) |
| **Latency median** (ms/vec) | 1.447 | **1.265** | **−0.182 (−12.6%)** |
| **Latency P95** (ms/vec) | 1.470 | **1.301** | −0.169 (−11.5%) |

**Summary**: In this run, **stream is better on both throughput and latency**: ~22% higher median throughput and ~13% lower median latency than coalesced.

---

## GPU utilization (observed)

Measured with nvidia-smi / nvitop during the same runs:

| Path | GPU utilization | Note |
|------|-----------------|------|
| **--use-coalesced** | ~**15–35%** | Low. |
| **--use-stream** | ~**25–40%** | Low; slightly higher than coalesced. |

**Neither path achieves high GPU utilization.** The GPU is idle most of the time; the bottleneck is one batch at a time on the GPU and small batch size (see QDP_OPTIMIZATION_PLAN_EN.md Part G). Next steps: larger batch size (e.g. `--batch-size 256`), `QDP_PINNED_POOL_SIZE=4`, or future concurrent GPU submissions.

---

## H2D overlap (from logs)

- **Coalesced**: Per-chunk H2D overlap mixed — many **50–66%** (large `encode_list` chunks, multi-chunk pipeline runs) and many **8–26%** (smaller chunks or first chunk of a run). Coalesced does fewer, larger `encode_list` calls.
- **Stream**: Per-chunk H2D overlap mostly **8–27%** (one batch = one pipeline run, ~4 chunks of 8 MB each; slot 0 overlap is low by definition). Some **52–67%** when pipeline has more chunks in flight. Stream does 100 pipeline runs (one per batch).

Overlap metric is per-slot (copy vs compute for the same chunk); real pipeline overlap is between chunk N compute and chunk N+1 copy. Coalesced shows higher overlap values when each `encode_list` call has many chunks; stream shows lower per-slot values because each run is small (one batch).

---

## Interpretation

1. **Throughput**: Stream’s worker + queue keeps the GPU fed one batch at a time with less Python-side chunk building; coalesced’s loop that builds large chunks (and possibly list handling) may add overhead, so in this benchmark stream wins.
2. **Latency**: Stream’s per-batch latency is lower (1.27 ms vs 1.45 ms median), consistent with smaller, more frequent pipeline runs and less waiting for a big chunk to fill.
3. **Overlap**: Coalesced reaches 50–66% H2D overlap on large chunks; stream stays 8–27% per slot for most runs. Higher overlap on coalesced did not translate to better throughput in this run — chunk construction and Python overhead likely dominate.

**Next steps**: Try larger batch size (e.g. `--batch-size 256`) for both paths, or increase coalesced chunk size (if configurable) to reduce number of `encode_list` calls and see if coalesced throughput improves.

---

*Same git commit, GPU, driver, CUDA as baseline reports. Generated from run_pipeline_baseline.py output.*
