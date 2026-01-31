cd /home/rich-wsl/mahout/qdp/qdp-python/benchmark
export QDP_ENABLE_POOL_METRICS=1
export QDP_ENABLE_OVERLAP_TRACKING=1
export RUST_LOG=info
uv run python run_pipeline_baseline.py --qubits 16 --batch-size 64 --prefetch 16 --batches 500 --trials 20

# baseline report (see pr3_phase3_comparison_20260130.md for before/after comparison)

- **Date**: 2026-01-29
- **Git commit**: ef00f92eb236
- **GPU**: NVIDIA GeForce RTX 3080
- **Driver**: 560.94
- **CUDA**: 12.1

## Parameters

- qubits: 16
- batch_size: 64
- prefetch: 16
- batches: 500
- trials: 20
- encoding: amplitude

## Results

| Metric | Median | P95 |
|--------|--------|-----|
| Throughput (vectors/sec) | 1454.2 | 1742.3 |
| Latency (ms/vector) | 0.720 | 0.731 |


---

## 後續優化（Pipeline + Coalescing）與預期效益

- **先前瓶頸**：nsys 顯示 H2D 約 60%、`cuStreamSynchronize` 約 14%，且基準腳本為 **500 × engine.encode(batch)**，每次皆走 **sync encode_batch**，未使用 run_dual_stream_pipeline，導致 CPU/GPU 利用率偏低（約 50% / 55%）。
- **已做變更**：
  1. **QdpEngine::encode_batch** 改為呼叫 **encoder.encode_batch_via_pipeline**（§3.6.1），因此每次 `engine.encode(2D array)` 會走 **run_dual_stream_pipeline**（chunk + H2D/Compute 重疊）。同一腳本 `run_pipeline_baseline.py`（不加 `--use-coalesced`）預期會有較高 GPU 利用率與較佳 throughput。
  2. **基準腳本** 新增 **--use-coalesced**：使用 **encode_list** 以 chunk 為單位一次送多筆，減少 sync 次數，預期可進一步拉高 CPU/GPU 利用率與 throughput。
- **如何確認走新 pipeline**：`run_pipeline_baseline.py`（不加 `--use-coalesced`）呼叫 `engine.encode(2D array)` → Python binding 辨識 ndim==2 後呼叫 Rust `QdpEngine::encode_batch` → `qdp-core/src/lib.rs` 內 `encode_batch` 呼叫 `encoder.encode_batch_via_pipeline`（amplitude/angle 使用 `run_dual_stream_pipeline_aligned`）。編譯通過即表示此路徑已接上；可選 `RUST_LOG=qdp_core=debug` 觀察 `encode_batch_via_pipeline` 的 debug 日誌。
- **為何 CPU/GPU 利用率仍在 30–50%**（未加 `--use-coalesced` 時）：
  1. **500 次逐批同步**：腳本為 `for batch in prefetched_batches: normalize → engine.encode(batch) → torch.from_dlpack(...).sum()`。每次 `engine.encode()` 都會等整批（H2D + 計算 + D2H）完成才回傳，再跑 PyTorch `.sum()`。**批次與批次之間沒有重疊**：GPU 在「我們的 pipeline + PyTorch 消費」時忙，在「Python 正規化 + 下一批準備」時閒置。
  2. **主線程瓶頸**：主線程依序做：取 prefetch 資料 → `normalize`（CPU）→ `encode`（GPU）→ `from_dlpack` / `.abs()` / `.sum()`（GPU + 同步）。若單批 encode 時間 < 單批的 Python/消費時間，GPU 閒置比例就高，整體利用率約 30–50%。
  3. **Prefetch 只預備資料**：prefetch 16 只讓「下一批的 NumPy 資料」先準備好，**不會**讓「下一批 encode」與「當前批消費」重疊；encode 仍是一次一次呼叫，所以無法靠 prefetch 把 GPU 佔滿。
- **建議**：要拉高利用率請加 **`--use-coalesced`**（一次 `encode_list` 送多批，減少 Python↔Rust 往返與同步次數），或加大 **`--batch-size`**（例如 128/256）以增加每批 GPU 工作量。
- **若 `--use-coalesced` 報錯 `no attribute 'encode_list'`**：表示目前使用的 _qdp 擴展是舊版，需重建。**注意**：從 `qdp/qdp-python/benchmark` 跑 `uv run` 時，uv 會用 **qdp-python 的 .venv**，且 sync 可能重裝 qumat-qdp 並覆蓋剛建好的 .so。兩種做法：
  1. **從 repo 根目錄跑**（使用根目錄 .venv，不會被覆蓋）：
     ```bash
     cd /path/to/mahout
     uv sync --extra qdp --reinstall-package qumat-qdp   # 必要時先重建
     uv run python qdp/qdp-python/benchmark/run_pipeline_baseline.py ... --use-coalesced
     ```
  2. **從 qdp-python 目錄跑**：先重建並用該 venv 的 python 直接跑（不用 `uv run`，避免 sync 覆蓋）：
     ```bash
     cd qdp/qdp-python && .venv/bin/python -m maturin develop --release
     cd benchmark && ../.venv/bin/python run_pipeline_baseline.py ... --use-coalesced
     ```
- **若 `maturin develop` 出現 `Failed to set rpath ... patchelf` 警告**：maturin 用 patchelf 設定 .so 的 rpath，若未安裝會跳過此步驟。若執行時沒有「找不到 .so」錯誤可忽略；否則請安裝：`sudo apt install patchelf`（WSL/Ubuntu），或 `pip install maturin[patchelf]`（若 maturin 由 pip 安裝）。
- **建議重跑**（需 CUDA + 可編譯之 qdp-python）：
  ```bash
  cd qdp/qdp-python/benchmark
  export QDP_ENABLE_POOL_METRICS=1 QDP_ENABLE_OVERLAP_TRACKING=1 RUST_LOG=info
  uv run python run_pipeline_baseline.py --qubits 16 --batch-size 64 --prefetch 16 --batches 500 --trials 20
  uv run python run_pipeline_baseline.py --qubits 16 --batch-size 64 --prefetch 16 --batches 500 --trials 20 --use-coalesced
  ```
  比較兩次報告（throughput / latency）與 nsys 的 H2D / sync 佔比。

---

## 正常 vs --use-coalesced 比較（2026-01-30）

### 50 batches × 64，3 trials

| 模式 | Throughput (median) | Throughput (p95) | Latency (median) | Latency (p95) |
|------|---------------------|------------------|------------------|---------------|
| **正常** | **1225.5** vec/s | 1231.5 vec/s | **1.051** ms/vec | 1.065 ms/vec |
| **--use-coalesced** | 751.6 vec/s | 763.1 vec/s | 1.434 ms/vec | 1.464 ms/vec |

### 500 batches × 64，3 trials（批數做大）

| 模式 | Throughput (median) | Throughput (p95) | Latency (median) | Latency (p95) |
|------|---------------------|------------------|------------------|---------------|
| **正常** | **1025.8** vec/s | 1080.8 vec/s | **1.069** ms/vec | 1.077 ms/vec |
| **--use-coalesced** | 668.9 vec/s | 687.2 vec/s | 1.445 ms/vec | 1.459 ms/vec |

- **結論**：兩組參數下皆為**正常模式較快**（500 批時 throughput 約 +53%、latency 約 -26%）。兩者皆走 **run_dual_stream_pipeline**（H2D overlap 約 50–60%）；正常模式每批一次 FFI，coalesced 以 chunk 合併多批再一次 FFI。
- **可能原因**：coalesced 的 chunk 組裝、copy 進 pinned buffer、依 boundaries 切回等成本，在目前 chunk 設定下仍大於「少 FFI」的收益；或 coalesced 路徑的 chunk 大小/數量導致 GPU 利用率不如逐批。**後續**：可調大 coalesced 的 chunk（減少 chunk 數）、或縮小 batch_size 再測，觀察是否反超。

---

cd /home/rich-wsl/mahout/qdp/qdp-python && nsys profile --trace=cuda,nvtx --output=../docs/optimization/results/baseline_before_uv uv run python benchmark/benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16 --frameworks mahout

cd /home/rich-wsl/mahout/qdp && nsys stats docs/optimization/results/baseline_before_uv.sqlite

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)              Name
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------
     60.3       1154513612        200  5772568.1  5600405.0   4603514   8527926     818264.7  cuMemcpyHtoDAsync_v2
     14.1        269678972        800   337098.7   160114.5      1525   2679828     573425.8  cuStreamSynchronize
     12.7        243282018        200  1216410.1  1035882.0     76003   2659132     499445.4  cuMemcpyDtoHAsync_v2
      4.8         91270891        800   114088.6     7550.5       465   7939629     329877.7  cuMemAllocAsync
      3.3         63990841       1200    53325.7    23571.5      6390  17779574     652454.9  cudaLaunchKernel
      3.0         56531757        400   141329.4   135325.0     26292    422046      65129.1  cudaMemGetInfo
      0.4          8475510        400    21188.8    12548.0      7223    100009      16110.7  cudaMemsetAsync
      0.4          6802300        200    34011.5    32870.5     17636    213820      17152.8  cuLaunchKernel
      0.3          6583789        200    32918.9    24025.5     14132    144830      18166.2  cuMemsetD8Async
      0.2          4736816       3002     1577.9      863.5       145     67826       2298.3  cuCtxSetCurrent
      0.2          3457162          2  1728581.0  1728581.0      8156   3449006    2433048.4  cudaDeviceSynchronize
      0.1          2554445        800     3193.1     2796.0       970     47088       3201.1  cuMemFreeAsync
      0.1          1376489          4   344122.3   346245.0    301121    382878      37481.3  cudaMalloc
      0.0           174142          1   174142.0   174142.0    174142    174142          0.0  cuModuleLoadData
      0.0            42998        383      112.3       90.0        52       708         73.1  cuGetProcAddress_v2
      0.0            23813          7     3401.9     2823.0       338     11571       3797.4  cudaStreamIsCapturing_v10000
      0.0             1173          1     1173.0     1173.0      1173      1173          0.0  cuEventCreate
      0.0             1064          1     1064.0     1064.0      1064      1064          0.0  cuInit
      0.0              599          1      599.0      599.0       599       599          0.0  cuEventDestroy_v2
      0.0               91          1       91.0       91.0        91        91          0.0  cuModuleGetLoadingMode
