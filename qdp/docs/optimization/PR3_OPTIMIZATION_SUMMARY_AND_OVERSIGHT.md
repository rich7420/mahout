# PR3 優化總結與疏忽分析（Fix 1–Fix 4）

**目的**：整理目前為止所有優化文件，並根據程式碼找出我們疏忽的地方，解釋為何 GPU/CPU 利用率始終不高（GPU <60%、CPU <50%）。

---

## 一、文件索引（目前保留）

| 文件 | 內容 |
|------|------|
| **OPTIMIZATION_ROADMAP.md** | 優化路線圖 |
| **PR3_COMPLETE_IMPLEMENTATION_PLAN.md** | Phase 3 完整實作計畫 |
| **PR3_PHASE3_DETAILED_PLAN.md** | Phase 3 細部計畫 |
| **PR3_OPTIMIZATION_SUMMARY_AND_OVERSIGHT.md** | 本文件：總結、疏忽分析、encode_batch pipeline 設計、優化選項 |
| **REQUEST_COALESCING_REFERENCE_AND_DESIGN.md** | Request Coalescing 參考：Triton / tower-batch / Ray Serve 官方文件與原始碼、相關論文（SMDP、Symphony、irregular workload batching）、可改進點、QDP 設計與實作要點；可同時用於 encode() 與 encode_batch() |
| **results/fix4_rebuild_instructions.md** | 為何 Nsight 仍 800 次 sync、正確編譯/安裝流程 |
| **results/pipeline_baseline_20260130_rep_config.md** | 最新 baseline 報告 |
| **results/pr3_baseline_20260129_rep_config.md** | PR3 早期 baseline |
| **results/pr3_baseline_20260129_rep.md** | PR3 早期 baseline（rep） |

---

## 二、做到目前為止的改動摘要

| 階段 | 改動 | 預期效果 |
|------|------|----------|
| **Phase 3** | 動態 chunk / pool（PipelineConfig、env、硬體預設） | 依 PCIe/GPU 調參 |
| **Fix 1** | 預設 pool_size 由 4 改為 2（與 legacy 一致） | 減少 sync 等待量 |
| **Fix 2** | chunk 一律 8 MB（避免 16 MB 導致 H2D 約 2×） | 降低 H2D 時間 |
| **Fix 3** | Prologue–Body–Epilogue、breadth-first（先 issue copy next、再 wait copy current、再 compute） | 同一 batch 內 copy–compute 重疊 |
| **Fix 4** | Token Ring：固定 pinned slot、per-slot `synchronize_copy_event`，主迴圈不再 `sync_copy_stream()` | 減少主迴圈 stream sync、保留 overlap |

**實際結果**：throughput 仍約 −12% vs baseline（1274 vs 1454 vec/s），latency 約 +19%（0.858 vs 0.720 ms/vec）。GPU/CPU 利用率始終不高（GPU <60%、CPU <50%）。

---

## 三、疏忽：瓶頸在「每 batch 一次 sync」，不是 chunk 內 sync

### 3.1 Benchmark 實際呼叫鏈

- **腳本**：`run_pipeline_baseline.py` → `benchmark_throughput.run_mahout()`。
- **迴圈**（`benchmark_throughput.py` 約 96–106 行）：
  ```python
  for batch in prefetched_batches(..., total_batches=200, batch_size=64, ...):
      normalized = np.ascontiguousarray(normalize_batch(batch, ...), dtype=np.float64)  # shape (64, 65536)
      qtensor = engine.encode(normalized, num_qubits, encoding_method)   # 2D → Python 走 encode_batch()
      tensor = torch.from_dlpack(qtensor).abs().to(torch.float32)
      _ = tensor.sum()
      processed += normalized.shape[0]
  ```
- **重要**：`normalized` 為 **2D**（64 × 65536），Python bindings（`qdp-python/src/lib.rs`）在 `ndim == 2` 時呼叫 **`engine.encode_batch(data_slice, num_samples=64, sample_size=65536, ...)`**，**不是** `encode()`。因此 **Fix 1–4 的 `run_dual_stream_pipeline` 根本沒有被這個 benchmark 用到**；每批走的是 **amplitude `encode_batch()`**（`htod_sync_copy` + 單一 kernel + `device.synchronize()`），每批一次 sync。
- **結論**：**200 個 batch = 200 次 `encode_batch(64 samples)` = 200 次 Epilogue sync**；batch 與 batch 之間無 overlap，且 **pipeline 不在當前 benchmark 路徑上**。

### 3.2 Rust 端：當前 benchmark 走的是 `encode_batch()`，不是 pipeline

- **Benchmark 傳入 2D**：Python 呼叫 `engine.encode(normalized, ...)` 時 `normalized.shape == (64, 65536)`，bindings 走 **2D 分支** → **`encoder.encode_batch(..., num_samples=64, sample_size=65536, ...)`**（見 `qdp-python/src/lib.rs` 約 494–518 行）。
- **Amplitude `encode_batch()`**（`gpu/encodings/amplitude.rs` 約 188–293 行）：**未使用** `run_dual_stream_pipeline`；流程為 `htod_sync_copy(batch_data)` → norm kernel → encode_batch kernel → **`device.synchronize()`**。每呼叫一次就 **sync 一次**。
- **Pipeline 僅在 `encode()` 且 1D 且 ≥1 MB 時使用**：`encode()` 路徑（1D 資料）在 `host_data.len() >= ASYNC_THRESHOLD` 時才走 **encode_async_pipeline()** → **run_dual_stream_pipeline()**；Epilogue 為 `sync_copy_stream()` + `wait_for(stream_compute)`。**此路徑在「每 batch 傳 2D」的 benchmark 下不會被執行**。
- **因此**：**目前 benchmark = 200 次 `encode_batch(64)` = 200 次 `device.synchronize()`**；batch 間無 overlap，且 **Fix 1–4 的 pipeline 不在這條呼叫鏈上**。

### 3.3 為何 GPU/CPU 都上不去？

- 我們優化的是 **同一個 batch 內** 的 4 個 chunk（copy 與 compute 重疊、減少 chunk 內 stream sync）。
- 但 **batch 與 batch 之間** 是嚴格串行：
  1. 跑完 batch 1 的 4 chunks（這裡有 overlap，但只佔一小段時間）；
  2. Epilogue sync（copy + compute stream 都等完）；
  3. 回傳 DLPack 給 Python；
  4. Python 做 `from_dlpack`、`.abs()`、`.sum()`（可能再觸發 sync 或 kernel）；
  5. 才開始 batch 2。

所以 **GPU 大部分時間在等 host「下一個 batch」**，而不是連續被餵下一批 work；**CPU 則在等 GPU 與 Python 邊界**。因此 GPU <60%、CPU <50% 是「每 batch 一次 sync + 跨 batch 無 overlap」的必然結果，Fix 1–4 再怎麼改 chunk 內行為，都無法消除這 200 次 sync 與 200 次 batch 邊界。

---

## 四、程式碼證據整理

| 位置 | 證據 |
|------|------|
| **Benchmark 迴圈** | `benchmark_throughput.py`：`for batch in prefetched_batches(...): engine.encode(normalized, ...)`，`normalized` 為 2D (64, 65536) → 200 次呼叫。 |
| **Python 2D 分支** | `qdp-python/src/lib.rs` 約 494–518 行：`ndim == 2` 時呼叫 **`engine.encode_batch(data_slice, num_samples, sample_size, num_qubits, encoding_method)`**，不是 `encode()`。 |
| **encode_batch() 路徑** | `amplitude.rs` `encode_batch()`：`htod_sync_copy` + norm kernel + encode_batch kernel + **`device.synchronize()`**；**未使用** `run_dual_stream_pipeline`。 |
| **Pipeline 僅在 encode() 1D 大資料** | `amplitude.rs` `encode()`：僅當 `host_data.len() >= ASYNC_THRESHOLD`（1 MB）時走 `encode_async_pipeline()` → `run_dual_stream_pipeline()`；benchmark 傳 2D 故不經此路徑。 |

---

## 五、建議的改進方向（換地方改）

### 5.1 短期：pipeline / API 端改進（不依賴呼叫端包大 batch）

- **方向**：優化應在 **pipeline / encode_batch** 內完成；不應期望呼叫端先合併成一大份再呼叫。
- **方案 A**：**encode_batch 大 batch 走 pipeline**（§七之一、§七之二）：當單次 `encode_batch()` 的資料量超過門檻時，在 Rust 內改走雙 stream + chunk pipeline，Epilogue 只 sync 一次。呼叫端傳入大份或小份皆可，由 pipeline 決定是否走 pipeline 路徑。
- **方案 B**：**伸縮調度**（§七之四）：在 pipeline 內實作 chunk 上下界、單次調度上限（`max_batch_bytes` / `max_batch_samples`），必要時內部拆成多輪 pipeline，避免單次過大 OOM、過小多餘開銷。
- **方案 C**：若未來要保留「每 batch 一個 DLPack」的介面且仍要少 sync，可提供非同步 API（encode_async / submit+wait）或多 batch 一次提交，讓多個邏輯 batch 在 GPU 上排隊、延後 sync。

### 5.2 中期：非同步 / 雙緩衝

- **非同步 encode**：例如 `encode_async()` 回傳 handle/future，不在此呼叫結尾 sync；由呼叫端在需要結果時再 wait。這樣 Python 可以「連續丟多個 batch」再一起取結果，GPU 可連續吃多個 batch。
- **雙緩衝**：Python 端先 encode(batch1)、encode(batch2)，再 consume(batch1)、encode(batch3)、consume(batch2)… 需要 Rust 端支援「未 sync 就回傳 DLPack」或延後 sync（生命週期與 ownership 要設計清楚）。

### 5.3 驗證與量測

- **Profile**：用 Nsight 或 host 計時，量「Epilogue sync + Python 邊界」佔總時間比例，確認多數時間是否卡在 200 次 sync 與 batch 邊界。
- **對照實驗**：實作「單一大量 encode」（或單次 encode_batch 吃 200×64 樣本），同條件下跑一次，看 GPU/CPU 利用率與 throughput 是否明顯上升；若上升，即可印證瓶頸確實在「每 batch 一次 sync」。

---

## 六、官方與參考資料

| 主題 | 出處 | 要點 |
|------|------|------|
| **非同步執行與 overlap** | [CUDA Programming Guide §2.3 Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) | 非同步 API 回傳後可繼續做其他事；**同步應延後到真正需要結果時**；Stream/Event 為核心抽象；`cudaStreamSynchronize(stream)` 會 block 到該 stream 清空。 |
| **Stream 同步** | 同上 §2.3.2.4, §2.3.7 | `cudaStreamSynchronize(stream)` 會 block；**盡量用 Event 做細粒度等待**（`cudaStreamWaitEvent`），避免整條 stream sync；「Synchronization of any kind should be delayed as long as possible」。 |
| **多 stream 與 GPU 利用率** | [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/), [Measuring GPU Occupancy of Multi-stream Workloads](https://developer.nvidia.com/blog/measuring-the-gpu-occupancy-of-multi-stream-workloads/) | 多 stream 可讓 H2D/D2H 與 compute 重疊；**減少 `cudaStreamSynchronize` 呼叫**可避免 pipeline 停頓；pinned memory + 多個並行 queue 可讓 PCIe 接近理論峰值。 |
| **Host 端 pipeline 模式** | [CUDA Programming Guide §4.10 Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) | 多 buffer producer–consumer、**先 submit 下一批再 consume 當前批**；device 端為 `cuda::pipeline`，host 端對應為「多 batch 在飛、延後 sync」。 |
| **CUDA Graphs** | 同上 §2.3.9.2, [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) | 可將重複的 API 鏈錄成 graph，多次執行時 **降低 host 端開銷**，適合「同一 DAG 重複跑」的場景。 |
| **H2D 與 kernel 重疊（batch pipeline）** | [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) | Copy chunk N in stream A、kernel on chunk N-1 in stream B；多 chunk 依序 pipeline 可顯著縮短總時間。 |

---

## 七、優化選項對照與實作要點

| 選項 | 預期 sync 次數 | 程式改動要點 | 驗證方式 |
|------|----------------|--------------|----------|
| **A. 單次大 encode_batch（API 支援）** | 200 → **1** | 呼叫端一次傳入整份資料時，Rust 端 `encode_batch()` 已存在，目前為 sync 路徑（1 次 H2D + 1 次 kernel + 1 次 sync）。優化方向：大 batch 時改走 pipeline（§七之一、§七之二）並配合伸縮調度（§七之四），不依賴呼叫端先包大 batch。 | 對照「單次大 batch」與「多批小 batch」的 throughput／利用率，印證瓶頸在 sync 次數。 |
| **B. encode_batch 大 batch 走 pipeline** | 1 次 sync，但 H2D 與 compute 可重疊 | Rust：在 `amplitude.rs` 的 `encode_batch()` 中，當 `batch_data` 超過某門檻（如 1 MB）時改走 `run_dual_stream_pipeline` 的 batch 版（每 sample 或每 chunk 一組 H2D+kernel），Epilogue 只 sync 一次。 | 與 A 對比：A 為 baseline（1 sync、無 pipeline），B 看 H2D 與 kernel 重疊是否再拉高吞吐。 |
| **C. 非同步 encode API（延後 sync）** | 200 → 每 N 批 1 次或 1 次 | Rust：新增 `encode_async()` 或回傳 handle，**不在此呼叫結尾 sync**；由呼叫端在需要 DLPack 時再 wait。Python：可連續呼叫多次 encode_async 再一起取結果，讓多個 batch 在 GPU 上排隊。 | 需設計 DLPack 生命週期（未 sync 就回傳時，state 需由引擎持有至 wait）。 |
| **D. Python 雙緩衝（多 batch 在飛）** | 仍 200 次 encode，但 GPU 端可重疊 | Python：例如先 encode(b1)、encode(b2)，再 consume(b1)、encode(b3)、consume(b2)… 需 Rust 支援「未 sync 就回傳」或專用 API（如 submit + wait 分離）。 | 同 C，需 API 與生命週期設計。 |

**建議順序**：以 **B（pipeline 版 encode_batch）+ §七之四 伸縮調度** 為主，讓 pipeline 端無論呼叫端傳大或傳小都能少 sync、不 OOM、不過多開銷；再視需求考慮 C/D（非同步／雙緩衝）。

---

## 七之一、encode_batch 的 pipeline 應該也要改嗎？怎麼改？

**結論：要。** 目前 `encode_batch()` 完全同步，沒有 H2D 與 compute 重疊；大 batch 時應改為「按 sample 分 chunk、雙 stream H2D + compute 重疊」，最後只 sync 一次。

### 目前 encode_batch 的流程（無 overlap）

- **位置**：`qdp-core/src/gpu/encodings/amplitude.rs` 約 188–293 行。
- **步驟**：
  1. `device.htod_sync_copy(batch_data)` → **整批 H2D，blocking**
  2. `launch_l2_norm_batch(..., default stream)` → norm kernel
  3. `device.dtoh_sync_copy(inv_norms_gpu)` → **D2H 驗證，blocking**
  4. `launch_amplitude_encode_batch(..., default stream)` → encode kernel
  5. `device.synchronize()` → **整機 sync**
- **問題**：H2D、D2H、kernel 全部串行且多處 blocking，GPU/CPU 利用率都上不去。

### 建議：大 batch 時改走「chunk + 雙 stream pipeline」

- **時機**：當 `batch_data.len() * 8`（或 `num_samples * sample_size * 8`）超過門檻（例如 ≥ 8 MB）時，改走 pipeline 路徑；小 batch 維持現有 sync 路徑即可。
- **Chunk 單位**：按 **sample** 切（例如每 chunk 64 samples），不要按 raw bytes 切，以免切到 sample 中間。
  - `chunk_samples = 64`（或與 `PipelineConfig` 的 chunk 對齊），`chunk_elements = chunk_samples * sample_size`。
- **雙 stream 邏輯**（與現有 `run_dual_stream_pipeline` 同概念）：
  - **Copy stream**：對每個 chunk 做 `cudaMemcpyAsync`（pinned → device chunk buffer）。
  - **Compute stream**：對每個 chunk 依序「wait copy 完成 → `launch_l2_norm_batch`（chunk 範圍）→ `launch_amplitude_encode_batch`（chunk 範圍，state 寫入 offset）」。
- **Kernel 不需改簽名**：現有 `launch_l2_norm_batch(input_batch_d, num_samples, sample_len, inv_norms_out_d, stream)`、`launch_amplitude_encode_batch(input_batch_d, state_batch_d, inv_norms_d, num_samples, input_len, state_len, stream)` 都可接受 **子區間**，只要傳對指標即可：
  - Chunk 從 sample 索引 `start` 開始、共 `chunk_samples` 筆：
    - `input_chunk_d = input_batch_d + start * sample_size`
    - `state_chunk_d = state_batch_d + start * state_len`（以 complex 為單位則乘上對應 element 大小）
    - `inv_norms_chunk_d = inv_norms_d + start`
  - 呼叫時傳 `num_samples = chunk_samples`、對應的 ptr 即可。
- **Norm 驗證**：目前是整批 D2H 驗證；pipeline 版可 (1) 改為每 chunk 做完後 D2H 該 chunk 的 norms 再驗證，或 (2) 最後一次 D2H 整份 inv_norms 再驗證，或 (3) 在效能關鍵路徑先不做 D2H 驗證（或做成可關閉的選項）。
- **Epilogue**：所有 chunk 都排完後，**只做一次** `sync_copy_stream` + `wait_for(stream_compute)`，再 `device.synchronize()`（若仍需最後一道同步）；不要每 chunk sync。

### 實作要點（對應現有程式碼）

| 項目 | 位置／做法 |
|------|------------|
| 門檻 | 在 `encode_batch()` 開頭算 `batch_bytes = num_samples * sample_size * size_of::<f64>()`，若 `batch_bytes >= PIPELINE_THRESHOLD`（如 8M）走 pipeline 分支。 |
| Pinned / device chunk buffer | 可沿用 `PipelineContext` + `PinnedBufferPool` 的「固定 slot 數、per-slot event」模式，或為 encode_batch 專用建一組 copy/compute stream + 每 chunk 一組 pinned buffer + device chunk buffer。 |
| Copy stream | 對 chunk `i`：`async_copy_to_device(pinned_chunk_i, device_chunk_i, chunk_elements)`，再 `record_copy_done(slot_i)`。 |
| Compute stream | 對 chunk `i`：`wait_for_copy(slot_i)` → `launch_l2_norm_batch(..., stream_compute)`（chunk 指標）→ `launch_amplitude_encode_batch(..., stream_compute)`（chunk 指標與 state offset）。 |
| 現有 kernel | `qdp-kernels` 的 `launch_l2_norm_batch` / `launch_amplitude_encode_batch` 已接受 `(ptr, num_samples, ...)`，只要傳子區間的 ptr 與 `num_samples = chunk_samples` 即可，**不需改 kernel 簽名**。 |

### 參考

- [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)：Copy chunk N in stream A, run kernel on chunk N-1 in stream B。
- [CUDA Programming Guide §2.3 Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)：`cudaMemcpyAsync` + 非 default stream，延後 sync。

---

## 七之二、encode_batch() 參考 encode() 的實作計畫（文件內計畫）

以下計畫讓 `encode_batch()` 大 batch 時**直接沿用** `encode()` 的 pipeline 基礎設施（`run_dual_stream_pipeline_aligned` + Prologue–Body–Epilogue + Token Ring），僅在 kernel_launcher 裡改成「per-chunk 的 norm_batch + encode_batch」。

### 1. encode() vs encode_batch() 對照

| 項目 | encode()（大 1D 資料） | encode_batch()（目前） |
|------|------------------------|------------------------|
| **入口** | `amplitude.rs` 約 78–172 行：`host_data.len() >= ASYNC_THRESHOLD` → `encode_async_pipeline()` | `amplitude.rs` 約 188–293 行：一律 sync 路徑 |
| **資料** | 單一向量 `host_data: &[f64]`，長度 = state_len | 扁平 batch `batch_data: &[f64]`，長度 = num_samples × sample_size |
| **Norm** | 單一 inv_norm（CPU 或 GPU） | 每 sample 一個 inv_norm：`launch_l2_norm_batch` → `inv_norms_gpu[num_samples]` |
| **Pipeline** | `run_dual_stream_pipeline(device, host_data, kernel_launcher)` | **無**：整批 `htod_sync_copy` → norm → D2H 驗證 → encode → sync |
| **kernel_launcher** | `(stream, input_ptr, chunk_offset, chunk_len)` → `launch_amplitude_encode(input_ptr, state_ptr+offset, chunk_len, state_len, inv_norm, stream)` | — |
| **Chunk 單位** | `chunk_size_elements`（來自 PipelineConfig，可 8M） | 欲改為「每 chunk 多個 sample」，chunk 邊界對齊 `sample_size` |

### 2. 目標：encode_batch() 大 batch 走「同一個 pipeline API」

- **沿用**：`run_dual_stream_pipeline_aligned(device, batch_data, align_elements, kernel_launcher)`，其中 `align_elements = sample_size`，讓 chunk 邊界不切到 sample。
- **Chunk 大小**：`run_dual_stream_pipeline_aligned` 內會算 `chunk_size_elements` 為 `sample_size` 的整數倍（見 `pipeline.rs` 約 444–448 行），因此每個 chunk = 若干個完整 sample。
- **預先分配**（在進 pipeline 前）：
  - `batch_state_vector`：與現有相同，一整塊 state。
  - `inv_norms_gpu`：`device.alloc::<f64>(num_samples)`，一整塊給所有 sample 的 inv_norm；每個 chunk 寫入自己的區間 `inv_norms_gpu[start_sample..start_sample + chunk_samples]`。
- **kernel_launcher 簽名**（與 encode() 一致）：`F: FnMut(&CudaStream, *const f64, usize, usize) -> Result<()>`，參數為 `(stream, input_ptr, chunk_offset, chunk_len)`，其中 `input_ptr` 為**該 chunk 的 device 指標**（pipeline 會把 host 的 chunk 拷到 device buffer，再傳入此指標）。

### 3. kernel_launcher 內要做的事（對應 encode() 的單一 kernel）

- 由 `chunk_offset` / `chunk_len` 推得 sample 區間（對齊 `sample_size`）：
  - `start_sample = chunk_offset / sample_size`
  - `chunk_samples = chunk_len / sample_size`
- 使用**現有 kernel**，只傳子區間指標與長度：
  1. **Norm**：`launch_l2_norm_batch(input_ptr, chunk_samples, sample_size, inv_norms_gpu.ptr().add(start_sample), stream)`
     - `input_ptr` 即該 chunk 的 device 輸入；`inv_norms_gpu` 為整塊 buffer，此 chunk 寫入 `[start_sample, start_sample + chunk_samples)`。
  2. **Encode**：`launch_amplitude_encode_batch(input_ptr, state_chunk_ptr, inv_norms_gpu.ptr().add(start_sample), chunk_samples, sample_size, state_len, stream)`
     - `state_chunk_ptr` = batch state 的該段寫入位置：
       - state 若為 `f64*` 且 layout 為 `[num_samples, state_len*2]`（每個 state 為 state_len 個 complex = state_len*2 個 f64），則
       - `state_chunk_ptr = state_ptr.add(start_sample * state_len * 2)` 再轉成 `*mut c_void`，或依現有 `GpuStateVector` 的實際 layout 換算。
- **不需改 kernel 簽名**：`launch_l2_norm_batch` / `launch_amplitude_encode_batch` 已接受 `(ptr, num_samples, ...)`，只要傳對子區間即可。

### 4. 實作步驟（依序在文件／程式內對應）

| 步驟 | 位置／內容 |
|------|-------------|
| 1. 門檻分支 | 在 `encode_batch()` 開頭（約 196 行後）：若 `batch_data.len() * size_of::<f64>() >= PIPELINE_THRESHOLD_BYTES`（例如 8M），走「pipeline 路徑」，否則維持現有 sync 路徑。 |
| 2. 預分配 state + inv_norms | 與現有一致：`GpuStateVector::new_batch(...)`、`device.alloc::<f64>(num_samples)` 作為 `inv_norms_gpu`。 |
| 3. 呼叫 pipeline | 使用既有 **`run_dual_stream_pipeline_aligned`**（`pipeline.rs` 約 417–458 行）：
   - `run_dual_stream_pipeline_aligned(device, batch_data, sample_size, kernel_launcher)`
   - 這樣 chunk 邊界必為 `sample_size` 的倍數，不會切到 sample。 |
| 4. kernel_launcher 閉包 | 閉包內依 `chunk_offset`/`chunk_len` 算 `start_sample`、`chunk_samples`；依序呼叫 `launch_l2_norm_batch`、`launch_amplitude_encode_batch`（同上節），並傳入 `stream`。閉包需 capture：`state_ptr`（或 batch_state_vector）、`inv_norms_gpu`、`sample_size`、`state_len`。 |
| 5. Norm 驗證（可選） | Pipeline 結束後，若仍需驗證：一次 `device.dtoh_sync_copy(&inv_norms_gpu)` 再檢查；或熱路徑先關閉驗證以量測純 pipeline 效益。 |
| 6. Epilogue | 不需額外 sync：`run_dual_stream_pipeline_with_chunk_size` 的 Epilogue 已做 copy/compute stream sync（`pipeline.rs` 約 689–701 行）。 |

### 5. 與 encode() 的對照（可複用處）

- **encode()**：`encode_async_pipeline` → `run_dual_stream_pipeline(device, host_data, |stream, input_ptr, chunk_offset, chunk_len| { launch_amplitude_encode(...) })`（`amplitude.rs` 約 330–374 行）。
- **encode_batch()（計畫）**：大 batch → `run_dual_stream_pipeline_aligned(device, batch_data, sample_size, |stream, input_ptr, chunk_offset, chunk_len| { launch_l2_norm_batch(...); launch_amplitude_encode_batch(...) })`。
- 差異只有：**align_elements = sample_size**、kernel 改為 **norm_batch + encode_batch**、以及 **inv_norms_gpu 整塊 + state 的 offset 計算**；Prologue–Body–Epilogue、Token Ring、pinned slot、event 等邏輯全部沿用，**不需改 pipeline.rs**。

### 6. 其他優化方式（可與本計畫並行或之後做）

| 方式 | 說明 | 建議時機 |
|------|------|----------|
| **B. encode_batch pipeline（本計畫）** | 大 batch 時 encode_batch() 走 `run_dual_stream_pipeline_aligned`，H2D 與 compute 重疊，最後只 sync 一次；配合 §七之四 伸縮調度（chunk 上下界、max_batch、必要時內部拆批）。 | **主線**：優化在 pipeline，不依賴呼叫端包大 batch。 |
| **C. 非同步 API** | 新增 `encode_batch_async()` 或回傳 handle，不在此呼叫結尾 sync；由呼叫端在需要結果時再 wait，讓多個 batch 在 GPU 上排隊。 | 當「多 batch 在飛」為主要需求時。 |
| **D. Python 雙緩衝** | Python 端先 submit 多個 batch 再依序 consume，需 Rust 支援未 sync 就回傳或 submit/wait 分離。 | 同 C，需 API 與生命週期設計。 |

---

## 七之三、驗證與優化優先順序（Pipeline → CUDA Graphs）

以下階段與官方／論文依據一併列入 encode_batch() 優化路線，**優化在 pipeline 端**，不依賴修改 benchmark。

| 優先級 | 項目 | 做法摘要 | 驗證／依據 |
|--------|------|----------|------------|
| **中** | encode_batch Pipeline 版 | 大 batch 時走 Micro-Pipeline（chunk + 雙 stream），Epilogue 只 sync 一次；§七之四 伸縮調度（chunk 上下界、max_batch、必要時內部拆批） | §七之一、§七之二、§七之四；CUDA Pipelines + Overlap 文件 |
| **進階** | CUDA Graphs | CPU launch 仍為瓶頸時，再考慮 Graph 錄製與重複 launch | CUDA Programming Guide §4.2；Constant Time Launch 部落格 |

### 1. 驗證方式（對照實驗）

**目的**：確認「每批 sync」是瓶頸時，可透過 **單次大 batch API**（呼叫端一次傳入大份資料、只呼叫一次 `encode_batch()`）做對照；若該路徑下 throughput／GPU 利用率明顯高於「每批小 batch 各 sync 一次」，即印證瓶頸在 sync 次數。此對照不需改 benchmark 腳本邏輯，只要 Rust 端支援單次大 batch 且 pipeline 實作正確即可。

**依據**（官方／論文）：
- [CUDA Programming Guide §2.3 Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)：*"Synchronization of any kind should be delayed as long as possible."*
- [Advanced API Performance: Async Compute and Overlap](https://developer.nvidia.com/blog/advanced-api-performance-async-compute-and-overlap/)：重疊 host/device 傳輸與計算可減少或消除傳輸開銷。
- [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)：Copy chunk N in stream A、kernel on chunk N−1 in stream B。

### 2. 實作 encode_batch 的 Pipeline 版（中優先）

**目的**：若必須支援「小 batch 連續呼叫」，在 encode_batch 內部實作 **Micro-Pipeline**（§七之一、§七之二的 chunk + 雙 stream），讓單次大 batch 或多次小 batch 時 H2D 與 compute 重疊。

**做法**：大 batch 時走 `run_dual_stream_pipeline_aligned`，chunk 對齊 `sample_size`，kernel_launcher 內做 per-chunk 的 `launch_l2_norm_batch` + `launch_amplitude_encode_batch`；Epilogue 只 sync 一次。

**優點**：比改寫成 CUDA Graphs 簡單，且能解決多數 latency／利用率問題。

**依據**（官方／論文證明做得到）：
- [CUDA Programming Guide §4.10 Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)：多 buffer producer–consumer、先 submit 下一批再 consume 當前批；device 端 `cuda::pipeline`、host 端對應為「多 batch 在飛、延後 sync」。
- [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)：多 stream 讓 H2D 與 kernel 並行；後段 chunk 傳輸可與前段 chunk 計算重疊。
- 論文：*"Performance enhancement of CUDA applications by overlapping data transfer and kernel execution"*（PCIe 頻寬低於 GPU 記憶體頻寬時，重疊傳輸與計算可顯著改善）— 與 H2D + encode 重疊一致。

### 3. 引入 CUDA Graphs（進階，最後一哩路）

**目的**：若做完上述後 **CPU 仍是瓶頸**（Launch Overhead），再考慮 CUDA Graphs，把「同一 DAG 重複執行」的 launch 成本攤到一次建圖／instantiate。

**做法**：用 **Stream Capture**（`cudaStreamBeginCapture` / `cudaStreamEndCapture`）或 Graph API 把單次 encode_batch 的 H2D + norm + encode 錄成 graph，再反覆 `cudaGraphLaunch`；必要時用 `cudaGraphExecUpdate` 更新參數。

**依據**（官方證明做得到）：
- [CUDA Programming Guide §4.2 CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)：*"CPU launch costs are reduced compared to streams, because much of the setup is done in advance"*；*"presenting the whole workflow to CUDA enables optimizations which might not be possible with the piecemeal work submission mechanism of streams"*。
- [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)：graph 定義與執行分離，重複 launch 時開銷極低。
- [Constant Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)：straight-line graph 可常數時間 launch，適合固定 DAG 重複執行；適合 encode_batch 此類固定流程。

**建議**：僅在「驗證 + encode_batch Pipeline」做完、且 profiling 顯示 **CPU launch 佔比仍高** 時再導入 CUDA Graphs。

---

## 七之四、彈性 pipeline 調度（伸縮調度）：原則與優化清單

### 原則

1. **優化對象是 pipeline，不依賴呼叫端**
   改動應在 **Rust pipeline / encode_batch** 內完成；不應期望呼叫端（如 Python）先包好大 batch 或改寫迴圈。無論傳入的是多個小 batch 還是一個大 batch，pipeline 都應在內部做到「少 sync、多 overlap、不 OOM」。

2. **伸縮調度的目標**
   - **大一點的調度**：在記憶體允許下，把多個邏輯 batch 或 chunk 組合成較大的調度單元，減少 sync 次數與 batch 邊界開銷。
   - **避免太大**：單次調度過大會導致 GPU/主機記憶體不足（OOM），需有上限或依可用記憶體動態收斂。
   - **避免太小**：chunk/batch 過小會產生多餘的 sync、launch、copy 開銷，需有下限或依 PCIe/GPU 特性選擇合理粒度。

### 優化項詳細清單（實作可依序或並行）

| 優先級 | 優化項 | 說明 | 依據／參考 |
|--------|--------|------|------------|
| **1** | **encode_batch 一律可走 pipeline** | 不論呼叫端傳入的是單次大 batch 還是多次小 batch，只要單次 `encode_batch()` 的資料量超過門檻，就應走雙 stream pipeline（chunk + overlap），Epilogue 只 sync 一次。不依賴「呼叫端先合併成一大份」才有效能。 | §七之一、§七之二；[How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) |
| **2** | **Chunk 大小上下界與對齊** | **下界**：chunk 至少足夠讓 H2D 與 kernel 有 overlap 效益（NVIDIA 建議傳輸 ≥ 約 64 KB–1 MB 以攤平 TLP 開銷）；**上界**：單 chunk 不超過「可用 device 記憶體 / pool_size」且不超過合理 PCIe 延遲隱藏所需。**對齊**：`encode_batch` 的 chunk 邊界必須對齊 `sample_size`，不切到單一 sample 中間。 | [CUDA Best Practices – Optimize Data Transfers](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)；pipeline_config 現有 1–256 MB 與 20% host 上限 |
| **3** | **依可用記憶體動態決定單次調度上限** | 在進入 pipeline 前查詢或估算：device 可用記憶體、當前 process 已用 pinned 量。據此決定「單次 encode_batch 允許的最大 num_samples（或最大 bytes）」或「本輪 pipeline 最多排多少 chunk」，超過則在 **Rust 內部** 拆成多輪，每輪內部仍用 pipeline、Epilogue 只 sync 一次。這樣呼叫端傳一大份也不會爆開。 | 類似 [Memory-aware dynamic batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) 的「依資源調整 batch」；[PIPEMESH: elastic pipeline schedule](https://arxiv.org/html/2510.00606) 依 communication time 與 available memory 決定 micro-batch 數 |
| **4** | **PipelineConfig 擴充：chunk 上下界與「最大單次 batch」** | 在 `PipelineConfig` 中支援（env 或硬體推導）：`chunk_min_mb` / `chunk_max_mb`（或 elements）、`max_batch_bytes` 或 `max_batch_samples`。pipeline 與 encode_batch 讀取這些值，決定 chunk 大小與是否在內部拆批。 | 現有 `pipeline_config.rs` 已有 chunk_size_mb、pool_size；擴充欄位即可 |
| **5** | **內部拆批（多輪 pipeline）** | 當單次 `encode_batch(host_data)` 的 `host_data.len()` 超過 `max_batch_bytes`（或換算的 samples 超過 `max_batch_samples`）時，在 **encode_batch 內** 按 `max_batch_*` 切多段，每段呼叫同一套 `run_dual_stream_pipeline_aligned`，段與段之間只做一次 sync。對呼叫端仍是「一次 API、一大份資料」，但內部不會 OOM。 | [CUDA Programming Guide §2.3](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)：延後 sync；[Advanced API Performance: Async Compute and Overlap](https://developer.nvidia.com/blog/advanced-api-performance-async-compute-and-overlap/) |
| **6** | **小 batch 路徑不浪費** | 當單次資料量低於 pipeline 門檻時，維持現有 sync 路徑（htod_sync_copy + kernel + 一次 sync）。不強制走 pipeline，避免過多 chunk 導致 launch/copy 開銷大於 overlap 收益。門檻可與 `chunk_min_*` 一致或略高。 | [NVIDIA: chunk granularity](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：chunk 需足夠大以維持效率；過小會增加開銷 |
| **7** | **可選：非同步 API（submit / wait 分離）** | 若未來要支援「多個 batch 在飛」，可新增 `encode_batch_async` 或回傳 handle，Epilogue 不在此呼叫 sync；由呼叫端在需要結果時 wait。這樣即使呼叫端連續傳多個小 batch，GPU 也能連續吃、減少空轉。與伸縮調度可並存：內部仍用 chunk 上下界與 max_batch 避免 OOM。 | §五 5.2、§七 選項 C/D；CUDA Guide：synchronization delayed as long as possible |

### 參考文獻與官方資料（彈性／動態調度）

| 主題 | 出處 | 要點 |
|------|------|------|
| **H2D 與 kernel 重疊** | [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) | Copy chunk N in stream A、kernel on chunk N−1 in stream B；多 chunk pipeline 可顯著縮短總時間。 |
| **Chunk 粒度** | [CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | Chunk 需足夠大以達成效能、足夠多以維持 overlap；underutilized 時可考慮較小 chunk 增加並行度。 |
| **Elastic pipeline schedule** | PIPEMESH（e.g. [ElasWave / pipeline resharding](https://arxiv.org/html/2510.00606)） | 依 communication time 與 available memory 決定 micro-batch 數，在 overlap 與記憶體之間做 trade-off。 |
| **Dynamic batching（記憶體感知）** | [NVIDIA Triton Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)、[Memory-aware dynamic batching (arXiv)](https://arxiv.org/abs/2503.05248) | 依 GPU 記憶體與延遲約束動態調整 batch 大小；可參考「最大 batch 上限 + 動態收斂」的思路。 |
| **Host 端 pipeline 模式** | [CUDA Programming Guide §4.10 Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) | 多 buffer producer–consumer、先 submit 下一批再 consume 當前批；延後 sync。 |

### 小結（伸縮調度）

- **誰改**：pipeline / encode_batch 在 **Rust 端** 實作「chunk 上下界 + 單次調度上限 + 必要時內部拆批」，呼叫端無需包大 batch。
- **大不爆**：透過 `max_batch_bytes` / `max_batch_samples` 與內部拆批，單次傳入很大也不會 OOM。
- **小不浪費**：低於門檻走 sync 路徑；高於門檻走 pipeline，chunk 不小於下界、不超過上界，平衡 sync 次數與 overlap 效益。

---

## 七之五、Request Coalescing（請求合併）：參考與實作

**目的**：將多個小 encode / encode_batch 請求在 **Rust 端** 合併成一大 batch 再走一次 pipeline，減少 sync 次數；可同時用於 **encode()** 與 **encode_batch()**。

### 參考專案與官方／論文

| 專案 | 核心邏輯 | 可取之處 | 可改進／我們要的 |
|------|----------|----------|------------------|
| **NVIDIA Triton Inference Server** | 每個 model 有 scheduler；請求進佇列，dynamic batcher 依 **preferred_batch_size / max batch size** 與 **max_queue_delay_microseconds** 合併；Priority Queue、Queue Policy（timeout_action、max_queue_size）、preserve_ordering。 | 延遲–吞吐取捨明確；優先級與逾時政策完整；官方教學有實測（無/有 dynamic batching、多 instance）。 | 以請求個數與 delay 為主；我們需 **max_batch_bytes / max_batch_samples** 防 OOM。 |
| **Rust tower / tower-batch** | tower::buffer 為 MPSC 背壓；tower-batch 為「達到最大數量或最長等待時間後，多個 Service::call 合併成一個 batch」呼叫內層 Service。 | max size + timeout 雙條件；與 Tower Service 抽象相容。 | 底層為 async Service；我們需 sync 友善的 submit + flush；batch 單位要能按 bytes/samples。 |
| **Ray Serve** | `@serve.batch`：`_BatchQueue` + `wait_for_batch()`（第一個請求到達後計時，滿 **max_batch_size** 或 **batch_wait_timeout_s** 即返回）+ `_process_batches()` 迴圈；**batch_size_fn** 可自訂有效 batch 大小（如 total tokens）。 | 邏輯與 Triton 一致；batch_size_fn 對應我們「按 samples/bytes 上限」；原始碼 `batching.py` 結構清楚。 | 介面為 Python async；我們在 Rust 做等價的 submit + flush（可選 timeout 需 background）。 |
| **論文（SMDP、Symphony、irregular workload batching）** | SMDP 動態 batching 在延遲與功耗間取捨；Symphony deferred batch scheduling 在滿足延遲下收集更多請求；irregular workload 以有效 batch 大小（total tokens/nodes）防 OOM。 | 支持「延遲–吞吐」與「有效 batch 大小」設計。 | 我們首版用顯式 flush；可選 batch_wait_timeout 或更細 policy。 |

**官方文件與原始碼**：
- Triton：[Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)、[Dynamic Batching & Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html)；scheduler 在 [triton-inference-server/core](https://github.com/triton-inference-server/core)。
- Ray Serve：[Dynamic Request Batching](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html)、[ray.serve.batch API](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.batch.html)；實作 [python/ray/serve/batching.py](https://github.com/ray-project/ray/blob/master/python/ray/serve/batching.py)。

### QDP 實作要點（encode() 與 encode_batch() 共用）

- **抽象**：`EncodeRequest::Single(data)`（對應一次 encode）與 `EncodeRequest::Batch { data, num_samples, sample_size }`（對應一次 encode_batch）；同一 coalescer 內 sample_size / num_qubits 固定。
- **觸發**：**max_batch_samples**、**max_batch_bytes**、**顯式 flush()**；可選 batch_wait_timeout（需 background 或 async）。
- **流程**：`submit()` 入隊；`flush(device, encoder)` 將佇列 flatten 成一大塊，呼叫 `encoder.encode_batch(...)` 一次，回傳 `(GpuStateVector, Vec<(start_sample, num_samples)>)`，呼叫端依邊界切分。
- **程式位置**：`qdp-core/src/gpu/coalescer.rs`（`EncodeCoalescer`、`EncodeRequest`、`CoalescerConfig`）；`encode()` / `encode_batch()` 維持不變，上層可對同一 coalescer 多次 `submit` 再一次 `flush` 以達成請求合併。

**建議採納（見 REQUEST_COALESCING_REFERENCE_AND_DESIGN.md §一之五、§3.5.1–§3.6.1、§3.8、§3.9）**：
- **Zero-Copy Coalescer**：**EncodeRequest<'a>** 持有 `&'a [f64]`，Coalescer **不**維護內部 `Vec<f64>`，只存 metadata；**flush** 時從 **Pinned Buffer Pool** 取得一塊，**一次** 將所有 slice 寫入（可並行 memcpy），再送 **run_dual_stream_pipeline** → **Zero-Copy Ingress + One-Copy to Pinned**，避免兩次 Host Copy 抵銷 PCIe 優勢。
- **Scope-based API**：**run_coalesced(callback)**；Python **`with engine.coalesce() as batcher:`**，離開 Scope 時自動 flush；編譯器保證資料在 flush 前有效。
- **Pinned Memory Aware Batcher**：緩衝區用 Pinned Buffer Pool，flush 時不分配 `Vec`，直接寫入 Pinned Memory。
- **與 Pipeline 整合**：Coalescer flush 必須呼叫 **Fix 4 的 run_dual_stream_pipeline**（Chunk + Overlap），不呼叫舊的 sync encode_batch。
- **彈性調度器**：雙門檻、Pass-through、Scheduler thread。
- **Python 非同步**：encode_async + 延後 get，讓 Dynamic Batching 生效。

**詳細設計與對照表**：見 **REQUEST_COALESCING_REFERENCE_AND_DESIGN.md**（含 Ownership/Data Copy 疑慮、Zero-Copy 修正、Scope-based API、與 Pipeline 整合、總結優化清單）。

---

## 八、小結與「如何優化」

- **Fix 1–Fix 4** 都在優化 **`encode()` 1D 大資料** 路徑下的 chunk 級 pipeline（`run_dual_stream_pipeline`）。但 **當前 benchmark 傳的是 2D**，Python 走 **`encode_batch(64)`**，每批一次 **`device.synchronize()`**，**pipeline 根本不在這條路徑上**；200 批 = 200 次 sync。
- **疏忽**：
  1. 沒有及早確認「瓶頸在 batch 邊界 sync」。
  2. 沒有確認 benchmark 實際走的是 **encode_batch()** 而非 **encode()**，導致 Fix 1–4 的 pipeline 從未被此 benchmark 使用。
  官方文件也指出：**Synchronization of any kind should be delayed as long as possible**；目前是「每 batch 就 sync」，與此相反。
- **如何優化**：
  1. **優化在 pipeline**：不依賴呼叫端包大 batch；在 **encode_batch** 內實作大 batch 走 pipeline（§七之一、§七之二）與 **伸縮調度**（§七之四：chunk 上下界、max_batch、必要時內部拆批），使少 sync、不 OOM、小 batch 不過多開銷。
  2. **Request Coalescing**（§七之五）：參考 Triton、tower-batch、Ray Serve 與相關論文，實作 **EncodeCoalescer**（`qdp-core/src/gpu/coalescer.rs`），可同時用於 **encode()** 與 **encode_batch()**；上層多次 `submit` 再一次 `flush` 即合併成一大 batch 走一次 pipeline。詳見 **REQUEST_COALESCING_REFERENCE_AND_DESIGN.md**。
  3. **可選**：設計 **選項 C/D**（非同步 API 或雙緩衝），讓多個 batch 在 GPU 上排隊、延後 sync。
  4. **量測**：用 Nsight 或 host 計時確認「Epilogue sync + Python 邊界」佔總時間比例，避免再優化錯層級。

---

## 九、安全性與記憶體檢查（Memory Leak / 資源釋放）

照計畫改動時需確保無 memory leak、無 use-after-free、錯誤路徑也會釋放資源。以下依現有程式與計畫中的改動（encode_batch Pipeline、伸縮調度、CUDA Graphs）整理結論與建議。

### 9.1 官方與業界建議（證明「做得到且可驗證」）

| 主題 | 出處 | 要點 |
|------|------|------|
| **RAII 與 cudaFree** | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：§2.3 *"Production code should systematically check the error code returned by each API call"*；業界常見做法為用 wrapper 在 destructor 中呼叫 `cudaFree()`，避免遺漏釋放。 | 每個 allocation 都應有對應的 release；錯誤路徑也要執行到 destructor 或顯式 cleanup。 |
| **Memory leak 成因** | [Analyzing memory leaks in CUDA applications](https://app.studyraid.com/en/read/11728/371477/analyzing-memory-leaks-in-cuda-applications) 與 Stack Overflow：CUDA 記憶體洩漏常見於 **未呼叫 cudaFree**、**exception/early return 跳過 cleanup**、**動態指標未集中管理**。 | 建議：多個 allocation 用容器（如 Vec）集中持有，失敗時依序 drop 即可。 |
| **cudarc CudaSlice** | [cudarc](https://docs.rs/cudarc/latest/cudarc/)：`CudaSlice<T>` 在 Rust 中實作 **Drop**，drop 時會 deallocate 對應的 device 記憶體。 | 只要用 `CudaSlice` / `GpuStateVector`（內含 CudaSlice）持有 GPU 記憶體，Rust 的 RAII 會保證釋放。 |
| **CUDA Graphs 釋放** | [NVIDIA Forums: CUDA Graph Memory](https://forums.developer.nvidia.com/t/cuda-graph-memory-reservations/233369)：graph 銷毀後，device 記憶體可能留在 driver 的 reserved pool，不立即還給 OS；可透過 `cuDeviceGraphMemTrim()` 釋回。 | 先 `cudaGraphExecDestroy` 再 `cudaGraphDestroy`；若需將 graph 用過的記憶體還給 OS，再呼叫 `cuDeviceGraphMemTrim()`。 |

### 9.2 現有程式碼檢查結果

| 元件 | 資源 | 釋放方式 | 錯誤路徑是否安全 |
|------|------|----------|------------------|
| **encode_batch()**（amplitude.rs） | `batch_state_vector`（GpuStateVector）、`input_batch_gpu`（CudaSlice）、`inv_norms_gpu`（CudaSlice） | 皆為 owned 型別，**Drop 時釋放**（GpuStateVector 內為 Arc\<BufferStorage\> → CudaSlice）。 | **是**：任一 `?` 或 `return Err(...)` 會導致函式返回，Rust 會依序 drop 已建立的區域變數，不會漏放。 |
| **PipelineContext**（pipeline.rs） | `events_copy_done`、`events_compute_done`（CUDA events） | **impl Drop**：對每個 event 呼叫 `cudaEventDestroy`。 | **是**：context 建立後若 pipeline 中途失敗，ctx 被 drop 時會銷毀所有 event。 |
| **PinnedBufferHandle**（buffer_pool.rs） | 從 pool 借出的 pinned buffer | **impl Drop**：將 buffer 還回 pool（`pool.free.push(buf)`），不直接 cudaFreeHost。 | **是**：handle 離開 scope 即還回 pool；pool 持有的 `PinnedHostBuffer` 在 **PinnedHostBuffer::drop** 中呼叫 `cudaFreeHost`。 |
| **PinnedHostBuffer**（memory.rs） | 單一 pinned host  allocation | **impl Drop**：`cudaFreeHost(self.ptr)`。 | **是**。 |
| **run_dual_stream_pipeline_with_chunk_size** | `ctx`、`pinned_pool`、`device_buffers: Vec<CudaSlice<f64>>`、`pinned_slots: Vec<PinnedBufferHandle>` | 全部為 owned；Epilogue 後顯式 `drop(device_buffers); drop(pinned_slots);`，其餘在函式結束時 drop。 | **是**：任一中途 `?` 會 drop 所有已建立的 locals（含 ctx、pool、device_buffers、pinned_slots），無漏放。 |

**結論**：目前 **encode_batch** 與 **pipeline / buffer pool / events** 均依賴 Rust RAII 與集中持有（Vec、Arc），**無額外 memory leak 風險**；錯誤路徑也會觸發 drop。

### 9.3 計畫改動的安全性評估

| 改動 | 風險點 | 結論與實作建議 |
|------|--------|----------------|
| **1. encode_batch 的 Pipeline 版** | 在 encode_batch 內先分配 `batch_state_vector`、`inv_norms_gpu`，再呼叫 `run_dual_stream_pipeline_aligned(..., kernel_launcher)`；closure 會 capture 上述 buffer 的指標／引用。 | **無 leak**：batch_state_vector、inv_norms_gpu 由 encode_batch 擁有，pipeline 回傳（Ok 或 Err）後才離開 scope，其後一併 drop。**Use-after-free**：kernel_launcher 僅在 pipeline 主迴圈內被呼叫，且 Epilogue 會 sync，不會在 pipeline 回傳後再使用這些 buffer；**安全**。 建議：closure 只 capture 必要指標／引用，不要在 kernel_launcher 內再 clone Arc 或延長生命週期。 |
| **2. 伸縮調度（內部拆批）** | 當單次資料超過 max_batch 時，encode_batch 內拆多段、每段走 pipeline；每段 alloc 的 buffer 需在該段結束後 drop，避免累積。 | **無 leak**：每段為獨立 pipeline 呼叫，buffer 為區域變數，段結束即 drop。**OOM 防護**：max_batch 上限需依 device 可用記憶體與 pinned 上限設定。 |
| **3. CUDA Graphs（進階）** | Graph / GraphExec 與錄製時用到的 stream、記憶體需正確釋放；graph 記憶體池可能不會立刻還給 OS。 | **實作時**：以 RAII 包裝 `cudaGraph_t` / `cudaGraphExec_t`，在 Drop 中依序呼叫 `cudaGraphExecDestroy`、`cudaGraphDestroy`。若需將 graph 用過的記憶體釋回 OS，可在適當時機呼叫 `cuDeviceGraphMemTrim()`。 避免在 **capture 區間內** 使用會動態 alloc/free 的 API（參見 [CUDA Graph capture + async memory](https://stackoverflow.com/questions/73087828/error-with-a-captured-cuda-graph-and-asynchronous-memory-allocations-in-a-loop)），否則可能導致錯誤或未定義行為。 |

### 9.4 實作 encode_batch Pipeline 時的檢查清單

- [ ] **門檻分支**：大 batch 走 pipeline、小 batch 走現有 sync 路徑；兩條路徑的 alloc 皆為 owned（GpuStateVector、CudaSlice），無共用 mutable 靜態狀態。
- [ ] **錯誤路徑**：pipeline 內任一 `?` 或 `return Err` 時，已建立的 `batch_state_vector`、`inv_norms_gpu`、以及 pipeline 內部的 `ctx`、`device_buffers`、`pinned_slots` 皆會依序 drop，無需手動 cleanup。
- [ ] **kernel_launcher 生命週期**：閉包僅在 `run_dual_stream_pipeline_with_chunk_size` 的 body 內被呼叫，且 Epilogue 會 sync；閉包未將 buffer 指標傳出或存成長期引用，無 use-after-free。
- [ ] **現有 pipeline 行為**：不修改 `run_dual_stream_pipeline_with_chunk_size` 的 Drop 順序；仍先 sync 再 `drop(device_buffers); drop(pinned_slots);`，符合「所有 GPU 工作完成後再釋放」的語意。

### 9.5 小結

- **現有程式**：encode_batch、pipeline、buffer pool、events 均為 RAII + 集中持有，錯誤路徑會觸發 drop，**未發現 memory leak 或 use-after-free**。
- **encode_batch Pipeline 版 + 伸縮調度**：只要 buffer 由 encode_batch 擁有、closure 不延長生命週期、內部拆批時每段獨立 alloc/drop，**設計上可避免 leak 與 use-after-free**；實作時依 §9.4 檢查即可。
- **CUDA Graphs**：以 RAII 管理 graph/exec、依序 destroy，並視需要呼叫 `cuDeviceGraphMemTrim()`；capture 區間內避免動態 alloc/free，可參考官方文件與上述連結。
