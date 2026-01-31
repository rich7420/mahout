# QDP Dataflow 與 Memory 實作分析

本文件詳細說明 QDP (Quantum Data Processing) 的 **dataflow 實作**與 **memory 管理**，並逐階段列出各路徑所呼叫的函數。

**對照程式碼路徑**：`qdp/qdp-core/src/`（Rust）、`qdp/qdp-core/src/gpu/`（GPU 管線與記憶體）。

**單 GPU 極致優化路徑**：若目標為突破目前 ~55% GPU 利用率，見 [QDP_OPTIMIZATION_PLAN_EN.md](QDP_OPTIMIZATION_PLAN_EN.md) **Part M**（硬核重構：零拷貝 cudaHostRegister、單線程 Master、Triple Buffer、Kernel Fusion）與 **Part N**（Clean-slate 實作清單）。本文件 §2、§3 描述之複製點與 dataflow 即為 Part M/N 欲消除或簡化之對象。

---

## 1. 總覽：Dataflow 架構

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  API 入口 (qdp-core/src/lib.rs: QdpEngine)                                       │
│  encode() | encode_batch() | encode_list() | encode_async()                      │
└─────────────────────────────────────────────────────────────────────────────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
┌─────────────┐    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Encoder     │    │ BatchPool       │   │ encode_list()    │   │ EncodeScheduler  │
│ .encode()   │    │ submit_handle() │   │ → run_coalesced  │   │ .submit(task)    │
│ (單一樣本)   │    │ → worker_loop   │   │ → coalescer.flush│   │ → scheduler_loop  │
└──────┬──────┘    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
       │                    │                     │                     │
       │                    │                     │                     │
       ▼                    ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  單一路徑：encode_batch_via_pipeline / encode_batch_via_pipeline_from_pinned        │
│  (encodings/amplitude.rs, angle.rs 等)                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  run_dual_stream_pipeline* (pipeline.rs)                                          │
│  - 雙 stream：stream_copy (H2D)、stream_compute (kernel)                           │
│  - Token Ring：pool_size 個 device buffer slot，event 同步                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  CUDA：cudaMemcpyAsync (H2D) + kernel (norm / encode)                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**原則**：所有 batch 編碼皆走 **單一 pipeline 路徑**（`run_dual_stream_pipeline*` + `encode_batch_via_pipeline` / `encode_batch_via_pinned`），無額外 sync-only 分支。

---

## 2. Memory 部分

### 2.1 Host 端記憶體

| 類型 | 檔案 | 說明 | 主要函數 |
|------|------|------|----------|
| **PinnedHostBuffer** | `gpu/memory.rs` | 單一 page-locked 緩衝區，供 H2D 高效傳輸 | `PinnedHostBuffer::new(elements)` → `cudaHostAlloc`；`as_slice()` / `as_slice_mut()`；Drop → `cudaFreeHost` |
| **RegisteredHostBuffer** | `gpu/memory.rs` | 對既有 host 記憶體做 cudaHostRegister，零拷貝 ingress | `RegisteredHostBuffer::new(ptr, len)` → `cudaHostRegister`；`as_slice()`；Drop → `cudaHostUnregister` |
| **PinnedBufferPool** | `gpu/buffer_pool.rs` | 固定大小的 Pinned 緩衝區池，Coalescer 使用 | `PinnedBufferPool::new(pool_size, elements_per_buffer)` → 多個 `PinnedHostBuffer::new`；`acquire()` → 取出一塊；`PinnedBufferHandle` Drop 時歸還 |
| **EncodeRequest** | `gpu/coalescer.rs` | 僅持有 `&[f64]`，Zero-Copy Ingress | `EncodeRequest::Single(&[f64])`、`EncodeRequest::Batch { data, num_samples, sample_size }`；`as_slice()` 供 flush 時複製 |
| **EncodeTask** | `gpu/scheduler.rs` | 跨 thread 傳遞時擁有資料 | `data: Vec<f64>`（`encode_async` 時 `data.to_vec()`） |
| **BatchJob** / **BatchJobData** | `gpu/batch_pool.rs` | 送進 BatchPool 的任務 | `batch_data: BatchJobData::Borrowed { ptr, len }` 或 `Owned(Vec<f64>)`；Borrowed 時 worker 用 `RegisteredHostBuffer`（0 次 host 複製）；Owned 時 worker 內一次複製到 `PinnedHostBuffer` |

**Pipeline 內 Host 端**（`pipeline.rs`）：

- `run_dual_stream_pipeline` / `run_dual_stream_pipeline_aligned`：內部建立 **暫存** `PinnedHostBuffer`，把 `host_data` 複製進去，再呼叫 `run_dual_stream_pipeline_with_chunk_size_from_pinned`。
- `run_dual_stream_pipeline_aligned_from_pinned`：**不再**複製到 pinned_slots，直接以呼叫端提供的 `pinned_data: &[f64]` 做 **async H2D only**。

### 2.2 Device 端記憶體

| 類型 | 檔案 | 說明 | 主要函數 |
|------|------|------|----------|
| **GpuStateVector** | `gpu/memory.rs` | 單一或 batch 量子態 (2^n 或 num_samples×2^n 個 complex) | `GpuStateVector::new(device, qubits)` → `device.alloc::<CuDoubleComplex>`；`new_batch(device, num_samples, qubits)`；`copy_sample_range()` → D2D；`to_precision()`；`to_dlpack()` |
| **BufferStorage** | `gpu/memory.rs` | F32/F64 複數緩衝區封裝 | `GpuBufferRaw<T>` 包裝 `CudaSlice<T>`，Drop 時釋放 |
| **Pipeline device_buffers** | `gpu/pipeline.rs` | 管線內 chunk 用的一組 device 緩衝區 | `run_dual_stream_pipeline_with_chunk_size_from_pinned` 內：`Vec<CudaSlice<f64>>`，共 `pool_size` 個，每個 `chunk_size_elements`；`ensure_device_memory_available()` 後 `device.alloc::<f64>(chunk_size_elements)` |
| **Encoder 暫存** | `gpu/encodings/amplitude.rs` | norm 用暫存 | `device.alloc_zeros::<f64>(num_samples)`（inv_norms_gpu） |

### 2.3 配置與參數（Pipeline）

| 組件 | 檔案 | 說明 |
|------|------|------|
| **PipelineConfig** | `gpu/pipeline_config.rs` | `chunk_size_mb`、`pinned_pool_size`；env：`QDP_CHUNK_SIZE_MB`、`QDP_PINNED_POOL_SIZE`、`QDP_PCIE_GEN`；`get_or_build(device, host_mem_gb)` 快取 per-device；`chunk_size_elements()`、`pinned_pool_size_resolved()` |
| **CoalescerConfig** | `gpu/coalescer.rs` | `max_batch_samples`、`max_batch_bytes`、`target_batch_size`、`max_delay`；Coalescer 的 PinnedBufferPool 單 block 容量 = `max_batch_bytes` 對應 elements |
| **SchedulerConfig** | `gpu/scheduler.rs` | `coalescer`、`channel_capacity`、`max_delay`、`target_batch_size` |

---

## 3. 各 API 路徑與階段函數呼叫

### 3.1 encode(data, num_qubits, encoding_method) — 單一樣本

**入口**：`qdp-core/src/lib.rs`：`QdpEngine::encode()`

| 階段 | 呼叫函數（依序） |
|------|------------------|
| 1 | `get_encoder(encoding_method)` → 取得 `Box<dyn QuantumEncoder>` |
| 2 | `encoder.encode(&self.device, data, num_qubits)` |
| 3 | `state_vector.to_precision(&self.device, self.precision)` |
| 4 | `state_vector.to_dlpack()` |

**encoder.encode() 內部**（以 Amplitude 為例，`encodings/amplitude.rs`）：

| 階段 | 呼叫函數 |
|------|----------|
| 2.1 | `Preprocessor::validate_input(host_data, num_qubits)` |
| 2.2 | `GpuStateVector::new(_device, num_qubits)` |
| 2.3 | 若 `host_data.len() < ASYNC_THRESHOLD`（1MB）：<br>• `ensure_device_memory_available(...)`<br>• `_device.htod_sync_copy(host_data)`<br>• （可選）`Self::calculate_inv_norm_gpu(...)` 或 `Preprocessor::calculate_l2_norm(host_data)`<br>• `launch_amplitude_encode(...)`<br>• `_device.synchronize()` |
| 2.4 | 否則（大資料 async pipeline）：<br>• `Preprocessor::calculate_l2_norm(host_data)`<br>• `Self::encode_async_pipeline(device, host_data, num_qubits, state_len, inv_norm, &state_vector)` |
| 2.5 | **encode_async_pipeline** 內：<br>• `run_dual_stream_pipeline(device, host_data, kernel_launcher)`<br>• kernel_launcher 內：`launch_amplitude_encode(input_ptr, state_ptr_offset, chunk_len, state_len, inv_norm, stream)`<br>• 若 `data_len < state_len`：`cudaMemsetAsync` 補零 + `device.synchronize()` |

**run_dual_stream_pipeline**（見 3.5 節）會：建 `PinnedHostBuffer`、複製 host_data、再呼叫 `run_dual_stream_pipeline_with_chunk_size_from_pinned`。

---

### 3.2 encode_batch(...) — 經 BatchPool（Linux 預設）

**入口**：`lib.rs`：`QdpEngine::encode_batch()` → `batch_pool.submit_handle(...)?.get()`

| 階段 | 呼叫函數 |
|------|----------|
| 1 | `self.batch_pool.submit_handle(batch_data.to_vec(), num_samples, sample_size, num_qubits, encoding_method.to_string())` |
| 2 | `BatchPool::submit_handle()`：`mpsc::channel()`，組 `BatchJob { batch_data, num_samples, sample_size, num_qubits, encoding_method, response_tx }`，`self.submit(BatchJob {...})`，回傳 `BatchHandle { rx }` |
| 3 | `BatchHandle::get()`：`rx.recv()` 取得 `Result<Arc<GpuStateVector>>`，再 `batch.to_dlpack()` |

**Worker 內**（`batch_pool.rs`：`worker_loop`）：

| 階段 | 呼叫函數 |
|------|----------|
| W1 | `job_rx.recv()` 取得 `BatchJob` |
| W2 | `get_encoder(&job.encoding_method)` |
| W3 | `PinnedHostBuffer::new(job.batch_data.len())` |
| W4 | `pinned.as_slice_mut()[..len].copy_from_slice(&job.batch_data)`（一次複製到 pinned） |
| W5 | `encoder.encode_batch_via_pipeline_from_pinned(&device, pinned.as_slice(), job.num_samples, job.sample_size, job.num_qubits)` |
| W6 | `state.to_precision(&device, precision)` |
| W7 | `job.response_tx.send(Ok(Arc::new(state)))` |

**encode_batch_via_pipeline_from_pinned**（amplitude，見 3.6 節）內部會呼叫 `run_dual_stream_pipeline_aligned_from_pinned`，不再做 host 端複製到 pinned_slots，僅 async H2D。

---

### 3.3 encode_list(slices, sample_size, num_qubits, encoding_method)

**入口**：`lib.rs`：`QdpEngine::encode_list()` → `crate::gpu::encode_list(...)`

| 階段 | 呼叫函數 |
|------|----------|
| 1 | `get_encoder(encoding_method)` |
| 2 | `crate::gpu::encode_list(&self.device, encoder.as_ref(), sample_size, num_qubits, config, slices)` |
| 3 | `batch_state.to_precision(...)`，`state_vector.to_dlpack()` |

**encode_list**（`coalescer.rs`）：

| 階段 | 呼叫函數 |
|------|----------|
| L1 | `run_coalesced(device, encoder, sample_size, num_qubits, config, callback)` |
| L2 | callback 內：對每個 `s in slices` 做 `batcher.submit(EncodeRequest::Single(s))` |
| L3 | `run_coalesced` 內：`EncodeCoalescer::new(sample_size, num_qubits, config)`，`callback(&mut coalescer)`，`coalescer.flush(device, encoder)` |

**flush**（`coalescer.rs`）：

| 階段 | 呼叫函數 |
|------|----------|
| F1 | 檢查 `!queue.is_empty()`，`total_elements`、`total_samples`，`pool.elements_per_buffer()` 容量檢查 |
| F2 | `pool.acquire()` → `PinnedBufferHandle`，`handle.as_slice_mut()` 取得可寫 slice |
| F3 | 建 `boundaries`: 每個 request 的 `(offset, len)` |
| F4 | `SendPtr(base_ptr).copy_into(off, src)` 在 rayon `par_iter()` 下並行複製到 pinned |
| F5 | `encoder.encode_batch_via_pipeline_from_pinned(device, &pinned_slice[..total_elements], total_samples, sample_size, num_qubits)` |
| F6 | 從 `queue` 建 `sample_boundaries`，`queue.clear()`，`drop(handle)` 歸還 buffer |

因此 **encode_list** 路徑：`encode_list` → `run_coalesced` → `EncodeCoalescer::new` + `submit`(多筆) → `flush` → **encode_batch_via_pipeline_from_pinned** → pipeline（見 3.5、3.6）。

---

### 3.4 encode_async(data, num_qubits, encoding_method)

**入口**：`lib.rs`：`QdpEngine::encode_async()`（僅 Linux）

| 階段 | 呼叫函數 |
|------|----------|
| 1 | `get_encoder(encoding_method)`（僅驗證） |
| 2 | `schedulers.lock()`，`guard.entry(key).or_insert_with(|| EncodeScheduler::new(...))`，取得或建立 `EncodeScheduler` |
| 3 | `mpsc::channel()`，組 `EncodeTask { data: data.to_vec(), num_samples: 1, sample_size, tx }`，`scheduler.submit(task)` |
| 4 | `EncodeHandle::new(rx, device, precision, num_qubits)` 回傳 |

**EncodeHandle::get()**（`scheduler.rs`）：

| 階段 | 呼叫函數 |
|------|----------|
| H1 | `rx.recv()` → `Result<(Arc<GpuStateVector>, (start, num))>` |
| H2 | `GpuStateVector::copy_sample_range(&self.device, &batch, start, num)` |
| H3 | `sub.to_precision(&self.device, self.precision)` |
| H4 | `state.to_dlpack()` |

**Scheduler 線程**（`scheduler.rs`：`scheduler_loop`）：

| 階段 | 呼叫函數 |
|------|----------|
| S1 | `task_rx.recv()` 或 `select! { recv(task_rx), recv(timeout) }` |
| S2 | 若 buffer 滿或逾時或達 target：`flush_buffer(device, encoder, sample_size, num_qubits, &config.coalescer, &mut buffer)` |
| S3 | **flush_buffer**：`run_coalesced(..., \|batcher\| { for t in tasks { batcher.submit(...) } })`，得到 `(batch, boundaries)`，對每個 task 送 `task.tx.send(Ok((Arc::clone(&batch_arc), (start, num))))`；錯誤時對所有 task 送 `Err(...)` |
| S4 | 單一 task 且 `task_bytes >= target` 且 buffer 空：直接 `run_coalesced` 只送這一個 task，再 `task.tx.send(Ok(...))` |

**run_coalesced** 內會建 coalescer、callback 裡 submit 所有 tasks、最後 **coalescer.flush** → **encode_batch_via_pipeline_from_pinned**（同 3.3）。

---

### 3.5 run_dual_stream_pipeline* — Pipeline 主體（含 memory 使用）

**三種公開入口**（皆在 `pipeline.rs`）：

1. **run_dual_stream_pipeline(device, host_data, kernel_launcher)**
   - `PinnedHostBuffer::new(host_data.len())`，`pinned.as_slice_mut()[..].copy_from_slice(host_data)`
   - `PipelineConfig::get_host_memory_gb()`，`PipelineConfig::get_or_build(device, host_mem_gb)`
   - `run_dual_stream_pipeline_with_chunk_size_from_pinned(device, pinned.as_slice(), chunk_size_elements, pool_size, kernel_launcher)`

2. **run_dual_stream_pipeline_aligned(device, host_data, align_elements, kernel_launcher)**
   - 檢查 `align_elements` 與 `host_data.len() % align_elements == 0`
   - 同上：`PinnedHostBuffer::new` + copy，config，再 `run_dual_stream_pipeline_with_chunk_size_from_pinned`

3. **run_dual_stream_pipeline_aligned_from_pinned(device, pinned_data, align_elements, kernel_launcher)**
   - 不複製，僅檢查對齊與長度
   - `PipelineConfig::get_or_build`，`chunk_size_elements()`、`pinned_pool_size_resolved()`
   - 直接 `run_dual_stream_pipeline_with_chunk_size_from_pinned(device, pinned_data, chunk_size_elements, pool_size, kernel_launcher)`

**run_dual_stream_pipeline_with_chunk_size_from_pinned**（內部實作）— 各階段與函數：

| 階段 | 呼叫函數 / 動作 |
|------|------------------|
| 前置 | `PipelineContext::new(device, pool_size)` → `device.fork_default_stream()` x2（stream_compute, stream_copy），`cudaEventCreateWithFlags` x pool_size（events_copy_done、events_compute_done） |
| 前置 | `num_chunks = (pinned_data.len() + chunk_size_elements - 1) / chunk_size_elements` |
| 前置 | `ensure_device_memory_available(pool_size * chunk_bytes, ...)` |
| 前置 | `device_buffers: Vec<CudaSlice<f64>>`，對每個 slot：`device.alloc::<f64>(chunk_size_elements)` |
| 前置 | 若 `QDP_ENABLE_OVERLAP_TRACKING`：`OverlapTracker::new(pool_size, true)` |
| **Prologue** | 對 `slot in 0..min(pool_size, num_chunks)`：<br>• `chunk_offset_len(pinned_data.len(), chunk_size_elements, slot)`<br>• （可選）`tracker.record_copy_start(&ctx.stream_copy, slot)`<br>• `ctx.async_copy_to_device(pinned_ptr+offset, device_buffers[slot], len)` → `cudaMemcpyAsync(..., stream_copy)`<br>• （可選）`tracker.record_copy_end(...)`<br>• `ctx.record_copy_done(slot)` → `cudaEventRecord(events_copy_done[slot], stream_copy)`<br>• `current_copy_idx = prologue_chunks`，`process_idx = 0` |
| **Chunk 迴圈** | `while process_idx < num_chunks`：<br>• `slot = process_idx % pool_size`，`(offset, len) = chunk_offset_len(..., process_idx)` |
| 迴圈-重複用 device 前 | 若需重用 slot（已滿 pool）：`ctx.wait_for_compute(reuse_slot)` → `cudaStreamWaitEvent(stream_copy, events_compute_done[reuse_slot])` |
| 迴圈-下一塊 H2D | 若還有 chunk 要 copy：<br>• 若 `current_copy_idx >= pool_size`：`ctx.synchronize_copy_event(copy_slot)` → `cudaEventSynchronize(events_copy_done[copy_slot])`（Token Ring：host 等該 slot 上次 H2D 完成才重用 pinned 語意；此處資料已在 pinned_data，實作上為 copy stream 上該 slot 的 event）<br>• `ctx.async_copy_to_device(...)`，`ctx.record_copy_done(copy_slot)`，`current_copy_idx += 1` |
| 迴圈-compute | `ctx.wait_for_copy(slot)` → `cudaStreamWaitEvent(stream_compute, events_copy_done[slot])`<br>• （可選）`tracker.record_compute_start/end(...)`<br>• **kernel_launcher(&ctx.stream_compute, input_ptr, offset, len)**<br>• `ctx.record_compute_done(slot)` → `cudaEventRecord(events_compute_done[slot], stream_compute)`<br>• （可選）每 10 chunk `tracker.log_overlap(...)`<br>• `process_idx += 1` |
| **Epilogue** | `ctx.sync_copy_stream()` → `cudaStreamSynchronize(stream_copy)`<br>• `device.wait_for(&ctx.stream_compute)`<br>• `drop(device_buffers)` |

**Memory 小結（Pipeline）**：

- **Host**：呼叫端提供 `pinned_data: &[f64]`（來自 BatchPool 的 `PinnedHostBuffer` 或 Coalescer 的 `pool.acquire()`）；若從 `run_dual_stream_pipeline` / `run_dual_stream_pipeline_aligned` 進入，則在 pipeline 內建一次 `PinnedHostBuffer` 並複製。
- **Device**：`pool_size` 個 `CudaSlice<f64>`，每個 `chunk_size_elements`；由 `PipelineConfig::get_or_build` 得到 `chunk_size_elements`、`pool_size`。

---

### 3.6 Encoder：encode_batch_via_pipeline / encode_batch_via_pipeline_from_pinned

以 **AmplitudeEncoder**（`encodings/amplitude.rs`）為例。

**encode_batch_via_pipeline(device, batch_data, num_samples, sample_size, num_qubits)**：

| 階段 | 呼叫函數 |
|------|----------|
| E1 | `Preprocessor::validate_batch(batch_data, num_samples, sample_size, num_qubits)` |
| E2 | `GpuStateVector::new_batch(device, num_samples, num_qubits)` |
| E3 | `device.alloc_zeros::<f64>(num_samples)`（inv_norms_gpu） |
| E4 | `run_dual_stream_pipeline_aligned(device, batch_data, sample_size, kernel_launcher)` |
| E5 | kernel_launcher 內每 chunk：<br>• `first_sample = chunk_offset / sample_size`，`num_samples_chunk = chunk_len / sample_size`<br>• `launch_l2_norm_batch(input_ptr, num_samples_chunk, sample_size, inv_norms_chunk_ptr, stream_ptr)`<br>• `launch_amplitude_encode_batch(input_ptr, state_chunk_ptr, inv_norms_chunk_ptr, num_samples_chunk, sample_size, state_len, stream_ptr)` |
| E6 | `device.dtoh_sync_copy(&inv_norms_gpu)`，檢查 `host_inv_norms` 是否 finite 且非 0 |

**encode_batch_via_pipeline_from_pinned(device, pinned_data, num_samples, sample_size, num_qubits)**：

| 階段 | 呼叫函數 |
|------|----------|
| E'1 | `Preprocessor::validate_batch(...)` |
| E'2 | `GpuStateVector::new_batch(device, num_samples, num_qubits)` |
| E'3 | `device.alloc_zeros::<f64>(num_samples)` |
| E'4 | `run_dual_stream_pipeline_aligned_from_pinned(device, pinned_data, sample_size, kernel_launcher)` |
| E'5 | kernel_launcher 同 E5：`launch_l2_norm_batch` + `launch_amplitude_encode_batch` |
| E'6 | 同上：D2H norm 驗證 |

**AngleEncoder** 等若實作 batch pipeline，結構類似（可能使用不同 kernel），最終仍透過同一套 `run_dual_stream_pipeline_*` 與 `encode_batch_via_pipeline_from_pinned`。

---

## 4. 函數呼叫鏈總表（依路徑）

### 4.1 encode() 單一樣本

```
QdpEngine::encode
  → get_encoder
  → encoder.encode(device, data, num_qubits)
      → Preprocessor::validate_input
      → GpuStateVector::new
      → [小資料] device.htod_sync_copy, launch_amplitude_encode, device.synchronize
      → [大資料] Preprocessor::calculate_l2_norm, encode_async_pipeline
          → run_dual_stream_pipeline
              → PinnedHostBuffer::new, copy_from_slice
              → PipelineConfig::get_or_build
              → run_dual_stream_pipeline_with_chunk_size_from_pinned
                  → PipelineContext::new, ensure_device_memory_available, device.alloc (x pool_size)
                  → Prologue: async_copy_to_device, record_copy_done (x prologue_chunks)
                  → Loop: wait_for_compute?, synchronize_copy_event?, async_copy_to_device, record_copy_done, wait_for_copy, kernel_launcher, record_compute_done
                  → sync_copy_stream, device.wait_for(stream_compute)
          → (optional) cudaMemsetAsync padding, device.synchronize
  → state_vector.to_precision
  → state_vector.to_dlpack
```

### 4.2 encode_batch()（經 BatchPool）

```
QdpEngine::encode_batch
  → batch_pool.submit_handle(batch_data.to_vec(), ...)
      → mpsc::channel, submit(BatchJob), BatchHandle{rx}
  → BatchHandle::get
      → rx.recv

worker_loop:
  → job_rx.recv
  → get_encoder
  → PinnedHostBuffer::new, copy_from_slice(job.batch_data)
  → encoder.encode_batch_via_pipeline_from_pinned(device, pinned.as_slice(), ...)
      → Preprocessor::validate_batch, GpuStateVector::new_batch, device.alloc_zeros(num_samples)
      → run_dual_stream_pipeline_aligned_from_pinned
          → PipelineConfig::get_or_build
          → run_dual_stream_pipeline_with_chunk_size_from_pinned (同上 Prologue/Loop/Epilogue)
          → kernel_launcher: launch_l2_norm_batch, launch_amplitude_encode_batch
      → device.dtoh_sync_copy(inv_norms_gpu), validate norms
  → state.to_precision
  → job.response_tx.send(Ok(Arc::new(state)))

BatchHandle::get
  → batch.to_dlpack
```

### 4.3 encode_list()

```
QdpEngine::encode_list
  → get_encoder
  → crate::gpu::encode_list(device, encoder, sample_size, num_qubits, config, slices)
      → run_coalesced(..., |batcher| { for s in slices { batcher.submit(EncodeRequest::Single(s)) } })
          → EncodeCoalescer::new (內建 PinnedBufferPool)
          → callback: coalescer.submit(EncodeRequest::Single(s)) x N
          → coalescer.flush(device, encoder)
              → pool.acquire() → PinnedBufferHandle
              → rayon par_iter: SendPtr::copy_into(off, req.as_slice())
              → encoder.encode_batch_via_pipeline_from_pinned(device, pinned_slice[..total_elements], ...)
                  → (同 4.2 的 encode_batch_via_pipeline_from_pinned 鏈)
              → queue.clear, drop(handle)
      → (batch_state, boundaries)
  → batch_state.to_precision, to_dlpack
```

### 4.4 encode_async() / EncodeHandle::get()

```
QdpEngine::encode_async
  → get_encoder (validate), schedulers.lock, EncodeScheduler::new or get
  → mpsc::channel, EncodeTask{ data: data.to_vec(), ... }, scheduler.submit(task)
  → EncodeHandle::new(rx, device, precision, num_qubits)

scheduler_loop:
  → task_rx.recv / select!(recv task_rx, recv timeout)
  → flush_buffer 或 run_coalesced(單 task)
      → run_coalesced(..., |batcher| { batcher.submit(EncodeRequest::Single/ Batch) })
      → coalescer.flush → encode_batch_via_pipeline_from_pinned (同 4.3)
      → task.tx.send(Ok((batch_arc, (start, num))))

EncodeHandle::get
  → rx.recv
  → GpuStateVector::copy_sample_range(device, batch, start, num)
  → sub.to_precision, state.to_dlpack
```

---

## 5. 檔案與模組對照

| 模組 | 路徑 | 職責 |
|------|------|------|
| Engine / API | `qdp-core/src/lib.rs` | QdpEngine：encode, encode_batch, encode_list, encode_async；BatchPool / Scheduler 持有與呼叫 |
| Pipeline | `qdp-core/src/gpu/pipeline.rs` | PipelineContext，run_dual_stream_pipeline*，雙 stream + event 同步 |
| Memory | `qdp-core/src/gpu/memory.rs` | GpuStateVector，PinnedHostBuffer，ensure_device_memory_available，map_allocation_error |
| Buffer pool | `qdp-core/src/gpu/buffer_pool.rs` | PinnedBufferPool，PinnedBufferHandle（Coalescer 用） |
| Coalescer | `qdp-core/src/gpu/coalescer.rs` | EncodeRequest，EncodeCoalescer，flush，run_coalesced，encode_list |
| Scheduler | `qdp-core/src/gpu/scheduler.rs` | EncodeTask，EncodeScheduler，scheduler_loop，flush_buffer，EncodeHandle |
| Batch pool | `qdp-core/src/gpu/batch_pool.rs` | BatchJob，BatchPool，worker_loop，BatchHandle |
| Pipeline config | `qdp-core/src/gpu/pipeline_config.rs` | PipelineConfig，chunk_size，pinned_pool_size，get_or_build |
| Encodings | `qdp-core/src/gpu/encodings/amplitude.rs`（及 angle 等） | encode，encode_batch，encode_batch_via_pipeline，encode_batch_via_pipeline_from_pinned，kernel_launcher |
| Encodings trait | `qdp-core/src/gpu/encodings/mod.rs` | QuantumEncoder trait，get_encoder |

---

## 6. 從 run_pipeline_baseline.py 出發的 Dataflow

本節描述 **baseline 效能腳本** `qdp/qdp-python/benchmark/run_pipeline_baseline.py` 的入口與三種模式（一般 / coalesced / stream）下，從 Python 到 Rust 的完整 dataflow。

**腳本入口**：`main()` 解析參數（`--qubits`, `--batch-size`, `--prefetch`, `--batches`, `--trials`, `--encoding-method`, `--use-coalesced`, `--use-stream`, `--in-flight`），依 `--skip-throughput` / `--skip-latency` 決定是否跑 throughput 與 latency；throughput 依 `--use-stream` / `--use-coalesced` 選擇 `run_mahout_throughput_stream`、`run_mahout_throughput_coalesced` 或 `run_mahout_throughput`；latency 依 `--use-coalesced` 選擇 `run_mahout_latency_coalesced` 或 `run_mahout_latency`。

**共用前置**（throughput / latency 皆同）：

- **資料來源**：`benchmark_throughput.py` / `benchmark_latency.py` 內呼叫 `prefetched_batches(total_batches, batch_size, vector_len, prefetch, encoding_method)`（`utils.prefetched_batches`），產生 2D NumPy batch 的 iterator；每批為 `(batch_size, vector_len)`，`vector_len = num_qubits`（angle）或 `1 << num_qubits`（amplitude/basis）。
- **正規化**：每批會做 `normalize_batch(batch, encoding_method)`（`utils.normalize_batch`），再 `np.ascontiguousarray(..., dtype=np.float64)` 得到 C-contiguous float64。
- **Engine**：`QdpEngine(0)` 來自 `qumat_qdp`（PyO3 綁定到 `qdp-core` 的 `QdpEngine`）。

以下分三種 **throughput 路徑** 與兩種 **latency 路徑** 列出 dataflow。

### 6.1 Throughput：一般路徑（per-batch encode）

**Python 呼叫鏈**（`benchmark_throughput.py` → `run_mahout`）：

| 階段 | 檔案 | 函數 / 動作 |
|------|------|-------------|
| 1 | `run_pipeline_baseline.py` | `run_throughput = run_mahout_throughput`，`run_throughput(qubits, batches, batch_size, prefetch, encoding_method)` |
| 2 | `benchmark_throughput.py` | `run_mahout(...)`：`engine = QdpEngine(0)`，`torch.cuda.synchronize()`，`start = time.perf_counter()` |
| 3 | `benchmark_throughput.py` | `for batch in prefetched_batches(...)`：每批 `normalized = np.ascontiguousarray(normalize_batch(batch, encoding_method), dtype=np.float64)` |
| 4 | `benchmark_throughput.py` | `qtensor = engine.encode(normalized, num_qubits, encoding_method)` |
| 5 | `benchmark_throughput.py` | `tensor = torch.from_dlpack(qtensor).abs().to(torch.float32)`，`processed += normalized.shape[0]` |
| 6 | `benchmark_throughput.py` | 迴圈結束後 `torch.cuda.synchronize()`，`duration = time.perf_counter() - start`，throughput = processed / duration |

**FFI**：`engine.encode(normalized, num_qubits, encoding_method)` 進入 PyO3 綁定（`qdp-python/src/lib.rs`）：依 `data` 型別分派（2D NumPy → `encode` 的 2D 路徑）。2D 時會將 array 轉成 `Vec<f64>`（或 slice），再呼叫 `self.engine.encode(data_slice, num_qubits, encoding_method)`（即 `qdp-core` 的 `QdpEngine::encode`）。回傳為 DLPack 指標，Python 端用 `torch.from_dlpack(qtensor)` 取得 Tensor。

**對應 Rust 路徑**：即 **§3.1 encode() 單一樣本**；若單次傳入為整批 2D，綁定層可能走 `encode_batch` 或逐列 encode，依目前綁定實作為準；若為逐列則每列一次 `QdpEngine::encode`。

### 6.2 Throughput：Coalesced 路徑（encode_list）

**Python 呼叫鏈**（`benchmark_throughput.py` → `run_mahout_coalesced`）：

| 階段 | 檔案 | 函數 / 動作 |
|------|------|-------------|
| 1 | `run_pipeline_baseline.py` | `run_throughput = run_mahout_throughput_coalesced`，`run_throughput(...)` |
| 2 | `benchmark_throughput.py` | `run_mahout_coalesced(...)`：`engine = QdpEngine(0)`，檢查 `hasattr(engine, "encode_list")` |
| 3 | `benchmark_throughput.py` | `chunk_samples = max(1, (64*1024*1024) // bytes_per_sample)`（≤ 64 MB 一 chunk），`chunk: list[np.ndarray] = []` |
| 4 | `benchmark_throughput.py` | `for batch in prefetched_batches(...)`：每樣本 `chunk.append(normalized[i:i+1].reshape(-1).copy())`，若 `len(chunk) >= chunk_samples` 則 `_qt, _bounds = engine.encode_list(chunk, vector_len, num_qubits, encoding_method)`，`torch.from_dlpack(_qt)`，`chunk = []` |
| 5 | `benchmark_throughput.py` | 若最後 `chunk` 非空，再呼叫一次 `engine.encode_list(chunk, ...)` |
| 6 | `benchmark_throughput.py` | `torch.cuda.synchronize()`，計算 throughput |

**FFI**：`engine.encode_list(chunk, vector_len, num_qubits, encoding_method)` 進入 PyO3（`qdp-python/src/lib.rs` 的 `encode_list`）：`chunk` 為 list of 1D NumPy arrays，轉成 `Vec<PyReadonlyArray1<f64>>` 與 `&[&[f64]]` refs，呼叫 `self.engine.encode_list(&refs, sample_size, num_qubits, encoding_method)`，回傳 `(dlpack_ptr, boundaries)`。

**對應 Rust 路徑**：即 **§3.3 encode_list()** → `run_coalesced` → `coalescer.submit` 多筆 → `flush` → `encode_batch_via_pipeline_from_pinned` → pipeline。

### 6.3 Throughput：Stream 路徑（encode_stream → encode_batch_submit）

**Python 呼叫鏈**（`benchmark_throughput.py` → `run_mahout_stream`，`qumat_qdp/stream.py`）：

| 階段 | 檔案 | 函數 / 動作 |
|------|------|-------------|
| 1 | `run_pipeline_baseline.py` | `run_throughput = run_mahout_throughput_stream`，`run_throughput(..., in_flight=args.in_flight)` |
| 2 | `benchmark_throughput.py` | `run_mahout_stream(...)`：`engine = QdpEngine(0)`，warmup：`prefetched_batches(2, ...)` → `encode_stream(engine, warmup_iter, num_qubits, encoding_method, in_flight=in_flight)`，消費 yield 的 qtensor |
| 3 | `benchmark_throughput.py` | 正式計時：`batches_iter = prefetched_batches(...)`，`for qtensor in encode_stream(engine, batches_iter, num_qubits, encoding_method, **stream_kw)`，`tensor = torch.from_dlpack(qtensor).abs().to(torch.float32)`，`processed += tensor.shape[0]` |
| 4 | `qumat_qdp/stream.py` | `encode_stream(...)`：若有 `engine.encode_batch_submit` 則走 `_encode_stream_pool`，否則單一 worker 線程 + queue |
| 5 | `qumat_qdp/stream.py` | **Pool 路徑** `_encode_stream_pool`：`for batch in batch_iterator`：`normalized = np.ascontiguousarray(_normalize_batch(batch, encoding_method), dtype=np.float64)`；若 `len(handles) - next_yield >= in_flight` 則 `qt = handles[next_yield].get()`、`yield qt`、`next_yield += 1`；`handles.append(submit_fn(normalized, num_qubits, encoding_method))`（`submit_fn` = `engine.encode_batch_submit`） |
| 6 | `qumat_qdp/stream.py` | 迭代結束後 `for i in range(next_yield, len(handles)): yield handles[i].get()` |

**FFI（現狀）**：`engine.encode_batch_submit(normalized, num_qubits, encoding_method)` 進入 PyO3（`qdp-python/src/lib.rs`）：2D NumPy 轉成 contiguous float64；**零拷貝路徑**：傳 slice 的 ptr + len（`BatchJobData::Borrowed`），**無 `to_vec()`**；呼叫 `self.engine.encode_batch_submit(job_data, ...)`，回傳 Python 端的 handle 物件（對應 Rust `BatchHandle`）；handle 的 `.get()` 在 Python 側會呼叫綁定的 `get()`，內部對應 `BatchHandle::get()`（`rx.recv()` 取 `Arc<GpuStateVector>`，再 `to_dlpack()`）。詳見 **§7.4** 現狀 dataflow。

**對應 Rust 路徑**：即 **§3.2 encode_batch()（經 BatchPool）**：`encode_batch_submit` → `BatchPool::submit_handle`（單一 master、有界佇列）→ worker 內 `RegisteredHostBuffer`（Borrowed）或 PinnedHostBuffer+copy（Owned）→ `encode_batch_via_pipeline_from_pinned` → pipeline；`handle.get()` → `BatchHandle::get()` → `rx.recv()` → `batch.to_dlpack()`。

### 6.4 Latency：一般路徑（per-batch encode）

**Python 呼叫鏈**（`benchmark_latency.py` → `run_mahout`）：

| 階段 | 檔案 | 函數 / 動作 |
|------|------|-------------|
| 1 | `run_pipeline_baseline.py` | `run_latency = run_mahout_latency`（無 `--use-coalesced` 時），`run_latency(...)` |
| 2 | `benchmark_latency.py` | `run_mahout(...)`：`engine = QdpEngine(0)`，`sync_cuda()`，`start = time.perf_counter()` |
| 3 | `benchmark_latency.py` | `for batch in prefetched_batches(...)`：`normalized = normalize_batch(batch, encoding_method)`，`qtensor = engine.encode(normalized, num_qubits, encoding_method)`，`_ = torch.utils.dlpack.from_dlpack(qtensor)`，`processed += normalized.shape[0]` |
| 4 | `benchmark_latency.py` | `sync_cuda()`，`latency_ms = (duration / processed) * 1000` |

**FFI / Rust**：與 **§6.1** 相同，每批一次 `engine.encode` → `QdpEngine::encode`（或綁定層的 2D 分派）。

### 6.5 Latency：Coalesced 路徑（encode_list）

**Python 呼叫鏈**（`benchmark_latency.py` → `run_mahout_coalesced`）：

| 階段 | 檔案 | 函數 / 動作 |
|------|------|-------------|
| 1 | `run_pipeline_baseline.py` | `run_latency = run_mahout_latency_coalesced`（`--use-coalesced` 時），`run_latency(...)` |
| 2 | `benchmark_latency.py` | `run_mahout_coalesced(...)`：`chunk_samples = max(1, (64*1024*1024) // bytes_per_sample)`，`chunk: list = []` |
| 3 | `benchmark_latency.py` | `for batch in prefetched_batches(...)`：每樣本 append 到 chunk，若 `len(chunk) >= chunk_samples` 則 `_qt, _bounds = engine.encode_list(chunk, vector_len, num_qubits, encoding_method)`，`_ = torch.utils.dlpack.from_dlpack(_qt)`，`chunk = []`；最後若 `chunk` 非空再呼叫一次 `encode_list` |
| 4 | `benchmark_latency.py` | `sync_cuda()`，`latency_ms = (duration / processed) * 1000` |

**FFI / Rust**：與 **§6.2** 相同，即 **§3.3 encode_list()** → run_coalesced → flush → encode_batch_via_pipeline_from_pinned → pipeline。

### 6.6 run_pipeline_baseline.py 小結

| 模式 | 腳本選擇 | Python 入口 | 主要 FFI | 對應 Rust 路徑 |
|------|----------|-------------|----------|----------------|
| Throughput 一般 | 預設（無 `--use-stream` / `--use-coalesced`） | `run_mahout_throughput` | `engine.encode(normalized, ...)` 每批 | §3.1 encode() |
| Throughput coalesced | `--use-coalesced` | `run_mahout_throughput_coalesced` | `engine.encode_list(chunk, ...)` 每 chunk | §3.3 encode_list() |
| Throughput stream | `--use-stream` | `run_mahout_throughput_stream` | `encode_stream` → `engine.encode_batch_submit(...)`，`handle.get()` | §3.2 BatchPool + encode_batch_submit |
| Latency 一般 | 預設 | `run_mahout_latency` | `engine.encode(normalized, ...)` 每批 | §3.1 encode() |
| Latency coalesced | `--use-coalesced` | `run_mahout_latency_coalesced` | `engine.encode_list(chunk, ...)` | §3.3 encode_list() |

**環境變數**（腳本內設定）：`QDP_ENABLE_POOL_METRICS=1`、`QDP_ENABLE_OVERLAP_TRACKING=1`、`RUST_LOG=info`，在 import Rust 模組前即設定，故 pipeline 與 pool 可讀取。

---

## 7. After：從 run_pipeline_baseline.py 出發的 Dataflow（專案目前狀態）

本節以 **現狀** 為準，詳細描述 **目前專案**（Phase 2 / Part N–O 已實施，見 [QDP_OPTIMIZATION_PLAN_EN.md](QDP_OPTIMIZATION_PLAN_EN.md) Implementation Status）從 `run_pipeline_baseline.py` 出發的 dataflow：每個模式下的 Python → FFI → Rust 各階段、型別與函數名稱、記憶體與複製點、pool / coalescer / pipeline 的實際行為。§6 為同一腳本之入口與模式對照；本節補足 **現有實作** 的完整 dataflow 與 memory 使用。**專案剛完成 Phase 2 變更**（零拷貝 Borrowed、單一 master、fused kernel、sync_stream_via_query、可選 thread_affinity），以下即該狀態之詳細紀錄。

---

### 7.1 現狀總覽：run_pipeline_baseline.py 三種模式對應的底層路徑

| 模式 | Python 入口（§6） | 現狀 FFI | 現狀 Rust 路徑 | 現狀複製點（次數） |
|------|-------------------|----------|----------------|--------------------|
| **Throughput 一般** | `run_mahout_throughput` | 每批 `engine.encode(normalized, ...)` | 綁定層依 2D/1D 分派 → `QdpEngine::encode` 或 `encode_batch`；encode 路徑可能 `run_dual_stream_pipeline`（內部建 PinnedHostBuffer + 一次 host 複製）或 sync 路徑；**encode_batch** 現狀為 `submit_handle(BatchJobData::Borrowed { ptr, len }).get()`，無 to_vec | Host：normalized 已 contiguous；若走 batch 則 **Borrowed**（0 次）；若走 encode pipeline 則 pipeline 內 1 次 host→pinned，再 async H2D |
| **Throughput coalesced** | `run_mahout_throughput_coalesced` | 每 chunk `engine.encode_list(chunk, ...)` | `QdpEngine::encode_list` → `run_coalesced` → coalescer 累積 `EncodeRequest::Single` → `flush`：pool.acquire()，rayon 並行複製到 pinned，`encode_batch_via_pipeline_from_pinned` | Host：flush 時 **1 次** 並行複製（各 request 寫入 pinned 一塊）；佇列 Zero-Copy Ingress；之後 async H2D only |
| **Throughput stream** | `run_mahout_throughput_stream` | `encode_stream` → 每批 `engine.encode_batch_submit(...)`，`handle.get()` | 綁定層 `array_2d.as_slice()` **無 to_vec**，傳 `&[f64]` → `engine.encode_batch_submit(data_slice, ...)` → Rust 建 `BatchJobData::Borrowed { ptr, len }`；**單一 master** `worker_loop`：`RegisteredHostBuffer::new(ptr, len)`（cudaHostRegister）→ `encode_batch_via_pipeline_from_pinned`；有界佇列 `QDP_BATCH_QUEUE_CAPACITY`（預設 4） | Host：**0 次**（Borrowed：cudaHostRegister 原地 pin）；Owned 路徑仍 1 次（worker copy→PinnedHostBuffer）；async H2D only |
| **Latency 一般** | `run_mahout_latency` | 每批 `engine.encode(normalized, ...)` | 同 Throughput 一般 | 同 Throughput 一般 |
| **Latency coalesced** | `run_mahout_latency_coalesced` | `engine.encode_list(chunk, ...)` | 同 Throughput coalesced | 同 Throughput coalesced |

---

### 7.2 現狀：Throughput 一般路徑（per-batch encode）— 逐步 dataflow

1. **Python**（`benchmark_throughput.py`）：`prefetched_batches(...)` 產出 2D `(batch_size, vector_len)`；每批 `normalized = np.ascontiguousarray(normalize_batch(batch, encoding_method), dtype=np.float64)`，`qtensor = engine.encode(normalized, num_qubits, encoding_method)`。
2. **FFI**（`qdp-python/src/lib.rs`）：`encode(data, num_qubits, encoding_method)` 依 `data` 型別分派；2D NumPy 時可能走 `encode`（傳 slice）或 `encode_batch`（傳 slice）。若走 **encode_batch**：綁定層取得 2D array 的 slice，呼叫 `self.engine.encode_batch(data_slice, num_samples, sample_size, num_qubits, encoding_method)`（**無 to_vec**），再 `handle.get()`。
3. **Rust**（`qdp-core/src/lib.rs`）：**encode_batch 現狀**：`encode_batch(batch_data: &[f64], ...)` → `batch_pool.submit_handle(BatchJobData::Borrowed { ptr: batch_data.as_ptr(), len: batch_data.len() }, ...).get()`，即與 stream 路徑相同之 Borrowed + 單一 master；**encode 現狀**：`QdpEngine::encode` → `encoder.encode(device, data, num_qubits)`；Amplitude 小資料 sync（htod_sync_copy + 單 kernel + synchronize），大資料 `encode_async_pipeline` → `run_dual_stream_pipeline`（內部 PinnedHostBuffer + copy_from_slice，再 `run_dual_stream_pipeline_with_chunk_size_from_pinned`）。
4. **Memory 現狀**：若走 **encode_batch**（2D）則與 §7.4 stream 之 Borrowed 路徑相同（0 次 host 複製、單一 master、RegisteredHostBuffer）。若走 `run_dual_stream_pipeline`，host 端 1 次 host_data → PinnedHostBuffer；device 端為 `PipelineConfig` 的 pool_size 個 `CudaSlice<f64>` chunk buffer + `GpuStateVector`；pipeline 為 dual stream、event Token Ring、Epilogue 使用 `sync_stream_via_query`。

---

### 7.3 現狀：Throughput coalesced 路徑（encode_list）— 逐步 dataflow

1. **Python**（`benchmark_throughput.py`）：累積 `chunk`（list of 1D arrays），達 `chunk_samples` 或結尾時 `_qt, _bounds = engine.encode_list(chunk, vector_len, num_qubits, encoding_method)`。
2. **FFI**（`qdp-python/src/lib.rs`）：`encode_list` 將 list 轉成 `Vec<PyReadonlyArray1<f64>>` 與 `&[&[f64]]` refs，呼叫 `self.engine.encode_list(&refs, sample_size, num_qubits, encoding_method)`，回傳 `(dlpack_ptr, boundaries)`。
3. **Rust**（`qdp-core`）：`QdpEngine::encode_list` → `crate::gpu::encode_list` → `run_coalesced(device, encoder, sample_size, num_qubits, config, callback)`；callback 內對每個 slice `batcher.submit(EncodeRequest::Single(s))`（僅存引用）；`run_coalesced` 最後 `coalescer.flush(device, encoder)`。
4. **flush 現狀**（`coalescer.rs`）：`pool.acquire()` 取 `PinnedBufferHandle`；依 request 邊界算 `(offset, len)`；rayon `par_iter()` 並行 `SendPtr::copy_into(off, req.as_slice())`；`encoder.encode_batch_via_pipeline_from_pinned(device, &pinned_slice[..total_elements], total_samples, sample_size, num_qubits)`；drop(handle) 歸還 buffer。
5. **Memory 現狀**：佇列僅存 `EncodeRequest::Single(&[f64])`（Zero-Copy Ingress）；複製 **1 次** — flush 時並行寫入 PinnedBufferPool 一塊；pipeline `run_dual_stream_pipeline_aligned_from_pinned`，async H2D only；device 端 pool_size 個 chunk buffer + GpuStateVector batch；amplitude 使用 `launch_amplitude_encode_batch_fused`，Epilogue 使用 `sync_stream_via_query`。

---

### 7.4 現狀：Throughput stream 路徑（encode_stream → encode_batch_submit）— 逐步 dataflow（最詳）

1. **Python**（`qumat_qdp/stream.py`）：`encode_stream` 若存在 `engine.encode_batch_submit` 則走 `_encode_stream_pool`；每批 `normalized = np.ascontiguousarray(_normalize_batch(batch, encoding_method), dtype=np.float64)`，`handles.append(submit_fn(normalized, num_qubits, encoding_method))`，之後 `yield handles[i].get()`。**契約**：caller 必須在 `handle.get()` 回傳前保持 `normalized` 的 buffer 有效（Rust 端以 Borrowed 使用該記憶體）。
2. **FFI**（`qdp-python/src/lib.rs`，`encode_batch_submit`）：取得 `PyReadonlyArray2<f64>`，`data_slice = array_2d.as_slice()`（**無 to_vec**）；呼叫 `self.engine.encode_batch_submit(data_slice, num_samples, sample_size, num_qubits, encoding_method)`，回傳 Python 端 `BatchHandle`（對應 Rust `BatchHandle`）。目前綁定**一律**傳 slice，故 Rust 端為 **Borrowed** 路徑。
3. **Rust Engine**（`qdp-core/src/lib.rs`）：`encode_batch_submit(batch_data: &[f64], ...)` → `batch_pool.submit_handle(BatchJobData::Borrowed { ptr: batch_data.as_ptr(), len: batch_data.len() }, num_samples, sample_size, num_qubits, encoding_method.to_string())`。
4. **BatchPool**（`qdp-core/src/gpu/batch_pool.rs`）：`submit_handle(batch_input: BatchJobData, ...)` 組 `BatchJob { data: batch_input, num_samples, sample_size, num_qubits, encoding_method, response_tx }`，`job_tx.send(job)`，回傳 `BatchHandle { rx }`。Channel 為 **crossbeam_channel::bounded(cap)**，cap 來自 `BatchPool::queue_capacity()`：env **`QDP_BATCH_QUEUE_CAPACITY`**（預設 4），clamp 2..=8。
5. **Worker 現狀（單一 master）**（`batch_pool.rs`）：**僅一條** `worker_loop` 線程；開頭 `set_master_thread_affinity_if_requested()`（env **`QDP_MASTER_CPU_ID`**，僅在 `feature = "thread_affinity"` 且 Linux 時生效）；`job_rx.recv()` 取 `BatchJob`；`match &job.data`：
   - **`BatchJobData::Borrowed { ptr, len }`**：`RegisteredHostBuffer::new(ptr, len)`（`gpu/memory.rs`：**cudaHostRegister** 原地鎖頁，**無 host 複製**）→ `reg.as_slice()` 取得 `&[f64]` → `encoder.encode_batch_via_pipeline_from_pinned(device, reg.as_slice(), ...)`；`RegisteredHostBuffer` Drop 時 **cudaHostUnregister**。
   - **`BatchJobData::Owned(v)`**：`PinnedHostBuffer::new(v.len())`，`pinned.as_slice_mut()[..v.len()].copy_from_slice(v)`（1 次 host 複製）→ `encode_batch_via_pipeline_from_pinned(device, pinned.as_slice(), ...)`。
   接著 `state.to_precision(device, precision)`，`job.response_tx.send(Ok(Arc::new(state)))`。
6. **Pipeline 現狀**（amplitude）：`encode_batch_via_pipeline_from_pinned` → `run_dual_stream_pipeline_aligned_from_pinned`（來源為 `reg.as_slice()` 或 `pinned.as_slice()`，已為可 DMA 之 host 記憶體），**async H2D only**。每 chunk 呼叫 **單一 kernel**：`launch_amplitude_encode_batch_fused`（`qdp-kernels`：L2 norm + amplitude encode 合併，見 `amplitude.cu`）；最後 D2H 拷貝 inv_norms 做驗證。Epilogue 使用 **sync_stream_via_query**（`pipeline.rs`：`cudaStreamQuery` 迴圈 + `std::thread::yield_now()`，無 blocking `cudaStreamSynchronize`）。
7. **Memory 現狀**：Host — Borrowed：佇列僅傳 `BatchJobData::Borrowed { ptr, len }`（descriptor），worker 端 `RegisteredHostBuffer` 原地 pin，**0 次 host 複製**；Owned：`Vec<f64>` 經 channel，worker 內 1 次複製到 `PinnedHostBuffer`。Device — pipeline 的 pool_size 個 `CudaSlice<f64>` chunk buffer + `GpuStateVector::new_batch` + inv_norms_gpu。Pool — **單一 master 線程**、有界佇列（`QDP_BATCH_QUEUE_CAPACITY` 預設 4）、`BatchJob` 含 `BatchJobData::Borrowed` 或 `Owned`。

---

### 7.5 現狀：Latency 一般 / coalesced

- **Latency 一般**：與 §7.2 相同，每批 `engine.encode(normalized, ...)`；計時為 `(duration / processed) * 1000` ms/vector。
- **Latency coalesced**：與 §7.3 相同，`engine.encode_list(chunk, ...)`，run_coalesced → flush → encode_batch_via_pipeline_from_pinned；計時方式同。

---

### 7.6 現狀：Memory 與複製彙總

| 路徑 | Host 複製（現狀） | Device | Pool / Coalescer |
|------|-------------------|--------|-------------------|
| **encode（一般）** | 若走 **encode_batch**（2D）：**0 次**（Borrowed + RegisteredHostBuffer）。若走 encode 大資料 pipeline：1 次（run_dual_stream_pipeline 內 host_data→PinnedHostBuffer）；sync 路徑則 htod_sync_copy + 單 kernel | GpuStateVector + pipeline chunk buffers；Epilogue sync_stream_via_query | encode_batch 經 BatchPool（單一 master、Borrowed） |
| **encode_list（coalesced）** | **1 次** — flush 時並行寫入 PinnedBufferPool 一塊 | from_pinned → async H2D；launch_amplitude_encode_batch_fused；sync_stream_via_query | Coalescer 持 PinnedBufferPool（預設 2 塊）；flush 時 acquire → 並行寫入 → encode_batch_via_pipeline_from_pinned → drop 歸還 |
| **encode_batch_submit（stream）** | **Borrowed：0 次**（RegisteredHostBuffer + cudaHostRegister）；**Owned：1 次**（worker 內 copy→PinnedHostBuffer） | 同左；from_pinned → async H2D；launch_amplitude_encode_batch_fused；sync_stream_via_query | BatchPool：**單一 master**（`worker_loop`），有界 channel **QDP_BATCH_QUEUE_CAPACITY**（預設 4，2..=8）；`BatchJob { data: BatchJobData::Borrowed { ptr, len } 或 Owned(Vec<f64>) }` |

---

### 7.7 現狀：Pipeline 與 Kernel（Phase 2 已實施）

- **Pipeline**（`qdp-core/src/gpu/pipeline.rs`）：`run_dual_stream_pipeline_with_chunk_size_from_pinned`；雙 stream（`stream_copy` / `stream_compute`）；pool_size 個 device chunk buffer（`PipelineConfig`）；Prologue 填滿前 pool_size 個 chunk 的 H2D；迴圈中 Token Ring：`wait_for_compute`、`synchronize_copy_event`、`async_copy_to_device`、`record_copy_done`、`wait_for_copy`、`kernel_launcher`、`record_compute_done`；Epilogue 為 **sync_stream_via_query(stream_copy)** 與 **sync_stream_via_query(stream_compute)**（`cudaStreamQuery` 迴圈 + `yield_now`，無 `cudaStreamSynchronize`）。
- **Kernel（現狀）**：Amplitude batch 路徑每 chunk **單一 launch** — **`launch_amplitude_encode_batch_fused`**（L2 norm + encode 合併），見 `qdp-kernels` 與 `qdp-core/src/gpu/encodings/amplitude.rs`。

---

### 7.8 現狀：型別、環境變數與可選功能（Phase 2 對應）

| 項目 | 現狀實作 | 檔案 / 備註 |
|------|----------|-------------|
| **BatchJobData** | `enum { Owned(Vec<f64>), Borrowed { ptr, len } }` | `qdp-core/src/gpu/batch_pool.rs`；佇列傳 descriptor，Borrowed 時無 Vec 複製 |
| **RegisteredHostBuffer** | `cudaHostRegister(ptr, len)`，`as_slice()`，Drop → `cudaHostUnregister` | `qdp-core/src/gpu/memory.rs`；Borrowed 路徑 worker 使用，0 次 host 複製 |
| **QDP_BATCH_QUEUE_CAPACITY** | 有界佇列容量，預設 4，clamp 2..=8 | `batch_pool.rs`：`BatchPool::queue_capacity()` |
| **QDP_MASTER_CPU_ID** | 可選：將 master 線程綁定至指定 CPU core | `batch_pool.rs`：`set_master_thread_affinity_if_requested()`；需 **feature `thread_affinity`** 且 Linux |
| **sync_stream_via_query** | Epilogue 以 `cudaStreamQuery` 迴圈 + `yield_now` 取代 `cudaStreamSynchronize` | `pipeline.rs`；減少 blocking 等待 |
| **launch_amplitude_encode_batch_fused** | 每 chunk 單一 kernel（L2 + encode） | `amplitude.rs`、`qdp-kernels` |

以上即為 **專案目前狀態**（Phase 2 / Part N–O 已實施，見 QDP_OPTIMIZATION_PLAN_EN.md Implementation Status）下，從 `run_pipeline_baseline.py` 出發的完整 dataflow 與 memory 使用；§9 對應 Part M/N 對照與 Part O 驗證。

---

## 8. 小結

- **單一 pipeline**：所有 batch 編碼最終都經 `encode_batch_via_pipeline` 或 `encode_batch_via_pipeline_from_pinned`，進入 `run_dual_stream_pipeline_*`，使用雙 stream（copy / compute）與 pool_size 個 device chunk buffer，以 event 做 Token Ring 同步。
- **Memory**：Host 使用 PinnedHostBuffer（或 PinnedBufferPool 的 handle）；Device 使用 GpuStateVector（結果）與 pipeline 的 `Vec<CudaSlice<f64>>`（chunk 暫存）；配置由 PipelineConfig（chunk_size_mb、pinned_pool_size）與 CoalescerConfig / SchedulerConfig 決定。
- **複製次數**：BatchPool **stream** 路徑 — Borrowed 時 **0 次** host 複製（cudaHostRegister）；Owned 時 1 次 Vec→Pinned（worker 內），再 async H2D。encode_list / encode_async 經 coalescer flush 為一次並行複製到 Pinned + 一次 encode_batch_via_pipeline_from_pinned（async H2D only）。

本文件可與 `QDP_OPTIMIZATION_PLAN_EN.md`、`REQUEST_COALESCING_REFERENCE_AND_DESIGN.md` 對照，用於稽核實作或新增路徑時保持單一 dataflow 與正確 memory 使用。

---

## 9. Part M / N 對應：修改既有 default（不新增分支）

以下皆為**修改既有 default 路徑**，同一 API、同一入口；不新增「fast path」或 feature flag。優化往底層做（FFI → pipeline → pool → kernel）。

| Part M/N 策略 | 對應現有 dataflow / memory | 變更方向（改既有路徑） |
|---------------|----------------------------|------------------------|
| **M.2 零拷貝 (cudaHostRegister)** | §2.1 BatchJob `batch_data: Vec<f64>`；§3.2 W3–W4 worker 內 `PinnedHostBuffer::new` + `copy_from_slice` | 當呼叫端提供 contiguous buffer（如 PyBuffer）時，在**同一** encode_batch 路徑內改為 `cudaHostRegister` + 傳 ptr 進 **既有** from_pinned；owned 時仍 copy→pinned→from_pinned（同一函數，無第二分支）。 |
| **M.3 單線程 Master** | §1 BatchPool 多 worker；§3.2 `worker_loop` 多線程競爭同一 GPU | **替換** BatchPool 實作為單一 GPU Master 線程 + 有界佇列（descriptor：ptr, len, response_tx）；**同一** API（encode_batch / encode_batch_submit）；僅內部實作改為 one master + triple buffer。 |
| **N.2 不搬 Vec<f64>** | §3.2 `submit_handle(batch_data.to_vec(), ...)`；BatchJob `batch_data: Vec<f64>` | 佇列改傳 descriptor（ptr, len, tx）；當 caller 提供 buffer 時不 to_vec；owned 路徑仍可 copy 一次進 pool-owned pinned 再傳 descriptor。同一 submit_handle API。 |
| **N.5 Kernel Fusion** | §3.5 / §3.6 pipeline 內 `L2 norm` + `encode` 兩次 kernel | **替換** encoder 內兩次 launch 為**同一** pipeline 內單一 fused kernel；同一 kernel_launcher 介面，僅 kernel 實作改變。 |

上述變更後之 dataflow 與 memory 使用應在本文件中另立章節或更新 §1–§3，以保持與計劃一致。
