# Request Coalescing 設計符合性檢查清單

對照 **REQUEST_COALESCING_REFERENCE_AND_DESIGN.md** 的實作檢查與依賴版本確認。
最後更新：設計 9 項均已開發完成；Dataflow 三路徑對照與 §四 整體邏輯順序已記錄；Rust 實作品質（Edition 2024／rust-development）已檢查並記錄於 §六；§3.11.6／§3.11.7 盲區與地雷已核對。

---

## 〇、設計各部分開發完成度（總結優化清單 §四／§六）

| # | 項目 | 設計要求 | 實作狀態 | 備註 |
|---|------|----------|----------|------|
| 1 | **Zero-Copy Coalescer** | EncodeRequest<'a> 持 `&[f64]`，Coalescer 只存 metadata；flush 從 Pinned Pool 取塊、並行 memcpy、run_dual_stream_pipeline | ✅ 完成 | coalescer.rs：queue 為 `Vec<EncodeRequest<'a>>`；flush 內 pool.acquire、par_iter 按邊界寫入、encoder.encode_batch_via_pipeline |
| 2 | **Scope-based API** | run_coalesced(callback)；Python `with engine.coalesce() as batcher:` | ✅ 完成 | **Rust**：`run_coalesced(device, encoder, ..., \|batcher\| { ... })` 已實作。**Python**：`engine.coalesce(sample_size, num_qubits, encoding_method)` 回傳 `CoalesceBatcher`；`with` 內 `batcher.submit(array)` / `batcher.submit_batch(...)`；離開時 `__exit__` 呼叫 `encode_list` 觸發 flush；`batcher.get_result()` 取得 (QuantumTensor, boundaries)。 |
| 3 | **Scheduler Thread** | Time + Size 雙門檻、獨立 thread、Task(tx)、flush 時 Pinned Buffer；Python 端優先 | ✅ 完成 | scheduler.rs：EncodeScheduler 獨立 thread、crossbeam select! + after()、flush_buffer → run_coalesced；bounded channel Backpressure |
| 4 | **統一 API 入口** | encode / encode_batch 經同一 Coalescer 或 Scheduler | ✅ 完成 | encode_list 走 Coalescer；Scheduler 收 EncodeTask 後 run_coalesced 合併；單次 encode 可經 Scheduler submit |
| 5 | **Pass-through** | 單筆 ≥ target_batch_size 且 buffer 空時直接執行 | ✅ 完成 | Scheduler 邏輯：單筆 ≥ target 且 buffer 空時 Pass-through（不進 buffer、直接 run_coalesced 該筆） |
| 6 | **Python Async** | encode_async + 延後 get，讓 Dynamic Batching 生效 | ✅ 完成 | qdp-core：`QdpEngine::encode_async` 送 Task 進 Scheduler、回傳 `EncodeHandle`；`EncodeHandle::get()` 阻塞取 DLPack。qdp-python：`engine.encode_async(data, num_qubits, encoding_method)` → `EncodeHandle`；`handle.get()` → QuantumTensor。§3.9 射後不理 + 延後取結果已實作。 |
| 7 | **encode_list** | 一次 FFI 收多個 slice、回傳 Vec<Result>／boundaries | ✅ 完成 | qdp-core `gpu::encode_list` + `QdpEngine::encode_list`；qdp-python `engine.encode_list(list_of_arrays, ...)` → (QuantumTensor, boundaries) |
| 8 | **Float32 預設存在** | 型別與介面 precision-aware；首版可僅 f64 | ✅ 符合 | 目前 EncodeRequest / Pool / Pipeline 為 f64；Engine 有 to_precision；介面可擴充 f32 |
| 9 | **實作注意 §3.11** | 並行寫入按 Request 邊界、Config、Backpressure、after() one-shot、§3.11.6／§3.11.7 | ✅ 完成 | 見 §二、§三、§五；盲區與地雷已核對 |

**小結**：總結優化清單 9 項均已對齊設計；Python 端 **`with engine.coalesce(...) as batcher:`** 已實作（CoalesceBatcher + flush 走 encode_list）；Observability 為可選、未實作。

---

## 〇之二、Dataflow 符合性（設計文件 §四「整體邏輯順序」）

設計文件 §四 規定之整體邏輯順序：

> 入口（encode/encode_batch → submit 或 run_coalesced）→ 佇列只存 metadata（Zero-Copy Ingress）→ 觸發（max_batch_* 或 flush）→ flush 寫入 Pinned Pool（One-Copy to Pinned）→ run_dual_stream_pipeline → 依邊界切回；Engine 為 Float32 時對結果 to_precision。

| 步驟 | 設計預期 | 實作對應 | 符合 |
|------|----------|----------|------|
| **1. 入口** | encode / encode_batch → submit 或 run_coalesced | **encode_list**：`gpu::encode_list(slices)` → `run_coalesced(..., \|batcher\| { for s in slices { batcher.submit(Single(s)) } })`。**Scheduler**：`submit_task` → `sender.send(task)`。 | ✅ |
| **2. 佇列只存 metadata** | Zero-Copy Ingress；不存 Vec&lt;f64&gt;，只存 ptr/len 或 `&[f64]` | `EncodeCoalescer.queue: Vec<EncodeRequest<'a>>`，`EncodeRequest::Single(&[f64])` / `Batch { data: &[f64], ... }`；僅引用，無 owned buffer | ✅ |
| **3. 觸發** | max_batch_samples / max_batch_bytes 或顯式 flush | **run_coalesced**：callback 結束後**顯式 flush**。**Scheduler**：Size 滿（≥ target / 下一筆超 max）或 **after()** 時間到 → flush_buffer | ✅ |
| **4. flush 寫入 Pinned** | 從 Pinned Buffer Pool 取一塊，**一次** 將所有 request 的 slice 寫入（並行 memcpy） | `Coalescer::flush`：`pool.acquire()` → `boundaries` 按 request 邊界 → `par_iter().zip(copy_ranges).for_each(copy_into)` 寫入 pinned_slice | ✅ |
| **5. run_dual_stream_pipeline** | Coalescer flush 呼叫 Fix 4 Pipeline，不呼叫舊 sync encode_batch | `encoder.encode_batch_via_pipeline(device, &pinned_slice[..total_elements], ...)`；AmplitudeEncoder 內為 `run_dual_stream_pipeline_aligned`（chunk + H2D/Compute overlap） | ✅ |
| **6. 依邊界切回** | 合併結果依 (start_sample, num_samples) 分發 | flush 回傳 `(batch_state, sample_boundaries)`；encode_list 回傳 boundaries；Scheduler 以 `zip(tasks, sample_boundaries)` 依序 `tx.send(Ok(sub_tensor))` | ✅ |
| **7. Engine to_precision** | Float32 時對結果 to_precision | `QdpEngine::encode_list` 內 `batch_state.to_precision(&self.device, self.precision)?` 再 to_dlpack | ✅ |

**結論**：Dataflow 與設計文件 §四「整體邏輯順序」**一致**；入口→佇列 metadata→觸發→flush 寫 Pinned→encode_batch_via_pipeline（run_dual_stream_pipeline）→依邊界切回→to_precision 均已按預期實作。

### 〇之三、Dataflow 三路徑逐項對照（實作追蹤）

| 路徑 | 入口 | 佇列／暫存 | 觸發 | flush 寫 Pinned | Pipeline | 切回／分發 |
|------|------|------------|------|-----------------|----------|------------|
| **encode_list** | `QdpEngine::encode_list(slices)` → `gpu::encode_list` → `run_coalesced(..., \|batcher\| { for s in slices { batcher.submit(Single(s)) } })` | `EncodeCoalescer.queue` 只存 `EncodeRequest::Single(&[f64])` | callback 結束後立即 `coalescer.flush()` | `pool.acquire()` → `par_iter().zip(copy_ranges).for_each(copy_into)` | `encoder.encode_batch_via_pipeline(device, pinned_slice, ...)` → run_dual_stream_pipeline_aligned | 回傳 `(batch_state, boundaries)`；Engine 內 to_precision → to_dlpack |
| **Python coalesce** | `with engine.coalesce(...) as batcher:` → `batcher.submit(array)` / `submit_batch(...)` 只存 `RefCell<Vec<Py<PyAny>>>` 引用 | 僅 Python 端 list 收集陣列引用，無 Rust 佇列 | `__exit__` 時組 list → 呼叫 `engine.encode_list(list, ...)` | 同 encode_list 路徑（一次 FFI 進入 Rust） | 同上 | `get_result()` 回傳 (QuantumTensor, boundaries) |
| **encode_async / Scheduler** | `engine.encode_async(data)` → `sender.send(EncodeTask)` | Scheduler 端 `buffer: Vec<EncodeTask>`（owned Vec&lt;f64&gt;，因跨 thread 無法持引用） | Size ≥ target 或下一筆超 max 或 `after(max_delay)` 到期 | `flush_buffer` → `run_coalesced(..., \|batcher\| { for t in tasks { batcher.submit(...) } })` → 同上 | 同上 | `zip(tasks, boundaries)` 依序 `task.tx.send(Ok((batch_arc, (start, num))))`；EncodeHandle::get() 用 copy_sample_range 取子範圍 |

**備註**：encode_list 與 Python coalesce 為 **Zero-Copy Ingress**（只存 `&[f64]` 或 Python 引用）；encode_async 因跨 thread 必須 clone 進 EncodeTask，設計文件 §3.9 已接受此取捨，並以 encode_list 減少 N 次 encode_async 的 FFI 開銷。

---

## 一、依賴版本（越新越好，查證來源 crates.io）

| Crate | 專案使用 | 查得最新 (crates.io) | 狀態 |
|-------|----------|----------------------|------|
| **crossbeam-channel** | 0.5.15 (qdp-core) | 0.5.15 (2025-04) | ✅ 已最新；§3.11.4 使用 `after`(one-shot)、`bounded`。 |
| **rayon** | 1.11 (workspace) | 1.11.0 (2025-08) | ✅ 已最新；並行寫入 Pinned 用。 |
| **thiserror** | 2.0 (workspace) | 2.0.x | ✅ 2.0 解析至最新 2.x。 |
| **cudarc** | 0.13 (workspace, cuda-12050) | 0.13.x / 0.18.2 | ✅ 0.13 對應 CUDA 12.5；0.18 主線需測相容性，維持 0.13。 |
| **log** | 0.4 (qdp-core) | 0.4.x | ✅ "0.4" 解析至最新 0.4.x。 |
| **nvtx** | 1.3 optional | 1.3.0 | ✅ 可選 observability。 |
| **pyo3** | 0.27 (qdp-python) | 0.27.2 (2025-11) | ✅ "0.27" 解析至 0.27.x；0.27.2 為最新 patch。 |
| **numpy** | 0.27 (qdp-python) | 0.27.1 (2025-12) | ✅ "0.27" 解析至 0.27.x；0.27.1 為最新 patch。 |

備註：
- **cudarc**：專案使用 `cuda-12050`（CUDA 12.5），與 0.13 對齊；0.18 支援 CUDA 11.4–13.0，升級需回歸測試。
- **log**：未鎖定小版號，`cargo update` 會取得 0.4.x 最新。
- **與設計無關的還原**：`qdp-core/src/encoding/mod.rs` 中將 `stream_encode` 的 event_slots 改為依 `PipelineConfig` 的變更已還原（該優化屬 PR3 Phase 3 pipeline 配置，非 Request Coalescing 設計文件範圍）。

---

## 二、§3.11.6 實作時要注意的地方 — 檢查結果

| 項目 | 要求 | 實作狀態 |
|------|------|----------|
| **空佇列 flush** | 空時 `flush()` 回傳 Err（或 Scheduler timeout 時 buffer 空則不 flush） | ✅ Coalescer 空時回傳 `Err(InvalidInput("flush called with empty queue"))`；Scheduler 僅在 buffer 非空時進入 select! timeout 分支並 flush。 |
| **結果順序** | encode_list / encode_async 結果與提交順序一致 | ✅ 依 `queue.iter()` / boundaries 順序分發；Scheduler 以 `zip(tasks, boundaries)` 依序 `tx.send`。 |
| **flush 失敗時** | 同一批每個 Task 的 tx 都要送 Err | ✅ Scheduler `flush_buffer` 在 `run_result` 為 Err 時對所有 `tasks` 送 `Err(...)`。 |
| **Config 來源一致** | CoalescerConfig 與 PipelineConfig 由同一層讀取，避免衝突 | ⚠️ 尚未在 Engine 層統一；之後整合時需注意 target_batch_size 與 pipeline chunk 對齊。 |
| **Backpressure** | Scheduler 用 bounded channel；send() block / try_send() 回傳 Full | ✅ `bounded(config.channel_capacity)`；`submit()`=send、`try_submit()`=try_send。 |
| **sample_size / num_qubits 一致** | 同一 coalescer 內一致，submit 時驗證 | ✅ submit 驗證 `sample_size`；num_qubits 為 coalescer 固定；encoding 由同一 encoder 保證。 |
| **Pinned Pool 單塊不足** | 方案 B：單次 flush 總大小 ≤ 單塊容量；submit 時 would_exceed 即先 flush 再收 | ✅ flush 時檢查 `total_elements > pool.elements_per_buffer()` 回傳 Err；submit 時總和超過 max_batch_bytes 回傳 Err。 |

---

## 三、§3.11.7 盲區檢查 — 檢查結果

| 盲區 | 要求 | 實作狀態 |
|------|------|----------|
| **Scope/callback panic** | flush 路徑中 Pinned Buffer 須經 RAII 歸還；panic 時 Drop 執行 | ✅ `PinnedBufferHandle` 在 flush 內取得，return/panic 時 drop 歸還 pool。 |
| **Pool 耗盡** | acquire 可 block 或 Err；避免與 pipeline 死鎖 | ✅ Coalescer 自建 pool（不與 pipeline 共用）；Scheduler 透過 run_coalesced 用 Coalescer 自建 pool；單一 flush 執行，無並行 acquire 死鎖。 |
| **encode_list 與 Scheduler** | 專用路徑或同一 Scheduler 二擇一/並存 | ✅ encode_list 走專用 Coalescer flush，不經 Scheduler（不經 max_delay）。 |
| **Pinned Buffer 歸還時機** | 必須在 pipeline 完成後（Ok 或 Err）歸還 | ✅ 先 `encoder.encode_batch_via_pipeline(...)` 完成，再 `drop(handle)`；Err 時同樣 drop handle。 |
| **Reentrancy** | flush 不在 flush 內部被呼叫 | ✅ Coalescer flush 不呼叫 coalescer；Scheduler flush_buffer 呼叫 run_coalesced（新建 coalescer），非同一 coalescer 重入。 |
| **單筆超過 max_batch_bytes／單塊容量** | 拒絕（submit Err）或 Pass-through | ✅ **已補**：submit 時若單筆 `request.total_elements()*8 > max_batch_bytes` 或 `> pool.elements_per_buffer()` 即回傳 Err（拒絕）。Scheduler 端單筆 ≥ target_batch_size 且 buffer 空時 Pass-through。 |
| **Observability** | 可選；佇列長度、flush 耗時等 | ⚠️ 未實作；可預留介面。 |

---

## 四、與設計文件差異（待補或取捨）

| 項目 | 文件 | 目前實作 | 說明 |
|------|------|----------|------|
| **Coalescer flush 呼叫對象** | §3.6.1：flush 必須呼叫 **run_dual_stream_pipeline**（Fix 4），不呼叫舊的 sync encode_batch | ✅ 改為呼叫 **encoder.encode_batch_via_pipeline**(device, pinned_slice, ...)；AmplitudeEncoder 實作內使用 run_dual_stream_pipeline_aligned，符合 §3.6.1。 |
| **target_batch_size 預設** | §3.8：8 MB 為 Safe Default，為 Config 不寫死 | ✅ SchedulerConfig.target_batch_size = 8*1024*1024；CoalescerConfig.target_batch_size 為 Option | 已滿足。 |
| **調度迴圈 one-shot** | §3.11.4：用 **after(duration)** 不要用 tick | ✅ Scheduler 使用 `after(d.saturating_duration_since(now))` | 已滿足。 |

---

## 五、盲區與地雷逐項核對（§3.11.7 第二遍順過）

以下對照設計文件「盲區檢查」與「實作時要注意的地方」，確認無遺漏或踩雷。

| 項目 | 文件要求 | 實作核對 | 結果 |
|------|----------|----------|------|
| **Coalescer 無內部 Vec&lt;f64&gt;** | §一之五、§3.6：Coalescer **不**維護內部 `Vec<f64>`，只存 metadata | `EncodeCoalescer.queue` 為 `Vec<EncodeRequest<'a>>`，`EncodeRequest` 僅持 `&[f64]`；無 `Vec<f64>` 資料緩衝區 | ✅ |
| **Scope/callback panic** | flush 路徑中 Pinned 須 RAII 歸還；panic 時 Drop 執行 | `run_coalesced` 先執行 callback（僅 submit），**再**呼叫 `flush`；`flush` 內取得 `PinnedBufferHandle`，任一步 panic 時 handle 在 stack 上會 Drop 歸還 | ✅ |
| **Pool 耗盡** | acquire 可 block 或 Err；避免與 pipeline 死鎖 | Coalescer 自建 pool（pool_size=2）；Scheduler 每次 `flush_buffer` 僅一次 `run_coalesced`，單一 flush 執行，無並行 acquire | ✅ |
| **encode_list 與 Scheduler** | 專用路徑或同一 Scheduler 二擇一/並存 | `encode_list` 走專用 `run_coalesced`，不經 Scheduler、不經 max_delay；Scheduler 為獨立 thread + channel | ✅ |
| **Pinned 歸還時機** | 必須在 pipeline 完成後（Ok 或 Err）歸還 | `flush` 內：`encode_batch_via_pipeline(...)?` 完成後才 `drop(handle)`；Err 時同樣 `drop(handle)` 再對各 tx 送 Err | ✅ |
| **Reentrancy** | flush 不在 flush 內部被呼叫 | Coalescer 的 `flush` 不呼叫 coalescer；Scheduler 的 `flush_buffer` 呼叫 `run_coalesced`（**新建** coalescer），非同一 coalescer 重入 | ✅ |
| **單筆超過 max_batch_bytes／單塊容量** | 拒絕（submit Err）或 Pass-through | **Coalescer**：`submit()` 檢查 `req_bytes > max_batch_bytes` 及 `request.total_elements() > pool.elements_per_buffer()` 即回傳 Err。**Scheduler**：單筆 ≥ target 且 buffer 空時 Pass-through；若單筆 > 單塊容量，該筆在 `run_coalesced` 內 `batcher.submit()` 會 Err，flush 回傳 Err 後對該批所有 task 送 Err | ✅ |
| **調度迴圈 one-shot** | §3.11.4：用 **after(duration)** 不要用 tick | `scheduler.rs` 使用 `after(d.saturating_duration_since(now))`，每輪 timeout 後重新建立，為 one-shot | ✅ |
| **並行寫入非重疊** | §3.11.5：按 Request 邊界切分、非重疊 | flush 內 `boundaries` 依序 (offset, len)，`par_iter().zip(copy_ranges)` 每 request 寫入各自 offset，區間不重疊 | ✅ |
| **Scheduler 單筆 > max 預先拒絕** | （可選）Scheduler 在 recv 後先檢查 task_bytes > max_batch_bytes 立即送 Err | ✅ **已實作**：若 `task_bytes > max_bytes` 立即 `task.tx.send(Err(...))` 並 `continue`，不進 buffer，減少無效 flush。 | ✅ |

結論：**未踩到文件所列盲區或地雷**；Scheduler 單筆過大預先拒絕已實作。

---

## 六、Rust 實作品質（慣例與 Edition 2024 相容）

對照 **rust-development** 技能：依賴查證、錯誤處理、unsafe 最小化、文件引用、命名慣例。專案使用 **edition = "2024"**、**rust-version = "1.85"**（qdp/Cargo.toml）。

| 項目 | 要求 | 實作狀態 | 備註 |
|------|------|----------|------|
| **錯誤處理** | 使用 `Result<T, E>`、`?` 傳播；庫程式碼避免 `unwrap()` | ✅ coalescer / scheduler / encode_list 均回傳 `Result`；錯誤以 `MahoutError` 傳播；`unwrap()` 僅出現在單元測試內 | 符合 Rust 慣例 |
| **unsafe 使用** | 最小化；必要時須有註解說明安全條件 | ✅ **coalescer**：`SendPtr::copy_into` 內 `from_raw_parts_mut` + `copy_from_slice`，註解註明「each thread writes to dst[off..off+len] with disjoint ranges」；`Send`/`Sync` 僅用於 rayon 並行寫入，範圍不重疊。**scheduler / encode_list** 無額外 unsafe | 符合 §3.11.1 非重疊寫入 |
| **文件引用** | 關鍵 API 附 Ref: docs.rs 或設計文件 | ✅ coalescer：`Ref: doc.rust-lang.org/std/primitive.slice.html`、設計 §3.2/§3.6；scheduler：`Ref: https://docs.rs/crossbeam-channel/0.5/...`（after、bounded）；qdp-core encode_list：設計 §3.9、§3.11.2 | 可追溯 |
| **命名** | snake_case 函式／方法；PascalCase 型別 | ✅ `run_coalesced`、`encode_list`、`EncodeRequest`、`CoalescerConfig`、`EncodeCoalescer` | 符合 |
| **依賴版本** | 與 Cargo.toml 一致；API 查證 | ✅ crossbeam-channel 0.5.15、rayon 1.11、thiserror 2.0 等已於 §一 對照 crates.io | 已確認 |
| **Deprecated** | 未使用已棄用 API | ✅ 未使用 deprecated API | — |

**建議（可選）**：
- **Config 來源一致**（§3.11.6）：未來在 Engine 層統一讀取 CoalescerConfig 與 PipelineConfig，可避免 target_batch_size 與 pipeline chunk 不一致。
- **Observability**：可預留佇列長度、flush 耗時等 metrics 介面（設計 §3.11.7 為可選）。

---

## 七、總結

- **版本**：crossbeam-channel 0.5.15、rayon 1.11、log 0.4、thiserror 2.0、cudarc 0.13、nvtx 1.3 均為目前採用／解析之版本，且與 crates.io 最新一致或為合理鎖定。
- **§3.11.6 / §3.11.7**：空佇列 flush、結果順序、flush 失敗分發、Backpressure、單筆超過拒絕（Coalescer submit）、RAII 歸還、不重入、encode_list 專用路徑、並行寫入非重疊、one-shot after() 均已對齊；Config 來源一致與 Observability 為後續整合與可選。
- **§3.6.1 已符合**：Coalescer flush 呼叫 **encoder.encode_batch_via_pipeline**；AmplitudeEncoder 實作使用 run_dual_stream_pipeline_aligned，串接 Batching 與 Pipelining。
- **盲區與地雷**：第二遍順過 §3.11.7 與設計要點，**未發現違反**；Scheduler 單筆 > max 預先拒絕已實作。
- **Python encode_list**：§3.9、§3.11.2 已對齊：qdp-core `QdpEngine::encode_list` + qdp-python `engine.encode_list(list_of_arrays, sample_size, num_qubits, encoding_method)` 回傳 `(QuantumTensor, boundaries)`，一次 FFI 走 coalescer + pipeline。
- **設計完成度（§〇）**：總結優化清單 9 項均已完成（含 Python `with engine.coalesce(...) as batcher:`）；encode_async + EncodeHandle、encode_list、CoalesceBatcher 已實作；核心 dataflow 與 §四 整體邏輯順序一致。
- **盲區／地雷**：§3.11.6、§3.11.7 已逐項核對，未踩到文件所列盲區；與設計無關的變動（encoding/mod.rs stream_encode 之 PipelineConfig）已還原。
- **設計完成度**：**REQUEST_COALESCING_REFERENCE_AND_DESIGN.md** 總結優化清單 §四／§六 共 9 項 **均已開發完成**；Dataflow 與 §四「整體邏輯順序」及三路徑（encode_list、Python coalesce、encode_async）**符合預期**；Rust 實作符合 Edition 2024 慣例與 rust-development 要點（錯誤處理、unsafe 最小化、文件引用）。
