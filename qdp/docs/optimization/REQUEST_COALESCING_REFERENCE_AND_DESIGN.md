# Request Coalescing（請求合併）參考與設計

**目的**：參考 Triton、tower-batch、Ray Serve 的請求合併實作與相關論文／官方文件，整理可改進之處，並設計可同時用於 `encode()` 與 `encode_batch()` 的 QDP 實作。

**從頭順一遍（文件結構）**：**一** 參考專案（Triton / tower / Ray）與論文 → **二** 共通模式與改進方向 → **三** QDP 設計（目標、抽象、觸發、API、Zero-Copy、實作要點、Pipeline 整合、彈性調度器 §3.8、統一入口與 encode_list §3.9、Review §3.10、Final Reality Check §3.11、要注意的地方 §3.11.6、**盲區檢查 §3.11.7**）→ **四** 小結與總結優化清單 → **五** Float32 預設存在 → **六** 小結與結論。實作時可依此順序對照檢查；**第二遍順過** 見 §3.11.7 確認無盲區。

**更新說明**：本設計經深度 Review 與 **Final Reality Check** 後已納入：（1）**Zero-Copy + One-Copy to Pinned** 的數據支撐（DPDK、memcpy／PCIe 頻寬）、（2）**Scope-based 與 Python GIL** 的交互與 **Scheduler Thread 建議**、（3）**Time-Size 雙門檻** 的參數建議（max_delay 0.1 ms ~ 1 ms）、（4）**Float32 預設存在**（§五）、（5）**極限狀況與實作注意**（§3.11）：Rayon 並行寫入 Pinned 按 Request 邊界切分與非重疊安全、Scheduler Backpressure（bounded channel / try_send）、**encode_list**、target_batch_size 為 Config、調度迴圈 crossbeam::select! + timer 或 recv_timeout；（6）**實作時要注意的地方**（§3.11.6）：空佇列 flush、結果順序（Triton/Ray preserve order）、flush 失敗分發、Config 來源一致、Backpressure 實作、sample_size 與 **num_qubits/encoding 一致**、**Pinned Pool 單塊不足**（建議 max_batch_bytes 與單塊容量對齊）。調度迴圈 **one-shot 建議用 crossbeam `after(duration)`**，不用 tick（§3.11.4）。（7）**盲區檢查（第二遍順過）**（§3.11.7）：Scope/callback panic 時 Pinned Buffer 須經 RAII 歸還、Pool 耗盡時 block 或 Err 與死鎖防範、**單筆超過 max_batch_bytes／單塊容量**（拒絕或 Pass-through + pipeline 內部拆批）、encode_list 與 Scheduler 關係（專用路徑 vs 同一 Scheduler）、Pinned 歸還時機（pipeline 完成或 Err 後）、flush 不重入、Observability 可選。**查證摘要**：Triton `max_queue_delay_microseconds` 預設 0、建議以 Performance Analyzer 調參，100 μs 為常見範例，與本文件 0.1 ms ~ 1 ms 一致；crossbeam `after(duration)` 為 **one-shot**（duration 後只送一筆），`tick(duration)` 為 **週期性**，見 [crossbeam after](https://docs.rs/crossbeam/latest/crossbeam/channel/fn.after.html)、[crossbeam tick](https://docs.rs/crossbeam/latest/crossbeam/channel/fn.tick.html)。見 §3.10、§3.11、§五。

---

## 一、參考專案與官方文件：他們怎麼做

### 1.1 NVIDIA Triton Inference Server（Dynamic Batcher）

**官方文件**：
- [Batchers — Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- [Schedulers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/scheduler.html)
- [Dynamic Batching & Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html)（教學與效能範例）

**原始碼位置**：scheduler 與 dynamic batching 邏輯在 **triton-inference-server/core**（[GitHub](https://github.com/triton-inference-server/core)）的 C++ 實作；server 倉庫（[triton-inference-server/server](https://github.com/triton-inference-server/server)）為 HTTP/gRPC 與多 server 整合。

**核心邏輯**（對應官方文件與教學）：

- **佇列**：每個 model 有一個 scheduler；inference 請求進入佇列，由 dynamic batcher **動態合併**成 batch，再分發給所有 model instance。
- **觸發條件**：
  - **preferred_batch_size / max batch size**：盡量湊成偏好或最大 batch 再送給 instance；若無 preferred，則送「目前佇列內能湊到的最大 batch（≤ max）」。
  - **max_queue_delay_microseconds**：若湊不滿 preferred/max，**最長等待時間**；時間到就把目前佇列內請求當成一個 batch 送出（避免單一請求等太久）。
- **Priority Queue**：`priority_levels`、`default_priority_level`、`priority_queue_policy`，高優先請求可插隊；同優先級內依接收順序。
- **Queue Policy**：`max_queue_size`、`timeout_action`（REJECT / DELAY）、`default_timeout_microseconds`、`allow_timeout_override`，控制排隊長度與逾時行為。
- **Preserve ordering**：可選是否依請求順序回傳結果。
- **Custom Batching**：可透過 `TRITONBACKEND_ModelBatchIncludeRequest` 等五個函式實作自訂合併策略（見 Batchers 文件）。

**官方教學實測**（Part 2 - Dynamic Batching）：
- 無 dynamic batching、單 instance：約 975 infer/sec（16 並發）。
- 僅 dynamic batching：約 3187 infer/sec（16 並發），queue wait 下降、throughput 明顯上升。
- Dynamic batching + 多 instance：約 4134 infer/sec；說明「單純開滿 feature 未必最優」，需依負載與 SLA 調參。

**可取之處**：
- 延遲與吞吐的明確取捨（delay vs batch size）。
- 優先級與逾時政策完整，適合多租戶／SLA。
- preferred_batch_size 可對齊 TensorRT 等多 profile 的效能甜點。
- 官方建議流程：先開 default dynamic_batching，用 Performance Analyzer 量測，再調 max batch size 與 delay。

**可改進／不直接適用**：
- 以「請求個數」與「delay」為主，較少以 **bytes / samples / GPU 記憶體** 為單位做上限；我們需要 **max_batch_bytes / max_batch_samples** 避免 OOM。
- 架構重（C++ scheduler、多 process）；我們要在單一 process 內、Rust 側做輕量 coalescing。

---

### 1.2 Rust tower Middleware（Buffer / Batch）

**tower::buffer**：[Buffer — tower](https://docs.rs/tower/latest/tower/buffer/struct.Buffer.html)

- **MPSC 佇列**：當下層 `Service` 滿負載時，請求先進 buffer（有界佇列），不丟棄。
- **語意**：背壓（backpressure），不是「把多個 request 合併成一個 batch 呼叫」；buffer 是「排隊」而非「合併」。

**tower-batch**（如 [chmodas/tower-batch](https://github.com/chmodas/tower-batch)）：

- **語意**：buffer 直到 **達到最大數量** 或 **最長等待時間**，然後把這段時間內累積的 **多個 `Service::call` 合併成一個 batch** 呼叫內層 Service。
- 內層 Service 的 `call` 簽名為 **batch**：`Request = Vec<SingleRequest>`，一次處理多筆。
- 與 Ray Serve / Triton 的「先排隊、再合併、再一次執行」一致。

**可取之處**：
- 明確的 **max size / timeout** 雙條件觸發。
- 與 Tower 的 `Service` 抽象相容，可疊加在其他 Layer 上。

**可改進／不直接適用**：
- 我們底層是 **同步** GPU API（encode / encode_batch），不是 async `Service`；需要一層適配（sync 介面 + 可選 background worker，或僅提供 submit/flush）。
- tower-batch 以「請求個數」為主；我們需要 **按 bytes/samples 上限** 與 **sample_size 對齊**，以配合 pipeline chunk 與 OOM 防護。

---

### 1.3 Ray Serve（Python）

**官方文件**：
- [Dynamic Request Batching](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html)
- [ray.serve.batch API](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.batch.html)

**原始碼**：`python/ray/serve/batching.py`（[GitHub](https://github.com/ray-project/ray/blob/master/python/ray/serve/batching.py)）

**核心邏輯**（對應 `batching.py` 實作）：

- **佇列**：`_BatchQueue` 使用 `asyncio.Queue[_SingleRequest]`；每個請求帶 `future`、`flattened_args`、`request_context`。
- **scheduler_loop 等價**：`_process_batches()` 無限迴圈：
  1. `await self.wait_for_batch()` 取得一批請求；
  2. 依 `multiplexed_model_id` 可選拆成 sub-batches；
  3. 在 `async with self.semaphore` 下呼叫 `_process_batch_inner()` 執行使用者 batch 函式，結果依序 `future.set_result()` 回傳給各請求。
- **wait_for_batch()**：
  - 先 `await self.queue.get()` 取得 **第一個請求**（至少一筆才返回）；
  - 以 `batch_start_time = time.time()` 開始計時；
  - 迴圈：若設有 `batch_size_fn`，則持續從 queue 取件並暫加入 batch，用 `_compute_batch_size(batch)` 檢查；若加入下一筆會超過 `max_batch_size`，則把該筆 **放回 queue 後端**（FIFO），並結束此 batch；若無 `batch_size_fn`，則用 `len(batch) < max_batch_size` 且 `not queue.empty()` 取滿為止；
  - 若 `time.time() - batch_start_time >= batch_wait_timeout_s` 或已達 max batch size，則跳出並返回當前 batch；
  - 有 **batch_wait_timeout_s** 的 `asyncio.wait_for(requests_available_event.wait(), remaining_batch_time_s)`，時間到即返回目前累積的 batch。
- **參數**：
  - `max_batch_size`（預設 10）
  - `batch_wait_timeout_s`（預設 0.01 s）
  - `max_concurrent_batches`（預設 1）：semaphore 限制同時執行的 batch 數
  - `batch_size_fn`：`(List[Item]) -> int`，用於 GNN total nodes、NLP total tokens 等，避免單一 batch 超過 GPU 記憶體；若單一 item 的 size 就大於 `max_batch_size`，會對該請求 `future.set_exception(RuntimeError)`。
- **Observability**：batch wait time、batch execution time、queue length、batch utilization、actual batch size 等 histogram/gauge。

**可取之處**：
- 邏輯與 Triton 的 delay + max size 一致，易對照。
- **batch_size_fn** 直接對應我們需求：按 **samples 或 bytes** 累積，上限用於 OOM 與 pipeline 伸縮。
- 原始碼結構清楚：queue → wait_for_batch（雙條件）→ _process_batches → semaphore + _process_batch_inner。

**可改進／已知限制**：
- 文件與 issue 提到：batch 執行本身是 **一次一個 batch**（由 semaphore 控制），多個 batch 可並行但受 `max_concurrent_batches` 限制；若底層為同步或 I/O bound，仍可能成為瓶頸。
- 介面為 Python async；我們在 Rust 要做等價的「佇列 + 雙條件觸發」，可用 sync + 明確 flush，或未來用 async + spawn worker。

---

## 一之四、相關論文與官方研究

| 來源 | 要點 | 與 QDP 的對應 |
|------|------|----------------|
| **SMDP-Based Dynamic Batching**（IEEE 等） | 將 batch service queue 建模為 semi-Markov decision process（SMDP），在延遲與功耗間做取捨；動態調整 batch 大小以適應負載，可達 3× 以上 throughput 並維持 SLA。 | 我們以 **max_batch_samples / max_batch_bytes + 顯式 flush** 做簡化版「動態合併」，未來可考慮 timeout 或更細的 policy。 |
| **Symphony / Deferred Batch Scheduling** | 「deferred batch scheduling」：在滿足延遲目標下定義可排程視窗，收集更多請求再合併，相較先前系統可達約 5× throughput。 | 與 Triton / Ray 的 delay + max size 一致；我們首版用顯式 flush，之後可加 **batch_wait_timeout** 對應「視窗」。 |
| **Static Batching of Irregular Workloads（MoE / GNN 等）** | 不規則 workload（變長序列、圖節點數）以 **有效 batch 大小**（如 total tokens、total nodes）做上限，避免 OOM 並維持 GPU 利用率。 | 對應 Ray 的 **batch_size_fn** 與我們的 **max_batch_bytes / max_batch_samples**；encode_batch 的 sample_size 對齊即「有效單位」。 |
| **Triton Model Analyzer** | 官方工具自動搜尋 dynamic batcher 與 instance 配置，平衡 latency 與 throughput。 | 我們可依 §七之四 伸縮調度與 coalescer 做手動或腳本化調參。 |

---

## 一之五、頂級開源專案對照與 QDP 改進方向（建議採納）

| 參考專案 | **NVIDIA Triton (Dynamic Batcher)** | **Rust `tower::batch`** | **Ray Serve (Python)** | **QDP-Core（我們的優化目標）** |
|----------|-------------------------------------|-------------------------|-------------------------|--------------------------------|
| **核心機制** | **Time-Size Dual Threshold**（時間到 or 塞滿就發車） | **Service Middleware**（裝飾者模式，攔截請求並緩衝） | **Decorator**（裝飾器，收集參數變 list） | **Pinned Memory Aware Batcher**（感知硬體的調度器） |
| **緩衝區管理** | Queue（先進先出），支援 Priority | `Vec<Request>` | Python List | **Pre-allocated Pinned Buffer**（或從 Pool 取） |
| **優點** | 成熟穩定，支援 Ragged Batching（不規則形狀） | Rust 生態標準，整合 Tokio 完美 | 對使用者透明，易用 | **針對 PCIe H2D 優化**，減少 Host 端記憶體搬運與重分配 |
| **缺點/瓶頸** | 通用設計，對 Host Memory Copy 優化較少 | 依賴 Tokio Runtime（若專案是同步的會難接） | Python overhead 高 | 需要自己寫邏輯 |

**我們可以改進的地方（超越通用方案）**：
- 通用方案（如 Triton）通常假設 Input 在 CPU Memory，然後 copy 到 Staging Area，再 copy 到 GPU。
- **我們的優勢**：我們知道輸入是 `f64` 且要送往 PCIe。可以 **直接將使用者的 Request 寫入 Pinned Memory**（從 Pinned Buffer Pool 取得一塊），省去中間的 `Vec` 搬運與重分配。
- **必須明確寫入實作計畫**：
  - **Coalescer 不應該有自己的 `Vec<f64>` buffer。**
  - **flush() 時**：先計算總大小 → 從 **Pinned Buffer Pool**（Fix 3/4 的產物）申請一塊夠大的 Pinned Memory → 將所有 Request 的 **`&[f64]`** **一次寫入**該 Pinned Memory（可 **並行 memcpy**，例如 rayon，利用 CPU 多核頻寬）→ 直接送進 Pipeline（**run_dual_stream_pipeline**）。
- 如此達成 **Zero-Copy 進入 Queue**（只存 ptr/len 或 `&[f64]`）+ **僅在 flush 時一次 Copy（來源 slice → Pinned）**，避免兩次 Host Copy 抵銷 PCIe 優勢（見 §3.5.1、§3.6）。

---

## 二、共通模式與可改進點整理

| 項目 | Triton | tower-batch | Ray Serve | 我們要的 |
|------|--------|-------------|-----------|----------|
| **觸發條件** | max size + max_queue_delay | max size + max duration | max_batch_size + batch_wait_timeout_s | **max_batch_samples / max_batch_bytes** + **batch_wait_timeout** |
| **Batch 單位** | 請求個數 / preferred_batch_size | 請求個數 | 請求個數 或 batch_size_fn | **samples 或 bytes**，對齊 sample_size |
| **優先級** | 有 | 無 | 無 | 可選（後續） |
| **逾時政策** | reject / defer | 無 | 無 | 可選（逾時仍 flush） |
| **記憶體上限** | 間接（max batch size） | 無 | batch_size_fn 可表達 | **必備**：max_batch_bytes 防 OOM |
| **適用 encode()** | 需適配 | 需適配 | 需適配 | **統一**：單次 encode 視為 1-sample batch |
| **適用 encode_batch()** | 是 | 是 | 是 | **統一**：多個小 batch 合併成一大 batch |

**改進方向**（相對上述專案）：
1. **以 bytes/samples 為主要上限**，並設 **max_batch_bytes / max_batch_samples**，與 §七之四 伸縮調度一致，避免 OOM。
2. **encode() 與 encode_batch() 共用同一套 coalescer**：單次 encode(data) 視為 1 筆「邏輯 sample」；encode_batch(data, n, ...) 視為 n 筆；合併時 flatten 成一大塊 + 每筆邊界，一次 pipeline 後再依邊界切回。
3. **同步友善**：先做 **顯式 submit + flush**，不強制 async/thread，呼叫端可一次多個 submit 再一次 flush。
4. **可選**：之後加 **batch_wait_timeout** 需 background worker 或 async runtime；首版可僅「滿 max 或顯式 flush」觸發。
5. **Pinned Memory 優化**：flush 時盡量使用 **預先分配 Pinned Buffer** 或 Pool，減少 Host 端 `Vec` 重分配與多餘搬運（見 §一之五）。
6. **Python 非同步**：若 Python 單執行緒迴圈 `for batch in batches: engine.encode(batch)`，每次呼叫都會 block，Scheduler 永遠收不到第二筆，Dynamic Batching 無法生效；**必須**提供 **encode_async**（射後不理）＋ 最後再收割結果（見 §3.9）。

---

## 三、QDP 設計：Request Coalescing 用於 encode() 與 encode_batch()

### 3.1 目標

- **encode()**：多個小 1D 呼叫可被合併成「一大塊 1D」或「一個多 sample batch」，走一次 pipeline，減少 sync 次數。
- **encode_batch()**：多個小 batch 呼叫可被合併成一個大 batch，走一次 pipeline（或內部拆批 §七之四），減少 sync。
- **共用**：同一佇列、同一套觸發條件（max_batch_samples / max_batch_bytes + 可選 timeout + 顯式 flush）。

### 3.2 抽象：Coalescer 輸入單位

- **EncodeRequest::Single(host_data)**：對應一次 `encode(host_data)`，邏輯上 1 sample，長度 = state_len。
- **EncodeRequest::Batch { batch_data, num_samples, sample_size }**：對應一次 `encode_batch(batch_data, num_samples, sample_size, ...)`，邏輯上 num_samples 筆。

合併時：
- 將所有 Single 與 Batch 的資料 **依序 flatten** 成一大塊 `Vec<f64>`；
- 記錄每筆的 **邊界**（offset, len）或 (start_sample, num_samples)；
- 若統一成「batch 語意」：可視為一大個 encode_batch(big_batch_data, total_samples, sample_size, ...)，其中 total_samples = sum(single 的 1 + batch 的 num_samples)，但 **Single 的 sample_size = state_len**，Batch 的 sample_size 可能不同（需同 num_qubits / encoding）。
- **約束**：同一 coalescer 只接受相同 **num_qubits / encoding / sample_size**（或僅支援 amplitude 同 sample_size），以便合併成一個大 batch。

### 3.3 觸發條件（與參考一致 + 我們擴充）

1. **max_batch_samples**：佇列內邏輯 sample 數 ≥ 此值則 flush。
2. **max_batch_bytes**：佇列內總 `f64` 數 × 8 ≥ 此值則 flush（防 OOM）。
3. **target_batch_size**（可選，對齊 §3.8 彈性調度器）：累積 bytes ≥ 此值（例如 8 MB，Pipeline 甜蜜點）即 flush；可與現有 max_batch_bytes 並存（target ≤ max）。
4. **batch_wait_timeout / max_delay**（可選）：自第一個請求進佇列起，超過此時間則 flush；需 Scheduler thread 或 async。
5. **顯式 flush()**：呼叫端呼叫時立即 flush。

首版可只做 1 + 2 + 5；target_batch_size 與 timeout 留作擴充（見 §3.8）。

### 3.4 API 形狀（建議）

**方案 A（顯式 submit + flush，同步，owned 資料）**

- `coalescer.submit(EncodeRequest::Single(host_data))`（`host_data: Vec<f64>` 或 owned）→ 回傳 `Ticket` 或 index。
- `coalescer.flush() -> Result<Vec<GpuStateVector>>`：將佇列內所有請求合併成一大 batch，走一次 pipeline，再依邊界切回。
- **缺點**：若 `EncodeRequest` 為 owned `Vec<f64>`，Python 傳入的 slice 需先 Clone 進 Queue，flush 時再 Copy 拼成大 Buffer → **兩次 Host Copy**（見 §3.5.1）。

**方案 B（相容現有 API 的內部 coalescing）**

- 在 encoder 內維護 coalescer；`encode()` / `encode_batch()` 依門檻決定直接執行或 submit 後延遲 flush。
- 需明確 flush 觸發點；若仍用 owned 資料，同樣有兩次 Copy 風險。

**方案 C（Scope-based / Callback，推薦，Zero-Copy）**

- **`engine.run_coalesced(|batcher| { batcher.submit(&req1); batcher.submit_batch(&batch3, ...); Ok(()) })`**：在 **同一 Scope 內** 提交 **`&[f64]`**，Scope 結束時自動 flush。
- Coalescer 只存 **metadata**（ptr, len），不存 `Vec<f64>`；flush 時從 **Pinned Buffer Pool** 取得一塊，**一次** 將所有 request 的 slice 寫入（可並行 memcpy），再送進 **run_dual_stream_pipeline**。
- **一次 Host Copy**（來源 → Pinned），無進 Queue 的 Clone；編譯器保證在 flush 前資料有效。
- **Python 端對應**：Context Manager **`with engine.coalesce() as batcher:`**，在 `with` 內 `batcher.submit(...)`，離開 `with` 時觸發 flush。

建議 **效能優先時採用方案 C**（Zero-Copy Ingress + One-Copy to Pinned）；若實作複雜度優先，可先做方案 A，再演進為 C。

### 3.5 與 encode() / encode_batch() 的對應

- **encode(host_data)**：
  - 不經 coalescer：維持現有 `encode_async_pipeline`（大資料）或 sync 路徑（小資料）。
  - 經 coalescer：`submit(EncodeRequest::Single(host_data))`；flush 時將所有 Single 與 Batch 合併，其中 Single 視為 1 sample、sample_size = len(host_data)，與其他 Batch 的 sample_size 需一致（同 num_qubits），合併後走 **encode_batch pipeline**（一大塊），再依邊界切出每筆的 `GpuStateVector`，第一筆對應該 Single。

- **encode_batch(batch_data, num_samples, sample_size, ...)**：
  - 不經 coalescer：維持現有 sync 或未來 pipeline 版。
  - 經 coalescer：`submit(EncodeRequest::Batch { batch_data, num_samples, sample_size })`；flush 時合併成一大 batch，走一次 pipeline，再依 (start_sample, num_samples) 切回多個 `GpuStateVector`。

兩者共用 **同一 pipeline 路徑**（大 batch 的 encode_batch + §七之四 伸縮調度），只是輸入來源是「多個小請求合併」或「單次大請求」。

---

### 3.5.1 潛在的效能與易用性陷阱：Ownership 與 Data Copy

若採用 **owned `Vec<f64>`** 的 `EncodeRequest`：

- Python 傳過來的 Numpy Array（借用 / Slice）必須先 **Clone** 一次才能放進 Coalescer 的 Queue。
- `flush` 時又要再 **Copy** 一次拼成大 Buffer。
- **兩次 Host Copy** 可能抵銷 Pipeline 帶來的 PCIe 優勢，是效能的底線風險。

因此設計上應追求：**Zero-Copy 進入 Queue + 僅在 flush 時發生一次 Copy（來源 slice → Pinned Memory）**。

---

### 3.5.2 Zero-Copy Coalescer（修正建議）

**目標**：**Zero-Copy Ingress + One-Copy to Pinned**，作為高效能 Coalescer 的底線。

- **進入 Queue**：不儲存 `Vec<f64>`，只儲存 **`&[f64]`**（或 metadata：ptr + len）。需保證在 flush 前資料有效 → 用 **Scope-based / Callback** API 讓編譯器保證生命週期。
- **flush**：Coalescer **不維護自己的 `Vec<f64>` buffer**。
  1. 計算總大小 `Total_N`。
  2. 從 **Pinned Buffer Pool**（Fix 3/4 的產物）申請一塊夠大的 Pinned Memory。
  3. 將所有 Request 的 `&[f64]` **一次寫入**這塊 Pinned Memory（**必須並行 memcpy**，例如 rayon：單核 memcpy 約 10–15 GB/s，PCIe Gen4 約 26 GB/s，不並行則 Host Copy 可能成為瓶頸；見 §3.10.1）。**實作時按 Request 邊界切分、每 Request 寫入專屬 offset，區間不重疊**，避免 False Sharing 並符合 Pinned Memory 多線程寫入安全條件（§3.11.1）。
  4. 將該 Pinned Buffer 送進 **run_dual_stream_pipeline**（Fix 4，自動切 Chunk、Overlap H2D/Compute）。
- 如此只有 **一次 Host Copy**（來源 slice → Pinned），避免進 Queue 時的 Clone 與 flush 時的額外 Vec 分配。

---

### 3.6 實作要點（Rust）（修正版）

- **EncodeRequest<'a>**：
  - 改為持有 **`data: &'a [f64]`**（Slice），不持有 `Vec<f64>`。
  - 例如：`Single(&'a [f64])`、`Batch { data: &'a [f64], num_samples, sample_size }`。
  - 這樣 **進 Queue 時無需 Clone**，僅記錄指標與長度；生命週期由 Scope-based API 保證（見下）。

- **EncodeCoalescer<'a>**（或僅在 `run_coalesced` 內使用的 Batcher）：
  - **不**維護內部 `Vec<f64>` 緩衝區，只存 **metadata**（每個 request 的 ptr、len、request_type / start_sample, num_samples）。
  - `submit(&mut self, request: EncodeRequest<'a>)`：只推入 metadata，不複製資料。
  - **flush**（或 `flush_into_pinned_buffer`）：
    1. 算總大小 `Total_N`；
    2. `let pinned_buf = pool.acquire(Total_N)`（從 Pinned Buffer Pool 取得）；
    3. 將所有 request 的 `&[f64]` **寫入** `pinned_buf`（可並行：多段 slice 並行 copy 到對應 offset，例如 rayon）；
    4. 呼叫 **run_dual_stream_pipeline**（或 encode_batch pipeline）傳入 `pinned_buf`，得到合併後的 `GpuStateVector`；
    5. 依邊界切出每筆結果，回傳或分發；歸還 Pinned Buffer 給 Pool。

- **API 介面（Scope-based / Callback）**：
  - 新增 **`engine.run_coalesced<F, R>(callback: F) -> Result<R>`**，其中 `callback` 簽名為 `F: FnOnce(&mut EncodeCoalescer<'?_>) -> Result<R>`（或等價的 Batcher 型別）。
  - 使用者在 **同一 Scope 內** 對 `batcher` 呼叫 `submit` / `submit_batch`，**Scope 結束時自動 flush**；Rust 編譯器保證在 flush 時所有 `&[f64]` 仍有效，無需 Clone。
  - 概念範例：
    ```rust
    engine.run_coalesced(|batcher| {
        batcher.submit(EncodeRequest::Single(req1));  // req1: &[f64]
        batcher.submit(EncodeRequest::Single(req2));
        batcher.submit_batch(batch3, num_samples, sample_size);
        Ok(())  // 或回傳其他 R；closure 結束時自動 flush
    })?;
    ```
  - **Python 端對應**：Context Manager **`with engine.coalesce() as batcher:`**，在 `with` 區塊內呼叫 `batcher.submit(...)` / `batcher.submit_batch(...)`，**離開 `with` 時觸發 flush**，回傳 `Vec<Tensor>` 或依序分發結果。

- **encode() / encode_batch()**：
  - 不經 coalescer 時維持現有路徑；
  - 經 coalescer 時改為在 **run_coalesced**（或 `with engine.coalesce()`）內提交，避免呼叫端手動持有資料所有權與兩次 Copy。

---

### 3.6.1 與現有 Pipeline 的整合（必須明確）

- **Coalescer 的 flush 必須呼叫 Fix 4 優化過、支援 Chunking 的 Pipeline**，而不是舊的 `encode_batch` 同步路徑。
- **流程**：
  - **Coalescer Flush** → 產生一塊大的 **PinnedBuffer**（從 Pool 取得，並行寫入所有 request 的 slice）→ 呼叫 **run_dual_stream_pipeline**（自動切 Chunk、Overlap H2D/Compute）。
- 這樣 **Batching（上游聚合）** 與 **Pipelining（下游切分 + 重疊）** 完美串接；若改為呼叫舊的 sync `encode_batch`，會失去 Fix 4 的 overlap 效益。

### 3.7 參考對照表

| 來源 | 概念 | QDP 對應 |
|------|------|----------|
| Triton | max_queue_delay_microseconds | batch_wait_timeout（可選） |
| Triton | preferred_batch_size / max batch size | max_batch_samples |
| Triton | priority_levels | 可選，後續 |
| Triton core | scheduler / dynamic batcher（C++） | 見 [triton-inference-server/core](https://github.com/triton-inference-server/core) |
| tower-batch | buffer until max size or max duration | max_batch_* + batch_wait_timeout |
| tower-batch | batch Service::call | flush() 一次呼叫 encode_batch pipeline |
| Ray Serve | max_batch_size, batch_wait_timeout_s | max_batch_samples, batch_wait_timeout |
| Ray Serve | batch_size_fn | 我們用 bytes/samples 上限，等同自訂 batch 大小度量 |
| Ray Serve batching.py | _BatchQueue.wait_for_batch(), _process_batches() | 見 [ray/serve/batching.py](https://github.com/ray-project/ray/blob/master/python/ray/serve/batching.py) |
| §七之四 | max_batch_bytes / 內部拆批 | coalescer 的 max_batch_bytes 與 pipeline 內部拆批一致 |
| SMDP / Symphony 等論文 | 延遲–吞吐取捨、deferred batching | 首版顯式 flush；可選 timeout 或 policy 擴充 |

---

### 3.8 彈性調度器實作策略（Elastic Scheduler，建議採納）

在 Engine 內部實作一個微型的 Triton-style 調度器，可同時應用在 `encode()`（大單）與 `encode_batch()`（小單）。

**核心邏輯：雙門檻觸發（Dual Threshold Trigger）**（參考 Triton dynamic_batching）：

- **條件 A（Size Limit）**：累積的 bytes ≥ **target_batch_size**（例如 8 MB，對齊 Fix 4 Pipeline 的最佳吞吐）→ 馬上發車。
- **條件 B（Time Limit）**：第一筆請求進入後，經過 **max_delay**（建議 **0.1 ms ~ 1 ms**）→ 發車（Latency 保護）。
  - **參數敏感度**（參考 Triton / 業界）：Triton 預設 `max_queue_delay_microseconds: 0`，可設為非零（例如 100 μs）以換取 throughput；過小（&lt; 0.1 ms）易導致 CPU 頻繁醒來、Context Switch 開銷大；過大（&gt; 5 ms）對延遲敏感應用不友善。
- **OOM 保護**：累積 bytes 即將超過 **max_batch_size**（例如 64 MB）→ **先 flush 舊的再收新單**（無請求丟失，優於 Triton 的 reject）。
- **Head-of-Line Blocking**：若第一筆進來後遲遲無後續請求，timeout 到即發車；設計上「先 flush 舊的」可避免佇列無限堆積。

**建議的 BatchConfig（可與現有 CoalescerConfig 對齊或擴充）**：

- `target_batch_size: usize` — Pipeline 甜蜜點；**不要寫死**，應為 **Config**（使用者或自動偵測），**8 MB 為 Safe Default**（PCIe Gen4 約 ~0.3 ms H2D；Gen3/NVLink 環境可調，見 §3.11.3）。
- `max_batch_size: usize` — OOM 保護，超過則先 flush 再收新單。
- `max_delay: Duration` — 自第一筆進入後的最長等待時間，**建議 0.1 ms ~ 1 ms**，時間到即 flush。
- **調度迴圈實作**：建議使用 **crossbeam::select!** + timer channel，或 **recv_timeout** + `deadline.saturating_duration_since(now)`，避免 Deadline 計算錯誤（§3.11.4）。

**調度迴圈（Scheduler Thread）**：

1. Buffer 空時：`recv` 無限等待（或 `recv_timeout(MAX)`）收第一筆。
2. 收到第一筆時：設定 `deadline = now + max_delay`。
3. 若 **單筆請求 ≥ target_batch_size 且 buffer 為空**：**Pass-through**，直接執行該筆，不進 buffer（避免大單被無謂 copy 進 buffer）。
4. 否則：將請求加入 buffer，`current_bytes += task_size`。
5. 若 `current_bytes + 下一筆 > max_batch_size`：先 **flush_buffer**，再收新單。
6. 若 `current_bytes >= target_batch_size`：**flush_buffer**。
7. 若 **deadline 已到**：`recv_timeout(0)` 或檢查 timeout → **flush_buffer**。
8. 否則：`recv_timeout(deadline - now)` 收下一筆，回到 3。

**flush_buffer 的關鍵優化（記憶體管理）**：

- 計算總大小後，**一次分配**或 **從 Pinned Buffer Pool 取得**一塊 Pinned Memory，作為合併目標，減少 `Vec` 重分配與多餘搬運。
- 若 pipeline 內部未來支援「多段 slice 依序 H2D」（不強制連續 host buffer），可進一步省去 host 端 concatenation；目前為配合 Fix 4 的 Block Copy 效率，Host 端連續記憶體仍建議保留。

**結果分發**：每個 Task 帶一個 `tx: mpsc::Sender<Result<GpuStateVector, Error>>`（或對等 handle）；flush 完成後依邊界切出每個請求的結果，`task.tx.send(Ok(sub_tensor)).ok();`。

---

### 3.9 Engine 統一入口與 Python 非同步（建議採納）

**統一入口**：無論 `encode()`（1D）還是 `encode_batch()`（2D），本質上都是一塊 `f64` 資料。Engine 可持有 `sender: Sender<Task>`，兩者都改為：

- `encode(data)` → `submit_task(data)`
- `encode_batch(batch_data, num_samples, sample_size, ...)` → 將 batch 視為一個 Task（或 N 個 Single Task）`submit_task(...)`

**submit_task**：建構 `Task { data, num_qubits, tx }`，`sender.send(task)`，接著 `rx.recv()` 等待結果（blocking）。Scheduler 在獨立 thread 中執行，可持續收別人的單；Client 端 block 在自家 `rx.recv()` 不影響 Scheduler 收單。

**Python 卡死問題（Dynamic Batching 生效的關鍵）**：
若 Python 是單執行緒迴圈：

```python
for batch in batches:
    engine.encode(batch)  # 這裡會 Block 住 Python
```

則每次呼叫都會 block 到該筆完成，Scheduler **永遠收不到第二筆**，無法合併，Dynamic Batching 無效。

**解決方式**：必須提供 **非同步 / 射後不理** 介面，讓 Python 先把請求全部丟出，再統一收割結果。

- **方案一（encode_async）**：`encode_async(data)` 只負責把 Task 丟進 Rust Channel 並立即回傳一個 **Handle / Future**；Python 端：
  ```python
  futures = [engine.encode_async(batch) for batch in batches]
  results = [f.get() for f in futures]
  ```
  這樣 Scheduler 在 Python 建構 `futures` 時就已收到多筆，可合併後再執行，最後 `f.get()` 只負責按順序取回結果。**注意**：若 `batches` 數量很大（例如上千筆），會產生大量 Python Future 物件，開銷可觀；見 **方案三（encode_list）**。
- **方案二（Submit All）**：同上，`encode_async` 不等待結果，只丟 task 並回傳 future；實作可用 `pyo3_asyncio` 或簡單回傳一個 Handle，Python 在需要時再 `get()`。
- **方案三（encode_list，建議）**：`engine.encode_list(batches)` — **一次 FFI** 傳入多個 slice（例如 `List[ndarray]` 或 batch 邊界），Rust 內部一次收齊、切分或合併、走 Coalescer/Scheduler + pipeline，回傳 `List[Result]` 或等價結構。相比 N 次 encode_async + N 個 Future，**一次 encode_list** 大幅減少 FFI 與 Python 物件開銷，建議納入設計與實作優先級（§3.11.2）。

**小結**：
- 彈性調度器（雙門檻 + Pass-through + Scheduler thread + flush 時 Pinned Buffer 優化）與 Engine 統一入口（encode / encode_batch → submit_task）可一併採納。
- **Python 端必須支援「射後不理」**（encode_async + 延後 get），否則 Dynamic Batching 無法生效；這是與頂級開源方案行為一致的前提。

---

### 3.10 Review 結論與數據支撐（外部驗證）

本節納入針對「Zero-Copy Ingress + One-Copy to Pinned」與「Scope vs Scheduler」的 **Review 結論** 與參考數據，確保設計有據可依。

#### 3.10.1 Zero-Copy + One-Copy to Pinned 的效益驗證

- **為何必須這樣做**：
  - 參考 **DPDK** 與高效能網路框架：記憶體拷貝是吞吐殺手；DPDK 的經驗是「最佳 memcpy 是避免 memcpy；其次才是優化特定路徑」。
  - **單核 memcpy**：約 **10–15 GB/s**（視 SIMD 優化）；**PCIe Gen4 x16** 約 **26 GB/s**。若不並行 memcpy，Host 端 Copy 可能比 PCIe 慢，成為瓶頸。
  - **結論**：設計中 **並行 memcpy 寫入 Pinned Buffer**（例如 rayon）可逼近 CPU 記憶體頻寬，確保 Host 端 Copy 不拖累 PCIe，讓 GPU 不餓肚子。
- **參考**：[DPDK Writing Efficient Code](https://doc.dpdk.org/guides-25.11/prog_guide/writing_efficient_code.html)、[Intel DPDK memcpy optimization](https://www.intel.com/content/www/us/en/developer/articles/technical/performance-optimization-of-memcpy-in-dpdk.html)。

#### 3.10.2 Scope-based API 與 Python（GIL）的交互

- **Scope-based 在純 Rust 下**：生命週期由編譯器保證，安全且優雅。
- **在 Python（PyO3）下**：
  - Python 的 numpy array 由 GC 管理；`with engine.coalesce() as batcher:` 時須確保在 submit 後、flush 前 Python 不修改 array 內容（PyO3 `PyReadonlyArrayDyn` 通常能保證，但可能持有 GIL）。
  - **PyO3 不自動釋放 GIL**：須顯式使用 `Python::allow_threads()` 才能釋放；若在 Scope 內做完所有 submit + flush，則整段可能一直持有 GIL，**多執行緒 Python 呼叫會變成串行**。
  - **建議**：實作 Scope-based API 時，**同時保留 Scheduler Thread（Channel）方案**。對 Python 而言，「把資料丟給背景 Thread 後立刻釋放 GIL」通常比「在當前 Thread 持 GIL 做完所有事」有更高整體吞吐；若時間允許，可 **直接採用 Scheduler Thread（Actor Model）** 作為 Python 端主路徑，體驗接近原生 Async 且效能上限更高。
- **參考**：[PyO3 parallelism / allow_threads](https://pyo3.rs/v0.8.2/parallelism.html)、[PyO3 GIL release discussion](https://github.com/PyO3/pyo3/discussions/3621)。

#### 3.10.3 小結（設計採納）

- **Zero-Copy Ingress + One-Copy to Pinned**：數據與 DPDK 經驗支持；並行 memcpy 寫入 Pinned 為必要。
- **Time-Size 雙門檻**：max_delay 建議 **0.1 ms ~ 1 ms**；target_batch_size 8 MB 對 PCIe 合理；Head-of-Line 以「先 flush 舊的」處理，無請求丟失。
- **API 策略**：Scope-based 保留給 Rust 與生命週期安全；**Python 端優先考慮 Scheduler Thread（Channel）**，以釋放 GIL 並提高吞吐。

---

### 3.11 Final Reality Check（極限狀況與實作注意）

本節納入 **極限狀況沙盤推演** 與實作細節，確保設計在邊界情況下仍安全且可維護。參考：NVIDIA 論壇（Pinned Memory 多線程）、Intel/AMD 優化手冊（NT Store）、crossbeam select + timer、PCIe Gen3/NVLink 差異。

#### 3.11.1 Rayon 並行寫入 Pinned Memory 的可行性

- **並發寫入安全性**：
  - **查證結果**：`cudaHostAlloc`（Pinned Memory）在 Host 端為一般虛擬位址，僅由 OS 鎖頁；**只要寫入區間不重疊（Non-overlapping）**，多執行緒同時寫入是 **安全且高效** 的。各 thread 寫入自己專屬的 buffer 區段時，無需額外同步。
  - **行動建議**：
    - **按 Request 邊界切分**：`rayon` 的並行迭代應依 **Request 邊界** 切分（每個 Request 對應一個 offset + len），**不要按 byte 切分**，以避免 False Sharing（多 thread 寫入同一 cache line 導致無效化）。
    - **每個 Request 寫入專屬 offset**：flush 時預先算好每個 request 在 Pinned Buffer 中的 `(offset, len)`，再 `par_iter`  over 這些區段並行 copy，保證區間完全不重疊。
  - **可選優化**：大塊連續寫入可考慮 **Non-temporal Store（NT Store）** 繞過 CPU Cache 直接寫入 RAM，符合 Intel/AMD 優化手冊對高頻寬記憶體拷貝的建議；rayon 預設 memcpy 已能逼近頻寬，NT Store 可作為進階選項。
- **參考**：NVIDIA Developer Forums（pinned memory multi-thread non-overlapping）、[rayon ParallelSlice par_chunks](https://docs.rs/rayon/latest/rayon/slice/trait.ParallelSlice.html)。

#### 3.11.2 Scheduler Thread 與 Python 的極限狀況

- **Scheduler CPU 佔用**：
  - **情境**：若 Python 端以極高頻率發送極小請求（例如百萬次小 batch），Scheduler Thread 可能變成 CPU bound（忙於 recv、判斷、搬運），與 Python 主線程或其它 worker 搶 CPU。
  - **查證**：Triton 的 Scheduler 為獨立 C++ Thread；Rust `crossbeam::channel` 雖高效，高吞吐下仍有 overhead。
  - **行動建議**：
    - **Backpressure**：使用 **bounded channel**（有界佇列）；Channel 滿時可讓發送端 **block**（`send()` 阻塞直到有空間）或 **非阻塞** 回傳「佇列滿」（例如 `try_send()` 回傳 `Err(TrySendError::Full)`），由 Python 端 Backoff 或重試，避免 Scheduler 被海量請求淹沒。
    - **Thread Priority**（可選）：若 OS 允許，可為 Scheduler Thread 設定優先級，依部署需求調整。
  - **QDP 場景**：實際負載以 H2D + Kernel 為主，**GPU 通常會先飽和**，Scheduler 成為 CPU bound 的機率較低；上述為防禦性設計。
  - **參考**：Rust bounded channel 下 `send()` 會 block 直到有空間、`try_send()` 回傳 `Full` 可讓呼叫端 Backoff；[crossbeam Sender](https://docs.rs/crossbeam/latest/crossbeam/channel/struct.Sender.html)、[Comprehensive Rust: bounded channels](https://google.github.io/comprehensive-rust/concurrency/channels/bounded.html)。

- **Python Future 開銷**：
  - **情境**：`futures = [engine.encode_async(b) for b in batches]` 若 `batches` 有上千筆，會產生上千個 Python Future 物件，建立與管理開銷可觀。
  - **行動建議**：再次驗證 **Batch API（encode_list）** 的重要性。
  - **encode_list**：`engine.encode_list(batches)` → **一次 FFI** 傳入多個 slice（例如 `Vec<&[f64]>` 或 batch 邊界），Rust 內部一次收齊、切分或合併、走 pipeline，回傳 `Vec<Result<...>>`。相比「N 次 encode_async + N 個 Future」，**一次 encode_list** 大幅減少 FFI 與 Python 物件開銷，建議納入設計與實作優先級。

#### 3.11.3 target_batch_size（8MB）的動態調整

- **硬體差異**：
  - **PCIe Gen4**：8 MB 約 ~0.3 ms H2D，為當前設計的甜蜜點。
  - **PCIe Gen3**：頻寬約減半，8 MB 傳輸時間約 ~0.6 ms；若 Kernel 很快，Pipeline 可能填充不足。
  - **NVLink（如 Grace Hopper）**：頻寬極高（~900 GB/s），H2D 幾乎非瓶頸，甜蜜點可能更大或無需特別調校。
- **行動建議**：
  - **不要寫死**：`target_batch_size` 應為 **Config**（例如 `CoalescerConfig` / `PipelineConfig`），可由使用者設定或依硬體自動偵測（PCIe 代數、是否 NVLink）。
  - **Safe Default**：保留 **8 MB** 作為預設值，在大多數現代 Server GPU（PCIe Gen4）上合理；Gen3 或 NVLink 環境可透過 config 調整。

#### 3.11.4 彈性調度迴圈的 Rust 實作模式

- **需求**：Scheduler 迴圈需同時處理「新請求到達」與「Timeout 到期」，且 Deadline 計算容易出錯。
- **建議模式一（crossbeam::select! + Timer）**：
  - 使用 **timer channel** 在 `select!` 中同時 `recv(rx)` 與 `recv(timer)`。
  - **One-shot 建議用 `after(duration)`**：自第一筆請求到達起等待 `max_delay` 後只觸發一次 → 用 **`crossbeam_channel::after(Duration::from_micros(max_delay_us))`** 建立 **one-shot** timer；**不要用 `tick(duration)`**（tick 為週期性，會重複觸發）。每輪 flush 後若需再等下一批，再建立新的 `after(max_delay)`。
  - `recv(rx) -> req`：新請求入 buffer、檢查是否滿或達 target → flush；若為第一筆則建立/重置 `after(max_delay)`；
  - `recv(timer) -> _`：Timeout 到期 → flush。
  - 邏輯清晰，Deadline 由 timer 負責，不易寫錯。
- **建議模式二（recv_timeout）**：
  - 若不用 `select!`，可用 **`recv_timeout(deadline.saturating_duration_since(now))`**；務必使用 **saturating_duration_since** 避免 underflow，且每次迴圈重新計算剩餘時間。
- **參考**：[crossbeam after (one-shot)](https://docs.rs/crossbeam/latest/crossbeam/channel/fn.after.html)、[crossbeam tick (periodic)](https://docs.rs/crossbeam/latest/crossbeam/channel/fn.tick.html)、[Crossbeam select with recv_timeout / timer](https://users.rust-lang.org/t/crossbeam-select-with-recv-timeout/78011)。

#### 3.11.5 小結（Final Reality Check）

- **Pinned 並行寫入**：非重疊區間安全；按 **Request 邊界** 切分、每 Request 專屬 offset，避免 False Sharing；可選 NT Store。
- **Scheduler**：Channel 滿時 Backpressure（bounded channel + block 或 try_send）；**encode_list**（一次 FFI 收多個 slice）建議納入設計，減少 Python Future 與 FFI 開銷。
- **target_batch_size**：Config、不寫死；8 MB 為 Safe Default；Gen3/NVLink 可調。
- **調度迴圈**：`crossbeam::select!` + timer channel 或 `recv_timeout` + `saturating_duration_since(now)`，避免 Deadline 計算錯誤。

#### 3.11.6 實作時要注意的地方（從頭順一遍檢查）

以下為從頭順過設計後整理的 **實作時注意事項**，避免邊界與一致性問題。

- **空佇列 flush**：Coalescer 在 queue 為空時呼叫 `flush()` 應回傳 **Err**（例如 "flush called with empty queue"），與現有 coalescer.rs 行為一致；Scheduler 端若 timeout 時 buffer 為空則不呼叫 flush、或約定「空 flush」為 no-op 並回傳空結果，需在規格中二擇一。
- **結果順序**：**encode_list**、**encode_async** 回傳的結果順序必須與 **提交順序一致**（依邊界切出後依序分發給各 Task 的 `tx`）。與 **Triton preserve_ordering**、**Ray Serve**（`future.set_result()` 依 batch 內順序）一致；我們依 boundaries 切出後依序分發即為 preserve order，實作時勿打亂順序。
- **flush 失敗時**：若 pipeline 或 encode_batch 回傳 Err，同一批合併的請求應 **依序對每個 Task 的 tx 送 Err**（或同一錯誤），讓呼叫端能區分是哪一批失敗；避免只回傳一個 Err 而其他 Task 永遠等不到結果。
- **Config 來源一致**：**CoalescerConfig**（target_batch_size、max_batch_size、max_delay）與 **PipelineConfig**（chunk_size、pool_size）若由同一層（例如 Engine）讀取或建構，應避免兩邊數值衝突（例如 target_batch_size 遠大於單次 pipeline 可處理的 chunk 總量）；建議 target_batch_size 與 pipeline 的 chunk 上下界對齊或由同一 config 推導。
- **Backpressure 實作**：Scheduler 的接收 channel 建議為 **bounded**；capacity 依部署負載可調，建議納入 Config。發送端用 `send()` 則自動 block（背壓），或用 `try_send()` 回傳 `Full` 讓 Python 端重試／回傳「佇列滿」。
- **單一 request 的 sample_size 與 num_qubits/encoding**：同一 coalescer 內所有 request 的 **sample_size** 必須一致（§3.2）；**num_qubits / encoding** 亦須一致方能合併。submit 時驗證並回傳 Err 若不一致，與現有 coalescer 行為一致。
- **Pinned Pool 單塊不足時**：若 flush 總大小超過 **單一 Pinned Buffer 的容量**（Pool 每塊固定大小），需在規格中約定：**方案 A** 從 Pool 取多塊並行寫入後再依序送 pipeline（需 pipeline 支援多段輸入或先 host 端合併）、**方案 B** 限制單次 flush 總大小 ≤ 單塊容量（submit 時若 would_exceed 即先 flush 再收）、**方案 C** 回傳 Err 要求呼叫端拆小。建議首版採用 **方案 B**（max_batch_bytes 與 Pool 單塊容量對齊），避免實作複雜度。

#### 3.11.7 盲區檢查（第二遍順過）

以下為 **第二遍順過** 時整理的潛在盲區，確保設計無遺漏、實作時有據可依。

- **Scope / callback panic**：若 **run_coalesced** 的 callback 在 **flush 前** panic，佇列內已存的 `&[f64]` 不會被 flush，生命週期隨 stack unwind 結束，**無需額外 copy**。若 flush 已執行且已從 Pool **acquire** 取得 Pinned Buffer、隨後 pipeline 或分發階段 panic，須保證 **Pinned Buffer 仍歸還 Pool**（例如用 RAII：PinnedBufferHandle 在 Drop 時歸還；panic 時 Rust 會執行 Drop）。實作時 **flush 路徑** 以「取得 handle → 寫入 → pipeline → 切出分發 → drop handle」順序撰寫，即可在任一階段 panic 時透過 Drop 歸還 Pool。
- **Pool 耗盡**：當 **pool.acquire(Total_N)** 時若 Pool 內無可用 buffer（例如所有 buffer 皆在 pipeline 使用中），可 **block 等待** 或 **回傳 Err**。若 Scheduler 與 pipeline 共用同一 Pool，須避免 **死鎖**（例如 Scheduler 等 acquire、pipeline 等 Scheduler 送下一批）。建議：**並行 flush 數 ≤ pool_size**（一次只有一個 Scheduler 在 flush_buffer，且 flush 完成並歸還 buffer 後才收下一批），或 acquire 帶 **timeout** 回傳 Err 讓呼叫端重試。
- **encode_list 與 Scheduler 關係**：**encode_list(batches)** 可實作為（1）**專用路徑**：一次將多筆 slice 交給 Coalescer、flush 一次、回傳 `Vec<Result>`，不經時間觸發的 Scheduler；或（2）**同一 Scheduler**：每筆 batch 當作一個 Task 送進 Scheduler channel，由 Scheduler 合併後 flush。兩者皆可；專用路徑延遲更低、Scheduler 路徑與 encode_async 一致。實作時二擇一或並存（例如 encode_list 走專用 Coalescer flush，不經 max_delay）。
- **Pinned Buffer 歸還時機**：**必須在 pipeline 完成後**（不論 Ok 或 Err）歸還。流程：取得 handle → 寫入 → **run_dual_stream_pipeline**（block 至完成）→ 依邊界切出 → 分發給各 `tx` → **drop(handle)** 觸發歸還 Pool。若 pipeline 回傳 **Err**，仍須 **drop(handle)** 歸還，再對該批所有 Task 的 tx 送 Err。
- **Reentrancy**：**flush** 不應在 **flush 內部** 被呼叫（例如 pipeline 內層又觸發 coalescer），否則佇列與 Pool 狀態易錯。Scheduler 為 **單一 consumer**、一次只執行一個 flush_buffer，自然滿足；Scope-based 為單 thread 依序 submit 後一次 flush，亦滿足。若未來有多入口，須以鎖或單一 flush 執行緒保證不重入。
- **單筆請求超過 max_batch_bytes／單塊容量**：若 **單一 request** 的 bytes 就超過 max_batch_bytes 或 Pool 單塊容量，會落入「先 flush 再收」仍無法容納。規格須二擇一：**拒絕**（submit 回傳 Err，與 Ray Serve 的 batch_size_fn 單筆超限時 `future.set_exception` 對應）、或 **Pass-through 單獨執行**（該筆不進 buffer，直接走 pipeline；若 pipeline 支援內部拆批則可處理，否則需多塊 Pinned 或 Err）。實作時與「Pinned Pool 單塊不足」策略一致並寫明。
- **Observability（可選）**：Ray Serve 有 batch wait time、batch execution time、queue length、batch utilization；我們可選實作 **佇列長度**、**單次 flush 的 sample 數／bytes**、**flush 耗時** 等 metrics，供調參與監控。非首版必要，但建議預留介面。

---

## 四、小結

- **整體邏輯順序**：入口（encode/encode_batch → submit 或 run_coalesced）→ 佇列只存 metadata（Zero-Copy Ingress）→ 觸發（max_batch_* 或 flush）→ flush 寫入 Pinned Pool（One-Copy to Pinned）→ run_dual_stream_pipeline → 依邊界切回；Engine 為 Float32 時對結果 to_precision。與現有程式碼對應：coalescer.rs 現為 Vec<f64>，計畫改為 EncodeRequest<'a>(&[f64])、flush 走 Pinned Pool + pipeline；encode_batch 大 batch 需走 pipeline（§七之一、§七之二）；首版 Pipeline/Pool 維持 f64。
- **參考**：Triton（delay + priority + queue policy）、tower-batch（size + duration → 合併 batch）、Ray Serve（max_batch_size + timeout + batch_size_fn，見 `batching.py`）都採用「佇列 + 雙條件觸發 + 一次執行」的 Request Coalescing 模式；相關論文支持延遲–吞吐取捨與以有效 batch 大小防 OOM。
- **改進**：我們以 **bytes/samples 上限** 為主、與 §七之四 伸縮調度一致，並讓 **encode() 與 encode_batch() 共用同一 coalescer**；**效能底線**為 **Zero-Copy Ingress + One-Copy to Pinned**（§3.5.1、§3.5.2、§3.6 修正版），避免兩次 Host Copy 抵銷 PCIe Pipeline 優勢；API 建議採 **Scope-based / Callback**（方案 C）以配合生命週期；進階可採 Pinned Memory Aware Batcher、彈性調度器與 Python encode_async。
- **實作**：現有 `EncodeCoalescer`（`coalescer.rs`）可演進為 **EncodeRequest<'a>**（`&'a [f64]`）、**無內部 Vec buffer**、**flush_into_pinned_buffer**（從 Pinned Buffer Pool 取一塊，並行寫入所有 slice，再呼叫 **run_dual_stream_pipeline**）；API 新增 **run_coalesced(callback)**，Python 對應 **`with engine.coalesce() as batcher:`**。
- **與 Pipeline 的整合**：Coalescer flush 必須呼叫 **Fix 4 的 run_dual_stream_pipeline**（Chunk + Overlap H2D/Compute），不呼叫舊的 sync encode_batch，才能串接 Batching（上游聚合）與 Pipelining（下游切分）。
- **與 PR3 優化總結的關係**：見 **PR3_OPTIMIZATION_SUMMARY_AND_OVERSIGHT.md** 的「七之五、Request Coalescing」與文件索引。

### 總結優化清單（建議採納）

1. **Zero-Copy Coalescer**：**EncodeRequest<'a>** 持有 `&'a [f64]`（或精度泛型），Coalescer 只存 metadata，**不**維護內部 Vec；**flush** 時從 **Pinned Buffer Pool** 取得一塊，**並行 memcpy** 寫入所有 slice，再送 **run_dual_stream_pipeline** → **Zero-Copy Ingress + One-Copy to Pinned**（§3.10.1）。
2. **Scope-based API**：**run_coalesced(callback)**；Python **`with engine.coalesce() as batcher:`**，離開 Scope 時自動 flush；實作時 **保留 Scheduler Thread 方案**（§3.10.2）。
3. **Scheduler Thread**（建議、尤其 Python）：Time + Size 雙門檻、獨立 thread、Task(tx)、flush 時 Pinned Buffer；**Python 端優先採用**可釋放 GIL、提高吞吐。
4. **統一 API 入口**：encode / encode_batch 經同一 Coalescer 或 Scheduler，合併後一次執行。
5. **Pass-through**：單筆 ≥ target_batch_size 且 buffer 空時直接執行，不進 buffer。
6. **Python Async**：encode_async + 延後 get，讓 Dynamic Batching 生效。
7. **encode_list**（§3.11.2）：一次 FFI 收多個 slice、回傳 `Vec<Result>`，減少 Python Future 與 FFI 開銷；建議納入設計與實作優先級。
8. **Float32 預設存在**（§五）：設計假定 Float32 路徑存在；EncodeRequest / Pool / Pipeline 為 precision-aware 或可擴充 f32；首版可只實作 f64，型別與介面預留 f32。未來可做 **early conversion**（寫入 Pinned 時 f64→f32）省 PCIe 與 GPU 算力。
9. **實作注意**（§3.11）：flush 並行寫入按 **Request 邊界** 切分、非重疊；target_batch_size 為 **Config**（8 MB 預設）；Scheduler 滿時 **Backpressure**（bounded channel / try_send）；調度迴圈 **one-shot 用 crossbeam `after(duration)`**（不用 tick）、或 **recv_timeout + saturating_duration_since**。**要注意的地方**（§3.11.6）：空佇列 flush、結果順序（preserve order）、flush 失敗分發、Config 來源一致、**sample_size 與 num_qubits/encoding 一致**、**Pinned Pool 單塊不足**（max_batch_bytes 與單塊容量對齊）。**盲區檢查**（§3.11.7）：Scope panic 時 RAII 歸還 Pinned、Pool 耗盡與死鎖防範、**單筆超過 max_batch_bytes／單塊容量**（拒絕或 Pass-through + pipeline 內部拆批）、encode_list 與 Scheduler 關係、Pinned 歸還時機、flush 不重入、Observability 可選。

---

## 五、Float32 預設存在（設計假定）

本設計 **從一開始就假定 Float32 路徑存在**：API 形狀、EncodeRequest、Pinned Buffer Pool、Pipeline 皆為 **precision-aware** 或可擴充為 f32，首版實作可只做 f64 分支，但型別與介面不寫死 f64，以便未來無痛接上 f32。

### 5.1 為什麼要預設存在？

- **現況**：輸入目前為 f64；輸出可為 Float32，由 Engine 的 `to_precision(device, Precision::Float32)` 在 encode 後轉換。
- **隱藏浪費**（Review 指出）：若流程為「輸入 f64 → Pipeline 全 f64 → 輸出再 to_precision(f32)」，則 **GPU 端全程 f64**：算力較 f32 慢（消費級約 32×、H100/A100 約 2×），State Vector 頻寬為 f32 的兩倍。
- **未來優化**：若目標為 f32 輸出且輸入可接受 f32，應 **盡早轉型**；即使輸入為 f64，若精度允許，可在 **寫入 Pinned Buffer 時** 轉成 f32（Copy 階段 f64→f32），PCIe 頻寬省一半、GPU 走 f32 pipeline。
- **設計原則**：EncodeRequest、Pool、Pipeline 在設計階段就區分或泛型化 **輸入／輸出精度**，避免日後重構。

### 5.2 設計上的預設（型別與介面）

- **EncodeRequest**：設計為可區分精度，例如 `Single(&[f64])` / `SingleF32(&[f32])` 或泛型 `EncodeRequest<'a, T: Float>`；同一 flush 內同型別。首版可只實作 `Single(&[f64])` / `Batch { data: &[f64], ... }`，但型別名稱與模組邊界預留 f32。
- **Pinned Buffer Pool**：設計為可依精度取得 buffer（例如 `Pool::acquire_f64(n)` / `Pool::acquire_f32(n)` 或泛型 `acquire<T>()`）；首版可僅實作 f64。
- **Pipeline**：設計為可接受 f64 或 f32 的 host buffer，並呼叫對應 kernel（`launch_*` / `launch_*_f32`）；首版可僅實作 f64 路徑。
- **Early conversion（未來）**：若 Engine 輸出精度為 Float32 且呼叫端允許，flush 寫入 Pinned Buffer 時可選「f64→f32」再送 f32 pipeline，省 PCIe 與 GPU 算力。

### 5.3 與 Coalescing 的對應

- **首版**：Coalescer 僅接受 f64 輸入；flush 寫入 f64 Pinned Buffer，走 f64 pipeline；輸出仍經 Engine 的 to_precision(Float32/Float64)。
- **未來**：同一套 Coalescer 可接受 f32 請求或「f64 輸入 + 寫入時轉 f32」；Pool/Pipeline 啟用 f32 分支後即可生效，無需改動 Coalescer 整體架構。

---

## 六、小結

- **參考**：Triton（delay + priority + queue policy）、tower-batch（size + duration → 合併 batch）、Ray Serve（max_batch_size + timeout + batch_size_fn，見 `batching.py`）都採用「佇列 + 雙條件觸發 + 一次執行」的 Request Coalescing 模式；相關論文支持延遲–吞吐取捨與以有效 batch 大小防 OOM。
- **改進**：我們以 **bytes/samples 上限** 為主、與 §七之四 伸縮調度一致，並讓 **encode() 與 encode_batch() 共用同一 coalescer**；**效能底線**為 **Zero-Copy Ingress + One-Copy to Pinned**（§3.5.1、§3.5.2、§3.6 修正版），避免兩次 Host Copy 抵銷 PCIe Pipeline 優勢；API 建議採 **Scope-based / Callback**（方案 C）以配合生命週期；進階可採 Pinned Memory Aware Batcher、彈性調度器與 Python encode_async。
- **實作**：現有 `EncodeCoalescer`（`coalescer.rs`）可演進為 **EncodeRequest<'a>**（`&'a [f64]`）、**無內部 Vec buffer**、**flush_into_pinned_buffer**（從 Pinned Buffer Pool 取一塊，並行寫入所有 slice，再呼叫 **run_dual_stream_pipeline**）；API 新增 **run_coalesced(callback)**，Python 對應 **`with engine.coalesce() as batcher:`**。
- **與 Pipeline 的整合**：Coalescer flush 必須呼叫 **Fix 4 的 run_dual_stream_pipeline**（Chunk + Overlap H2D/Compute），不呼叫舊的 sync encode_batch，才能串接 Batching（上游聚合）與 Pipelining（下游切分）。
- **與 PR3 優化總結的關係**：見 **PR3_OPTIMIZATION_SUMMARY_AND_OVERSIGHT.md** 的「七之五、Request Coalescing」與文件索引。
- **Float32**：**設計上 Float32 預設存在**（§五）；輸入目前皆 f64，輸出經 Engine to_precision；型別與介面預留 f32，未來可做 early conversion 與 f32 pipeline。
- **結論（Final Reality Check）**：本設計（Revision with Zero-Copy & Scheduler）已涵蓋：（1）**PCIe 頻寬利用率**（Pinned Memory + Pipeline）、（2）**CPU/Host 瓶頸**（Parallel Memcpy + Zero-Copy Ingress）、（3）**Python 交互瓶頸**（Async / Scheduler Thread 解鎖 GIL、encode_list 減少 FFI）、（4）**動態負載適應**（Time-Size 雙門檻、Backpressure、Config 化 target_batch_size）。極限狀況與實作細節見 §3.11。**可基於本文件進行實作。**

### 總結優化清單（建議採納）

1. **Zero-Copy Coalescer**：**EncodeRequest<'a>** 持有 `&'a [f64]`（或精度泛型），Coalescer 只存 metadata；**flush** 時從 **Pinned Buffer Pool** 取得一塊，**並行 memcpy** 寫入，再送 **run_dual_stream_pipeline**（§3.10.1）。
2. **Scope-based API**：**run_coalesced(callback)**；實作時 **保留 Scheduler Thread 方案**（§3.10.2）。
3. **Scheduler Thread**（建議、尤其 Python）：雙門檻、獨立 thread、Task(tx)；**Python 端優先採用**以釋放 GIL。
4. **統一 API 入口**：encode / encode_batch 經同一 Coalescer 或 Scheduler。
5. **Pass-through**：單筆 ≥ target_batch_size 且 buffer 空時直接執行。
6. **Python Async**：encode_async + 延後 get。
7. **encode_list**（§3.11.2）：一次 FFI 收多個 slice、回傳 `Vec<Result>`，減少 Future 與 FFI 開銷。
8. **Float32 預設存在**（§五）：設計假定 f32 路徑存在；型別與介面 precision-aware；未來 early conversion 省 PCIe 與 GPU 算力。
9. **實作注意**（§3.11）：並行寫入按 Request 邊界、非重疊；target_batch_size 為 Config；Scheduler Backpressure；調度 **after() one-shot** 或 recv_timeout。**要注意的地方**（§3.11.6）：空佇列 flush、結果順序、flush 失敗分發、Config 一致、**sample_size 與 num_qubits/encoding 一致**、Pinned Pool 單塊不足。**盲區檢查**（§3.11.7）：Scope panic 時 RAII 歸還、Pool 耗盡與死鎖、**單筆超過 max_batch_bytes／單塊容量**、encode_list 與 Scheduler、Pinned 歸還時機、flush 不重入、Observability 可選。
