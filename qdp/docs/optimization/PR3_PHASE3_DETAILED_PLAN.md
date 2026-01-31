# PR3 第三階段詳細計劃：C3 動態參數與安全調優

## 文檔資訊

- **版本**: 1.5
- **日期**: 2026-01-30
- **對應**: PR3_COMPLETE_IMPLEMENTATION_PLAN.md 階段 3
- **依賴**: 階段 1（可觀測性）已完成；可與階段 2 並行，建議在階段 4 前合入
- **目標**: 新增 `PipelineConfig` 與硬體感知的 chunk size / pool size，透過環境變數可覆蓋，並做安全驗證（pinned 記憶體上限等）
- **大前提**: **效能優化優先**——預設值與驗證規則皆以最大化 H2D overlap、減少 stall、適配 PCIe/GPU 頻寬為依據。
- **程式碼對照**: 已依 2026-01-30 專案程式碼核對；`pipeline_config.rs` 尚未存在，需新增；`pipeline.rs`、`cuda_ffi.rs` 行號與內容已對齊。`CudaDevice::ordinal()` 已於 `qdp-core/src/gpu/memory.rs` 使用（GpuStateVector 等），可於 pipeline_config 以 `device.ordinal()` 作為 `cudaDeviceGetAttribute(..., device_id)` 之 device id。`gpu/mod.rs` 尚未宣告 `pipeline_config`，新增時需加 `#[cfg(target_os = "linux")] pub mod pipeline_config;`。Workspace 使用 **cudarc 0.13**（qdp/Cargo.toml），qdp-core 依賴 workspace 之 cudarc。Pipeline 內 TODO「tune dynamically based on GPU/PCIe bandwidth」（約 364 行）即由本階段之 PipelineConfig 實作。

---

## 零、效能優化優先原則（Performance-First）

本階段之設計與預設值均以「最大化 pipeline 吞吐、H2D overlap、減少 copy stream 等待」為前提。

### 0.1 為何要動態 chunk size？

- **PCIe 頻寬主導 H2D 時間**：實測 PCIe 頻寬遠低於 GPU 記憶體頻寬（約 5–16 GB/s vs 數百 GB/s），見 [NVIDIA 資料傳輸優化](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)。
- **小傳輸效率差**：過小 chunk 會放大每筆傳輸的固定開銷（TLP、封包頭），見 [CUDA 論壇：小傳輸吞吐](https://forums.developer.nvidia.com/t/why-is-the-transfer-throughput-low-when-transferring-small-size-data-from-host-to-device-or-device-to-host/153962)。
- **過大 chunk 不利 overlap**：單次 H2D 時間過長會拉長「等 copy 完成」的窗口，不利與 compute 重疊；且需更多 pinned 記憶體。
- **結論**：依 PCIe 代數與 GPU 選擇「夠大以吃滿頻寬、夠小以利 overlap」的 chunk（見預設值表），可望帶來 **10–40% 吞吐提升**（硬體相關）。

### 0.2 為何要動態 pool size？

- **同向 H2D 在 PCIe 上串行**：多個 `cudaMemcpyAsync` 同方向仍會排隊，見 [CUDA Programming Guide 4.11 Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。
- **足夠 buffer 才能 overlap**：copy stream 在傳 chunk N+1 時，compute stream 在算 chunk N；至少需 2 個 slot，高頻寬下 3–4 個可減少「等 buffer」的機率。
- **20% 主機記憶體上限**：過多 pinned 會影響系統與 swap，見 [CUDA Best Practices - Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)；驗證時強制遵守。

### 0.3 參考資料（效能面向）

| 主題 | 要點 | 連結 |
|------|------|------|
| Host-Device 傳輸優化 | Pinned memory、批次傳輸、非同步複製 | https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/ |
| 異步複製與 overlap | Memcpy-kernel overlap、同向 H2D 串行 | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html |
| PCIe Gen3/Gen4 頻寬 | 8 GT/s → ~1 GB/s/lane, 16 GT/s → ~2 GB/s/lane（編碼後） | https://www.passmark.com/products/pcie-gen4-test-card/practical-throughput.php |
| CUDA 記憶體最佳實踐 | Pinned 使用、20% 建議 | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations |
| cudaDeviceGetPCIBusId | 取得 GPU 對應 PCI BDF，用於 sysfs 查鏈路速度 | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html |

### 0.4 更多效能參考（補充查證）

- **Overlap 實作**: NVIDIA 官方部落格 [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) 說明雙 stream + pinned 以達成 copy 與 compute 重疊；至少需 2 個 buffer，高頻寬下 3–4 個可減少「等 buffer」。
- **傳輸大小**: Pinned 緩衝建議僅用於**大於 64 KB** 的非同步傳輸；過小會放大每筆 TLP/封包開銷。多筆小傳輸應合併為單一大傳輸以提升吞吐。
- **PCIe 實測**: PCIe Gen4 x16 理論約 32 GB/s，實測 H2D 約 **5–11 GB/s**（多數 GPU 約 5–11 GB/s；高階卡可達 10–12 GB/s）。Gen3 約 5–7 GB/s（pinned）。小 chunk 導致 payload 相對 28 位元組 TLP 開銷比例差，大 chunk 較利頻寬利用。
- **批次原則**: 文獻建議**盡量減少傳輸次數**，以單次較大傳輸取代多次小傳輸；結合 pinned + 非同步 stream 可同時拉高頻寬與 overlap。
- **Pinned 上限**: 文獻建議 pinned 總量**低於主機記憶體約 20%**，超過 50–60% 可能影響系統或失敗；傳輸完成後應釋放或重用，避免長期佔用。
- **cudaDeviceGetAttribute**: 簽名 `cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)`；Compute Capability 使用 `cudaDevAttrComputeCapabilityMajor = 75`、`cudaDevAttrComputeCapabilityMinor = 76`（見 CUDART Types 官方列舉）。
- **GpuOverlap / 非同步引擎**: `cudaDevAttrGpuOverlap = 15` 表示裝置是否支援 copy 與 kernel 並行；`cudaDevAttrAsyncEngineCount = 40` 為非同步引擎數量。多數 dGPU 支援 overlap；若查詢失敗或為 0，動態 chunk/pool 仍可改善頻寬利用。
- **Pinned 與 OS 限制**: Linux 下 pinned 總量還受 `ulimit -l`（memlock）限制；單次過大（如 >1–4 GB）可能失敗。20% 主機記憶體為保守上限，兼顧系統穩定性與效能。
- **Chunk 與 overlap 取捨**: 傳輸與 kernel 皆佔用 GPU 記憶體頻寬；kernel 若為 memory-bound，過度 overlap 可能使兩者變慢。chunk 夠大以攤銷固定開銷、夠小以利 copy 與 compute 時間重疊，與預設值表設計一致。文獻指出約 10 MB 級傳輸在高頻寬裝置上約 1–2 ms 級，適合做 overlap 單元。

---

## 一、現有程式碼剖析

### 1.1 Pipeline 入口與常數（需改為動態）

**檔案**: `qdp/qdp-core/src/gpu/pipeline.rs`（已依目前專案程式碼對照）

| 行號 | 當前程式碼 | 說明 |
|------|------------|------|
| 267 | `const CHUNK_SIZE_ELEMENTS: usize = 8*1024*1024/size_of::<f64>();` | 公開 API 固定 8MB chunk |
| 268-271 | `run_dual_stream_pipeline_with_chunk_size(..., CHUNK_SIZE_ELEMENTS, ...)` | 未傳 pool_size |
| 305-309 | `BASE_CHUNK_SIZE_ELEMENTS` 8MB；對齊後 `chunk_size_elements` | `run_dual_stream_pipeline_aligned` 依對齊調整 |
| 312-316 | `run_dual_stream_pipeline_with_chunk_size(..., chunk_size_elements, ...)` | 未傳 pool_size |
| 321-324 | `run_dual_stream_pipeline_with_chunk_size(device, host_data, chunk_size_elements, kernel_launcher)` | 簽名無 pool_size |
| 337 | `const PINNED_POOL_SIZE: usize = 2;` | 內部固定 pool=2 |
| 348-349 | `PipelineContext::new(device, PINNED_POOL_SIZE)`；`PinnedBufferPool::new(PINNED_POOL_SIZE, chunk_size_elements)` | 建立 context 與 pool |
| 359 | `OverlapTracker::new(PINNED_POOL_SIZE, true)` | 可觀測性依 pool_size |
| 380 | `let event_slot = chunk_idx % PINNED_POOL_SIZE;` | slot 計算 |
| 446-452 | `if in_flight_pinned.len() == PINNED_POOL_SIZE { ctx.sync_copy_stream()?; in_flight_pinned.clear(); }` | 滿池時定期同步（階段 4 將改為事件驅動） |

**結論**: 需在呼叫 `run_dual_stream_pipeline_with_chunk_size` 前取得 `(chunk_size_elements, pool_size)`，並改為使用單一 `pool_size` 參數（來自 config）；簽名改為傳入 `pool_size`。

### 1.2 呼叫鏈（保持對外 API 不變）

- **Amplitude**: `run_dual_stream_pipeline(device, host_data, kernel_launcher)`（約 332 行）
- **Angle**: `run_dual_stream_pipeline_aligned(device, host_data, align_elements, kernel_launcher)`（約 270 行）
- **對外**: `mod.rs` 僅 re-export `run_dual_stream_pipeline`

**設計**: 不變更對外函式簽名；在兩者內部以 `PipelineConfig::from_env().with_hardware_defaults(...)` 並 `validate()`，用 `config.chunk_size_elements()`、`config.pinned_pool_size_resolved()` 取得參數，再傳入 `run_dual_stream_pipeline_with_chunk_size(..., chunk_size_elements, pool_size, kernel_launcher)`。

### 1.3 Buffer Pool 與 Context API（已滿足）

- `PinnedBufferPool::new(pool_size, elements_per_buffer)`（buffer_pool.rs:79）
- `PipelineContext::new(device, event_slots)`（pipeline.rs:66）
- `OverlapTracker::new(pool_size, enabled)` 已接受 `pool_size`

### 1.4 專案依賴與 FFI

- **Cargo.toml**: qdp-core 依賴 cudarc 0.13（workspace），無 sysinfo。
- **cuda_ffi.rs**: 已有 Event/Stream；**無** `cudaDeviceGetAttribute`。GPU 計算能力需透過 CUDA Runtime FFI 取得。
- **錯誤型別**: `MahoutError::InvalidInput(String)`、`MahoutError::Cuda(String)` 已足供 config 驗證使用。

---

## 二、官方文件與參考

### 2.1 GPU 計算能力（Compute Capability）

- **CUDA Runtime API**: cudaDeviceGetAttribute
  `cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)`
  文件: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
- **屬性列舉**: cudaDevAttrComputeCapabilityMajor = 75, Minor = 76
  文件: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
- **對應架構**: (9,0) Hopper; (8,9) Ada; (8,0)/(8,6) Ampere; 其餘 Unknown
  文件: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#compute-capabilities
- **cudarc**: 專案使用 cudarc 0.13；若無公開 compute_capability()，則在 cuda_ffi.rs 新增 cudaDeviceGetAttribute，以 device.ordinal() 作為 device id。

### 2.2 PCIe 代數（主機與 GPU 間頻寬）

- **Linux sysfs**: 路徑 `/sys/bus/pci/devices/<BDF>/` 或 `/sys/devices/pci*/*/`；BDF 可經 `cudaDeviceGetPCIBusId` 取得後對應至 sysfs 目錄。
  文件: https://docs.kernel.org/PCI/sysfs-pci.html
- **鏈路速度**: `current_link_speed`、`max_link_speed` 報告 GT/s（字串如 `"8.0 GT/s"`、`"16.0 GT/s"`、`"32.0 GT/s"`）。
  對應: 8.0 GT/s → Gen3, 16.0 GT/s → Gen4, 32.0 GT/s → Gen5（較新核心支援 32 GT/s 報告）。
  若找不到 GPU BDF 或檔案不存在，回退 `PCIeGeneration::Unknown`。
- **環境變數**: `QDP_PCIE_GEN`（3/4/5 或 gen3/gen4/gen5）覆蓋自動檢測。

### 2.3 主機記憶體（Pinned 上限 20%）

- **Linux**: 讀取 `/proc/meminfo` 的 `MemTotal:`（KB），換算為 GB。
  文件: https://www.kernel.org/doc/html/latest/filesystems/proc.html#meminfo
- **CUDA 最佳實踐**: Pinned memory 不超過主機記憶體約 20%。
  文件: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations

### 2.4 Rust 環境變數

- **std::env**: env::var 用於 QDP_CHUNK_SIZE_MB、QDP_PINNED_POOL_SIZE、QDP_PCIE_GEN。
  文件: https://doc.rust-lang.org/std/env/fn.var.html

---

## 三、模組與型別設計

### 3.1 新增檔案與模組

- **檔案**: `qdp/qdp-core/src/gpu/pipeline_config.rs`
- **mod.rs**: 在 gpu/mod.rs 增加 `pub mod pipeline_config;`（僅在 target_os = "linux"）

### 3.2 型別定義

- **PCIeGeneration**: Gen3, Gen4, Gen5, Unknown（Default）
- **ComputeCapability**: Ampere, Ada, Hopper, Unknown（Default）
- **PipelineConfig**: chunk_size_mb: Option<usize>, pinned_pool_size: Option<usize>, enable_async_alloc: bool（預留）

### 3.3 方法簽名與行為

- **PipelineConfig::from_env() -> Self**
  從 QDP_CHUNK_SIZE_MB、QDP_PINNED_POOL_SIZE、QDP_PCIE_GEN 讀取；解析失敗視為未設定。

- **PipelineConfig::with_hardware_defaults(self, device: &Arc<CudaDevice>, host_mem_gb: Option<usize>) -> Result<Self>**
  - chunk_size_mb 未設定：依 PCIe + GPU 查表（見 3.4）。
  - pinned_pool_size 未設定：依 PCIe 建議 2–4，且 pinned_total <= 20% host_mem_gb；上下限 1–16。

- **PipelineConfig::validate(&self) -> Result<()>**
  chunk_size_mb ∈ [1, 256]，pinned_pool_size ∈ [1, 16]，且 pinned 總量 ≤ 20% host_mem_gb。

- **PipelineConfig::chunk_size_elements(&self) -> Result<usize>**
  chunk_size_mb * 1024 * 1024 / size_of::<f64>()。

- **PipelineConfig::pinned_pool_size_resolved(&self) -> Result<usize>**
  回傳 pinned_pool_size 或錯誤（with_hardware_defaults 後應已填滿）。

### 3.4 預設值表（chunk_size_mb 未設定時）

| PCIe | GPU | chunk_size_mb |
|------|-----|----------------|
| Gen5 | * | 16 |
| Gen4 | Hopper | 12 |
| Gen4 | 其他 | 8 |
| Gen3 | * | 4 |
| Unknown | * | 8 |

**pinned_pool_size（未設定時）**: 建議 2–4，pinned_total <= 20% host_mem；上限 16，下限 1。

---

## 四、硬體檢測實作要點

### 4.1 PCIe 代數

- **優先**: 讀取 QDP_PCIE_GEN，解析為 PCIeGeneration。
- **本階段範圍**: **僅實作 env 解析**（QDP_PCIE_GEN → PCIeGeneration）；未設定或解析失敗→Unknown。**不實作** sysfs（cudaDeviceGetPCIBusId + current_link_speed），以控制程式碼量；見第十節精準範圍。

### 4.2 GPU 計算能力

- **方式**: cuda_ffi 新增 cudaDeviceGetAttribute(value, attr, device)。
  常數 cudaDevAttrComputeCapabilityMajor = 75, Minor = 76。
  以 device.ordinal() 作為 device id，查詢 major/minor，對應到 ComputeCapability。
- **錯誤**: 查詢失敗回傳 Unknown，不讓 pipeline 建立失敗。

### 4.3 主機記憶體

- **Linux**: 讀取 /proc/meminfo 的 MemTotal（KB），換算為 GB。
- **失敗**: 回傳 None 或 16，validate 中以 16 GB 計算 20% 上限。

---

## 五、與 Pipeline 的整合步驟

### 5.1 不變更的對外 API

- run_dual_stream_pipeline(device, host_data, kernel_launcher)
- run_dual_stream_pipeline_aligned(device, host_data, align_elements, kernel_launcher)

### 5.2 解析 Config 的時機

- 在 run_dual_stream_pipeline 內：取得 host_mem_gb；config = PipelineConfig::from_env().with_hardware_defaults(device, host_mem_gb)?；validate()；chunk_size_elements = config.chunk_size_elements()?；pool_size = config.pinned_pool_size_resolved()?；呼叫 run_dual_stream_pipeline_with_chunk_size(..., chunk_size_elements, pool_size, kernel_launcher)。
- 在 run_dual_stream_pipeline_aligned 內：同樣取得 config 並 validate()；base_chunk = config.chunk_size_elements()?；對齊邏輯與現有一致得到 chunk_size_elements；pool_size = config.pinned_pool_size_resolved()?；呼叫 run_dual_stream_pipeline_with_chunk_size(..., chunk_size_elements, pool_size, ...)。

### 5.3 修改 run_dual_stream_pipeline_with_chunk_size

- **簽名**: 新增參數 pool_size: usize。
- **內部**: 刪除 const PINNED_POOL_SIZE；所有 PINNED_POOL_SIZE 改為 pool_size（Context::new, PinnedBufferPool::new, OverlapTracker::new, chunk_idx % pool_size, in_flight_pinned.len() == pool_size）。
- **驗證**: 函式開頭檢查 pool_size >= 1。

---

## 六、環境變數與文件

| 變數 | 說明 | 預設 | 範圍/備註 |
|------|------|------|-----------|
| QDP_CHUNK_SIZE_MB | Chunk 大小（MB） | 硬體檢測 | 1–256 |
| QDP_PINNED_POOL_SIZE | Pinned buffer 數量 | 硬體檢測 | 1–16；pinned 總量 ≤ 20% 主機記憶體 |
| QDP_PCIE_GEN | PCIe 代數（覆蓋檢測） | 自動 | 3/4/5 或 gen3/gen4/gen5 |
| QDP_USE_ASYNC_ALLOC | 預留 | false | 本階段不實作 |

在 pipeline_config.rs 或優化文件中註明上述變數與建議範圍。

---

## 七、FFI 新增（cuda_ffi.rs）

- **常數**: `cudaDevAttrComputeCapabilityMajor = 75`、`cudaDevAttrComputeCapabilityMinor = 76`（來自 [CUDART Types - cudaDeviceAttr](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html) 列舉，數值穩定）。
- **函式**: `cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32`。
  文件: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
  回傳 `cudaSuccess`(0) 成功；`cudaErrorInvalidDevice`、`cudaErrorInvalidValue` 等失敗。查詢失敗時回退 `ComputeCapability::Unknown`，不讓 pipeline 建立失敗。

僅在 `target_os = "linux"` 的 cuda_ffi 區塊內加入。

---

## 八、單元測試與驗收

### 8.1 單元測試（pipeline_config.rs）

- from_env: 設定/未設定各環境變數，檢查解析結果。
- validate: 合法範圍通過；chunk_size_mb 0 或 257、pinned_pool_size 0 或 17 失敗；超過 20% host memory 失敗。
- with_hardware_defaults: 無 GPU 可 skip；有 GPU 時檢查回傳在預期範圍且通過 validate。
- 邊界: chunk_size_mb=1/256、pinned_pool_size=1/16 通過驗證。

### 8.2 階段 3 完成標準

- [x] pipeline_config.rs 已實現並通過 cargo build / cargo test。
- [x] 單元測試覆蓋 from_env、validate、邊界與非法值。
- [x] Pipeline 使用 config 的 chunk/pool 參數，對外 API 不變。
- [x] 文檔或註解說明環境變數與預設邏輯。
- [x] 程式碼量符合第十節邊界（不實作 sysfs PCIe、不查 GpuOverlap/asyncEngineCount）。

---

## 九、實作順序建議

1. cuda_ffi.rs: 新增 cudaDeviceGetAttribute 與常數。
2. pipeline_config.rs: 實作 PCIeGeneration、ComputeCapability、PipelineConfig、from_env、get_host_memory_gb、detect_pcie_generation（先 env + Unknown）、detect_compute_capability、with_hardware_defaults、validate、chunk_size_elements、pinned_pool_size_resolved。
3. gpu/mod.rs: 新增 pipeline_config 模組。
4. pipeline.rs: run_dual_stream_pipeline / run_dual_stream_pipeline_aligned 內解析 config；run_dual_stream_pipeline_with_chunk_size 簽名新增 pool_size，內部改用參數。
5. 單元測試與文件。

---

## 十、程式碼量與精準範圍（第三階段邊界）

本節以**效能優化為前提**、**最精準完整**為原則，明確界定第三階段程式碼邊界。**第三階段總程式碼量上限約 500 行**（含註解與空行）；若以「效能優先、多查資料」為前提，**新增程式碼量可放寬至約 1000 行以內**。實作已含效能優先說明（零、0.1/0.2）與官方文件引用（rust-development / qdp-development）。

### 10.1 變更範圍一覽（總量 ≤500 行；放寬時 ≤1000 行）

| 檔案 | 變更類型 | 預估量 | 說明 |
|------|----------|--------|------|
| **cuda_ffi.rs** | 新增 | **≤12 行** | 常數 75/76、cudaDeviceGetAttribute 宣告；無 cudaDeviceGetPCIBusId（本階段不實作 sysfs）。 |
| **pipeline_config.rs** | 新增檔案 | **約 200～260 行** | 見 10.2；不含 sysfs、不含 GpuOverlap/asyncEngineCount；精簡註解以控行數。 |
| **gpu/mod.rs** | 新增 | **2 行** | `#[cfg(target_os = "linux")]` + `pub mod pipeline_config;`。 |
| **pipeline.rs** | 修改 | **+約 28～32 行、改 6 處** | 兩處入口解析 config；`run_dual_stream_pipeline_with_chunk_size` 簽名 +1 參數、刪除 const、6 處 PINNED_POOL_SIZE→pool_size（337,348,349,359,380,446）。 |
| **測試** | 新增 | **約 80～100 行** | pipeline_config 單元測試：from_env、validate、邊界、with_hardware_defaults（有 GPU 時）；精簡以控行數。 |
| **合計** | — | **約 322～406 行** | **上限 500 行**；建議實作落在 350～420 行，預留緩衝。 |

### 10.2 pipeline_config.rs 精準邊界

- **必須實作**（否則 pipeline 無法接上）
  - 型別：`PCIeGeneration`（4 變體）、`ComputeCapability`（4 變體）、`PipelineConfig`（3 欄位：chunk_size_mb, pinned_pool_size, enable_async_alloc 預留）。
  - `from_env()`：僅讀 QDP_CHUNK_SIZE_MB、QDP_PINNED_POOL_SIZE、QDP_PCIE_GEN；解析失敗→None，不新增依賴。
  - `get_host_memory_gb()`：讀 `/proc/meminfo` MemTotal，失敗→None。
  - `detect_pcie_generation()`：**僅實作 env 解析**（3/4/5 或 gen3/gen4/gen5→對應列舉）；未設定或解析失敗→Unknown。**本階段不實作 sysfs**。
  - `detect_compute_capability(device)`：呼叫 cuda_ffi cudaDeviceGetAttribute(75/76)，major/minor→ComputeCapability；失敗→Unknown。
  - `with_hardware_defaults(self, device, host_mem_gb)`：chunk 未設→查表（3.4）；pool 未設→2～4 且 ≤20% host、clamp 1..=16。
  - `validate(&self)`：chunk 1..=256、pool 1..=16、pinned 總量 ≤20% host_mem_gb（host 未知時用 16 GB）。
  - `chunk_size_elements(&self)`、`pinned_pool_size_resolved(&self)`：依已填欄位計算/回傳，未填→Err。

- **本階段不實作**（控制程式碼量）
  - PCIe sysfs（cudaDeviceGetPCIBusId + 讀 current_link_speed）。
  - cudaDevAttrGpuOverlap / asyncEngineCount 查詢。
  - 額外 crate（如 envy/envparse）；僅用 `std::env::var` + `str::parse::<usize>()` / 簡單 match。

- **預估行數拆解（總和 ≤260 行）**：enum 約 12；struct + Default 約 8；from_env 約 22；get_host_memory_gb 約 18；detect_pcie 約 12；detect_compute_capability 約 28；with_hardware_defaults 約 48；validate 約 32；chunk_size_elements / pinned_pool_size_resolved 約 12；模組頭/必要註解約 18。合計約 **200～260 行**，以符合階段總量 ≤500 行。

### 10.3 pipeline.rs 修改點（精準清單）

- **run_dual_stream_pipeline**（約 257～274）：
  - 刪除 `const CHUNK_SIZE_ELEMENTS`。
  - 新增：`let host_mem_gb = crate::gpu::pipeline_config::get_host_memory_gb();`、`let config = PipelineConfig::from_env().with_hardware_defaults(device, host_mem_gb)?;`、`config.validate()?;`、`let chunk_size_elements = config.chunk_size_elements()?;`、`let pool_size = config.pinned_pool_size_resolved()?;`。
  - 呼叫改為 `run_dual_stream_pipeline_with_chunk_size(device, host_data, chunk_size_elements, pool_size, kernel_launcher)`。

- **run_dual_stream_pipeline_aligned**（約 281～317）：
  - 同上取得 config、validate、`base_chunk = config.chunk_size_elements()?`、`pool_size = config.pinned_pool_size_resolved()?`。
  - 對齊邏輯改為以 `base_chunk_elements`（由 config 來）取代 `BASE_CHUNK_SIZE_ELEMENTS`，其餘不變。
  - 呼叫改為 `run_dual_stream_pipeline_with_chunk_size(..., chunk_size_elements, pool_size, ...)`。

- **run_dual_stream_pipeline_with_chunk_size**：
  - 簽名新增 `pool_size: usize`。
  - 開頭 `if pool_size == 0 { return Err(...); }`。
  - 刪除 `const PINNED_POOL_SIZE`；以下 5～6 處改為使用參數 `pool_size`：Context::new、PinnedBufferPool::new、OverlapTracker::new、event_slot 計算、in_flight_pinned.len() == pool_size。

### 10.4 完成標準（對應 8.2）

- pipeline_config.rs 存在且通過 `cargo build` / `cargo test`（含 gpu 相關 tests）。
- 對外 API 不變；僅內部使用 config 與 pool_size。
- 環境變數與預設邏輯在模組註解或 doc 中說明。
- **不要求**：PCIe sysfs、GpuOverlap 查詢、額外依賴；**要求**：預設值表（3.4）、20% 驗證、邊界測試。

---

## 十一、執行方向確認

實作執行順序與計畫 17–30 行（效能優先）一致：

1. **入口**：`run_dual_stream_pipeline` / `run_dual_stream_pipeline_aligned` 不變對外簽名。
2. **解析 config**：`host_mem_gb = PipelineConfig::get_host_memory_gb()` → `config = PipelineConfig::from_env().with_hardware_defaults(device, host_mem_gb)?` → `config.validate()?`。
3. **取得參數**：`chunk_size_elements = config.chunk_size_elements()?`、`pool_size = config.pinned_pool_size_resolved()?`。
4. **呼叫內部**：`run_dual_stream_pipeline_with_chunk_size(..., chunk_size_elements, pool_size, kernel_launcher)`。
5. **內部使用**：Context::new(device, pool_size)、PinnedBufferPool::new(pool_size, chunk_size_elements)、OverlapTracker::new(pool_size, …)、event_slot = chunk_idx % pool_size、滿池判斷 `in_flight_pinned.len() == pool_size`。

程式碼中已加入：`pipeline_config.rs` 模組 doc 的效能優先說明與 NVIDIA/CUDA 官方連結；`cuda_ffi.rs` 與相關函式的 `Ref:` 註解（CUDART Device、CUDART Types、/proc/meminfo、std::env）；`pipeline.rs` 的 config 流程說明。

---

## 十二、參考文件一覽

| 主題 | 連結 |
|------|------|
| cudaDeviceGetAttribute | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html |
| cudaDeviceAttr（Major=75, Minor=76 為列舉值） | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html |
| cudaDeviceGetPCIBusId（PCIe 檢測用） | 同上 Device 章節 |
| Compute Capabilities | https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#compute-capabilities |
| PCI sysfs（current_link_speed / max_link_speed） | https://docs.kernel.org/PCI/sysfs-pci.html |
| /proc/meminfo | https://www.kernel.org/doc/html/latest/filesystems/proc.html#meminfo |
| CUDA Memory Best Practices | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations |
| NVIDIA 資料傳輸優化 | https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/ |
| **NVIDIA Overlap 傳輸**（雙 stream + pinned） | https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/ |
| CUDA 異步複製 (4.11) | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html |
| PCIe Gen4 實測吞吐 | https://www.passmark.com/products/pcie-gen4-test-card/practical-throughput.php |
| H2D 傳輸優化（批次、pinned、非同步） | https://www.microway.com/hpc-tech-tips/optimize-cuda-host-to-device-transfers/ |
| cudaDeviceProp / deviceOverlap、asyncEngineCount | https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html |
| Pinned memory 上限與 OS memlock（論壇） | https://forums.developer.nvidia.com/t/max-amount-of-host-pinned-memory-available-for-allocation/56053 |
| Overlap 與 chunk/kernel 時間取捨（SO） | https://stackoverflow.com/questions/24915488/memory-compute-overlap-affects-kernel-duration |
| Rust std::env / 數值解析 | https://doc.rust-lang.org/std/env/fn.var.html ；解析失敗用 Option、不新增 crate。 |

（實作已於程式碼內附上對應 Ref 註解，符合 rust-development 引用要求。）
