# 為何 Nsight 仍看到 800 次 cuStreamSynchronize？（Fix 4 沒生效）

## 結論：Python 用的是舊的 extension，沒有連結到新編譯的 qdp-core

- **Fix 3**：Body 每 `pool_size`（2）個 chunk 呼叫 `sync_copy_stream()` → 200 chunks → **100 次/run**；若 8 runs → **800 次**，與你 Nsight 的 800 完全吻合。
- **Fix 4**：主迴圈已不再呼叫 `sync_copy_stream()`，僅 Epilogue 一次 → 每 run 應只有 **1 次** cuStreamSynchronize（copy stream），總數應是 run 數的倍數（約 10～20），不會是 800。

所以目前 Nsight 抓到的 binary **仍是 Fix 3 行為**，代表 **benchmark 沒有用到新編譯的 qdp-core**。

---

## 正確編譯／安裝流程（讓 Python 用到 Fix 4）

只跑 `cargo build -p qdp-core --release` **只會編譯 qdp-core**，不會重編 **qdp-python** 的 extension（`_qdp`）。
`uv run python run_pipeline_baseline.py` 載入的是 **已安裝的** `_qdp`，若沒重新安裝，會繼續用舊的 qdp-core。

請在 **qdp-python** 目錄下重新安裝（會重編 extension 並連結目前的 qdp-core）：

```bash
cd /home/rich-wsl/mahout/qdp/qdp-python
uv pip install -e . --force-reinstall
```

或用 maturin 直接開發安裝（release 版）：

```bash
cd /home/rich-wsl/mahout/qdp/qdp-python
uv run maturin develop --release
```

之後再跑 benchmark（與 Nsight）：

```bash
cd /home/rich-wsl/mahout/qdp/qdp-python/benchmark
export QDP_ENABLE_POOL_METRICS=0
export QDP_ENABLE_OVERLAP_TRACKING=0
uv run python run_pipeline_baseline.py --qubits 16 --batch-size 64 --prefetch 16 --batches 200 --trials 5
```

用 Nsight 再抓一次：**cuStreamSynchronize** 應只剩約 10～20 次（每 run 僅 Epilogue），**cudaEventSynchronize** 會出現、次數約為 `(num_chunks - pool_size) × run 數**。
