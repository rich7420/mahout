# Benchmarks

This directory contains Python benchmarks for Mahout QDP. There are four main
scripts:

- `benchmark_mnist.py`: **MNIST image classification benchmark** - Demonstrates
  QDP's GPU-accelerated amplitude encoding vs PennyLane on real-world image data.
  This is the best showcase of QDP's performance advantage for quantum machine learning.
- `benchmark_e2e.py`: end-to-end latency from disk to GPU VRAM (includes IO,
  normalization, encoding, transfer, and a dummy forward pass).
- `benchmark_throughput.py`: DataLoader-style throughput benchmark
  that measures vectors/sec across Mahout, PennyLane, and Qiskit.
- `benchmark_latency.py`: Data-to-State latency benchmark (CPU RAM -> GPU VRAM).

## Quick Start

From the repo root:

```bash
cd qdp
make benchmark
```

This installs the QDP Python package (if needed), installs benchmark
dependencies, and runs both benchmarks.

## Manual Setup

```bash
cd qdp/qdp-python
uv sync --group benchmark
```

Then run benchmarks with `uv run python ...` or activate the virtual
environment and use `python ...`.

## Complete Training Benchmark (Forward + Backward Pass)

**This is the most comprehensive benchmark showing QDP's advantage in real training scenarios!**

The training benchmark compares the complete training pipeline (forward + backward pass)
between PennyLane's native AmplitudeEmbedding and QDP's GPU-accelerated encoding.

**Key Features:**
- Fair comparison: Same VQC structure, same backend, same differentiation method
- Complete training loop: Forward pass + backward pass + optimization
- Real-world scenario: MNIST binary classification (0 vs 1)
- **QDP requires GPU backend (lightning.gpu) for end-to-end GPU pipeline**

```bash
cd qdp/qdp-python/benchmark
uv run python benchmark_training.py
```

**Quick Options:**
```bash
# Larger batch size for better QDP advantage
uv run python benchmark_training.py --batch-size 64 --limit-batches 10

# More qubits (larger state vectors = more encoding overhead)
uv run python benchmark_training.py --n-qubits 12 --batch-size 32

# Skip one method for faster testing
uv run python benchmark_training.py --skip-pennylane
```

**Expected Results:**
- Small batches (< 16): Similar performance
- Medium batches (16-64): QDP 1.5-3x faster
- Large batches (64+): QDP 3-10x faster
- With GPU backend: QDP 5-10x faster

**📖 Complete Documentation**: See [`BENCHMARK_DOCUMENTATION.md`](BENCHMARK_DOCUMENTATION.md) for:
- Detailed performance analysis
- QDP overhead analysis (only ~1.5% of total training time)
- Complete workflow explanation
- Code architecture
- Troubleshooting guide
- Improvement suggestions

## MNIST Benchmark (Image Classification)

**This is the recommended benchmark to showcase QDP's performance!**

The MNIST benchmark demonstrates QDP's GPU-accelerated amplitude encoding
on real-world image classification data. It compares QDP against PennyLane's
CPU-based approach, showing dramatic speedups (typically 50-100x faster).

**Why MNIST?**
- High-dimensional data: 784 features (28×28 pixels)
- Real-world use case: Image classification is a common QML application
- Encoding challenge: L2 normalization + padding to 2^n state vector
- Batch processing: QDP's single GPU kernel launch vs PennyLane's Python loops

```bash
cd qdp/qdp-python/benchmark
python benchmark_mnist.py
```

**Options:**
```bash
# Custom batch size and qubits
python benchmark_mnist.py --batch-size 256 --n-qubits 10

# Process more batches for better statistics
python benchmark_mnist.py --num-batches 10

# Use float64 precision (slower but more accurate)
python benchmark_mnist.py --precision float64

# Skip PennyLane benchmark (faster execution)
python benchmark_mnist.py --skip-pennylane

# Skip output verification
python benchmark_mnist.py --skip-verification
```

**Expected Results:**
- QDP: ~0.01-0.1 seconds for 128 samples (batch processing)
- PennyLane: ~5-10 seconds for 128 samples (sequential processing)
- **Speedup: 50-100x** (or more depending on hardware)

**Key Features:**
- Automatic MNIST download via torchvision
- Zero-copy PyTorch integration via DLPack
- Output verification (compares QDP vs PennyLane outputs)
- Detailed performance statistics

## E2E Benchmark (Disk -> GPU)

```bash
cd qdp/qdp-python/benchmark
python benchmark_e2e.py
```

Additional options:

```bash
python benchmark_e2e.py --qubits 16 --samples 200 --frameworks mahout-parquet mahout-arrow
python benchmark_e2e.py --frameworks all
```

Notes:

- `--frameworks` accepts a space-separated list or `all`.
  Options: `mahout-parquet`, `mahout-arrow`, `pennylane`, `qiskit`.
- The script writes `final_benchmark_data.parquet` and
  `final_benchmark_data.arrow` in the current working directory and overwrites
  them on each run.
- If multiple frameworks run, the script compares output states for
  correctness at the end.

## Data-to-State Latency Benchmark

```bash
cd qdp/qdp-python/benchmark
python benchmark_latency.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
python benchmark_latency.py --frameworks mahout,pennylane
```

Notes:

- `--frameworks` is a comma-separated list or `all`.
  Options: `mahout`, `pennylane`, `qiskit-init`, `qiskit-statevector`.
- The latency test reports average milliseconds per vector.
- Flags:
  - `--qubits`: controls vector length (`2^qubits`).
  - `--batches`: number of host-side batches to stream.
  - `--batch-size`: vectors per batch; raises total samples (`batches * batch-size`).
  - `--prefetch`: CPU queue depth; higher values help keep the pipeline fed.
- See `qdp/qdp-python/benchmark/benchmark_latency.md` for details and example output.

## DataLoader Throughput Benchmark

Simulates a typical QML training loop by continuously loading batches of 64
vectors (default). Goal: demonstrate that QDP can saturate GPU utilization and
avoid the "starvation" often seen in hybrid training loops.

See `qdp/qdp-python/benchmark/benchmark_throughput.md` for details and example
output.

```bash
cd qdp/qdp-python/benchmark
python benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
python benchmark_throughput.py --frameworks mahout,pennylane
```

Notes:

- `--frameworks` is a comma-separated list or `all`.
  Options: `mahout`, `pennylane`, `qiskit`.
- Throughput is reported in vectors/sec (higher is better).

## Dependency Notes

- Qiskit and PennyLane are optional. If they are not installed, their benchmark
  legs are skipped automatically.
- For Mahout-only runs, you can uninstall the competitor frameworks:
  `uv pip uninstall qiskit pennylane`.

### We can also run benchmarks on colab notebooks(without owning a GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apache/mahout/blob/dev-qdp/qdp/qdp-python/benchmark/notebooks/mahout_benchmark.ipynb)
