# QDP Benchmarking Suite

Comprehensive benchmarking suite comparing **Mahout QDP** against **Qiskit** and **PennyLane** in real-world Quantum Machine Learning (QML) workflows.

## Overview

This benchmark suite consists of three main components:

1. **Synthetic Data Scaling (4-22 Qubits)**: Measures pure encoding speed and fidelity as dimensionality grows exponentially.
2. **Real-World Data Pipeline (Parquet/Arrow)**: Simulates a training loop where data must be read from disk and encoded. This isolates the "Zero-Copy" advantage.
3. **Data Generation Utility**: Creates appropriate Parquet datasets for stress testing.

## Installation

```bash
uv pip install -r requirements.txt
```

**Note**: Assumes `mahout_qdp` is already installed (built from source).

## Quick Start

### 1. Synthetic Scaling Benchmark

Tests raw computation capability across different qubit counts:

```bash
python benchmark_synthetic.py
```

This will:
- Compare Mahout QDP, Qiskit, and PennyLane across 10-22 qubits
- Generate `benchmark_synthetic.png` showing time vs qubits (log scale)
- Display fidelity measurements

**Expected Results**: Mahout should show relatively flat latency (PCIe bound) for low qubits, crossing over Qiskit/PennyLane around 12-14 qubits. At 22 qubits, Qiskit will hit memory bandwidth limits while Mahout utilizes high-bandwidth GPU memory.

### 2. Pipeline Throughput Benchmark

Simulates real-world IO + encoding workflow:

```bash
# Generate test dataset (18 qubits, 500 rows)
python generate_dataset.py --qubits 18 --rows 500 --out benchmarks_data.parquet

# Run benchmark
python benchmark_pipeline.py
```

**Expected Results**: Standard SOP (Pandas) usually chokes on "exploding" list columns into Numpy arrays. Mahout's approach (streaming encoding + GPU normalization) should yield **10x-50x speedup** depending on disk speed and CPU single-core performance.

### 3. Real-World Dataset Benchmark

Uses actual QML datasets (MNIST & Credit Card Fraud Detection):

```bash
# Prepare MNIST dataset (10 qubits, 10000 samples)
python prepare_data.py --dataset mnist --rows 10000

# Run benchmark
python benchmark_real_world.py --file data_cache/mnist_q10_10000.parquet --qubits 10

# Prepare Credit Card dataset (5 qubits, 50000 samples)
python prepare_data.py --dataset credit --rows 50000

# Run benchmark
python benchmark_real_world.py --file data_cache/credit_q5_50000.parquet --qubits 5
```

**Expected Results**:
- **Credit Card (5 Qubits)**: Mahout should show **5-10x** speedup (CPU-friendly scenario, but Python loop overhead is still a bottleneck).
- **MNIST (10 Qubits)**: Mahout should demonstrate **10-30x** speedup (CPU L2 norm becomes noticeable, Qiskit Statevector validation slows down).
- **Synthetic Large Data (22 Qubits)**: This is Mahout's main battlefield. CPU will struggle with reading and normalizing 4M-dimensional vectors (Memory Bandwidth Bound). Mahout's Async Pipeline can achieve **50x+** speedup (limited by PCIe bandwidth).

## Files

- `requirements.txt`: Python dependencies
- `generate_dataset.py`: Utility to create synthetic Parquet files for stress testing
- `benchmark_synthetic.py`: CPU vs GPU scaling and fidelity tests
- `benchmark_pipeline.py`: Real-world IO + Encoding throughput comparison
- `prepare_data.py`: Download and preprocess real datasets (MNIST & Credit Card)
- `benchmark_real_world.py`: Real-world training loop benchmark

## Fidelity Requirements

All benchmarks should maintain fidelity `> 0.999999` (floating point noise only). This validates that Mahout QDP maintains numerical accuracy while achieving performance improvements.

## Notes

- For large datasets (22 qubits), ensure sufficient disk space and RAM
- GPU memory requirements scale with qubit count: `2^n * 16 bytes` (complex128)
- The benchmarks include DLPack conversion overhead to ensure fair comparison
- Real-world benchmarks simulate actual training loops with batch processing
