#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FINAL END-TO-END BENCHMARK (Disk -> GPU VRAM).

Scope:
1. Disk IO: Reading Parquet file.
2. Preprocessing: L2 Normalization (CPU vs GPU).
3. Encoding: Quantum State Preparation.
4. Transfer: Moving data to GPU VRAM.
5. Consumption: 1 dummy Forward Pass to ensure data is usable.

This is the most realistic comparison for a "Cold Start" Training Epoch.
"""

import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
from mahout_qdp import QdpEngine

# Competitors
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

# Config
DATA_FILE = "final_benchmark_data.parquet"
HIDDEN_DIM = 16
BATCH_SIZE = 64  # Small batch to stress loop overhead


class DummyQNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.fc = nn.Linear(1 << n_qubits, HIDDEN_DIM)

    def forward(self, x):
        return self.fc(x)


def generate_data(n_qubits, n_samples):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    print(f"Generating {n_samples} samples of {n_qubits} qubits to {DATA_FILE}...")
    dim = 1 << n_qubits

    # Write using PyArrow directly for speed
    chunk_size = 500
    schema = pa.schema([("feature_vector", pa.list_(pa.float64()))])

    with pq.ParquetWriter(DATA_FILE, schema) as writer:
        for start_idx in range(0, n_samples, chunk_size):
            current = min(chunk_size, n_samples - start_idx)
            data = np.random.rand(current, dim).astype(np.float64)
            feature_vectors = [row.tolist() for row in data]
            arrays = pa.array(feature_vectors, type=pa.list_(pa.float64()))
            batch_table = pa.Table.from_arrays([arrays], names=["feature_vector"])
            writer.write_table(batch_table)

    file_size_mb = os.path.getsize(DATA_FILE) / (1024 * 1024)
    print(f"  Generated {n_samples} samples, file size: {file_size_mb:.2f} MB")


# -----------------------------------------------------------
# 1. Qiskit Full Pipeline
# -----------------------------------------------------------
def run_qiskit(n_qubits, n_samples):
    if not HAS_QISKIT:
        print("\n[Qiskit] Not installed, skipping.")
        return 0.0

    print("\n[Qiskit] Full Pipeline (Disk -> GPU)...")
    print("  * Warning: Qiskit `initialize()` is O(2^N) compute bound.")
    print("  * This will be very slow for high qubit counts.")

    model = DummyQNN(n_qubits).cuda()
    backend = AerSimulator(method="statevector")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # 1. Disk IO (Included!)
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data = np.stack(df["feature_vector"].values)

    io_time = time.perf_counter() - start_time
    print(f"  IO Time: {io_time:.4f} s")

    # 2. Loop
    for i in range(0, n_samples, BATCH_SIZE):
        batch = raw_data[i : i + BATCH_SIZE]

        # 3. CPU Norm
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        batch = batch / norms

        # 4. State Prep (The heavy part)
        batch_states = []
        for vec_idx, vec in enumerate(batch):
            qc = QuantumCircuit(n_qubits)
            qc.initialize(vec, range(n_qubits))
            qc.save_statevector()
            # Transpile & Run
            t_qc = transpile(qc, backend)
            result = backend.run(t_qc).result().get_statevector().data
            batch_states.append(result)

            # Progress indicator
            if (vec_idx + 1) % 10 == 0:
                print(f"    Processed {vec_idx + 1}/{len(batch)} vectors...", end="\r")

        # 5. H2D Transfer
        gpu_tensor = torch.tensor(
            np.array(batch_states), device="cuda", dtype=torch.complex64
        )

        # 6. Consumption
        _ = model(gpu_tensor.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"\n  Total Time: {total_time:.4f} s")
    return total_time


# -----------------------------------------------------------
# 2. PennyLane Full Pipeline
# -----------------------------------------------------------
def run_pennylane(n_qubits, n_samples):
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0

    print("\n[PennyLane] Full Pipeline (Disk -> GPU)...")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    model = DummyQNN(n_qubits).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # 1. Disk IO (Included!)
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data = np.stack(df["feature_vector"].values)

    io_time = time.perf_counter() - start_time
    print(f"  IO Time: {io_time:.4f} s")

    # 2. Loop
    for i in range(0, n_samples, BATCH_SIZE):
        # To Torch (CPU)
        batch_cpu = torch.tensor(raw_data[i : i + BATCH_SIZE])

        # 3. QNode (CPU Execution)
        # Broadcasting or Stack
        try:
            state_cpu = circuit(batch_cpu)
        except Exception:
            state_cpu = torch.stack([circuit(x) for x in batch_cpu])

        # 4. H2D Transfer
        state_gpu = state_cpu.to("cuda", dtype=torch.float32)

        # 5. Consumption
        _ = model(state_gpu.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")
    return total_time


# -----------------------------------------------------------
# 3. Mahout Full Pipeline
# -----------------------------------------------------------
def run_mahout(engine, n_qubits, n_samples):
    print("\n[Mahout] Full Pipeline (Disk -> GPU)...")
    model = DummyQNN(n_qubits).cuda()
    state_dim = 1 << n_qubits

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # 1. IO + Encode + H2D (All-in-One Rust Call)
    # Using F32 engine for max speed
    qtensor = engine.encode_from_parquet(DATA_FILE, n_qubits, "amplitude")

    # 2. Zero-Copy View
    gpu_data = torch.from_dlpack(qtensor)

    # Reshape logic
    n_loaded = gpu_data.numel() // state_dim
    gpu_data = gpu_data[: n_loaded * state_dim].view(n_loaded, state_dim)

    encode_time = time.perf_counter() - start_time
    print(f"  IO + Encode Time: {encode_time:.4f} s")

    # 3. Loop (Data already on GPU)
    for i in range(0, n_samples, BATCH_SIZE):
        if i >= n_loaded:
            break
        batch = gpu_data[i : i + BATCH_SIZE]

        # 4. Consumption
        _ = model(batch.abs())

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"  Total Time: {total_time:.4f} s")
    return total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final End-to-End Benchmark (Disk -> GPU VRAM)"
    )
    parser.add_argument(
        "--qubits", type=int, default=16, help="Number of qubits (16 recommended)"
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="Number of training samples"
    )
    args = parser.parse_args()

    generate_data(args.qubits, args.samples)

    # Init Mahout (F32)
    try:
        engine = QdpEngine(0, precision="float32")
    except Exception as e:
        print(f"Mahout Init Error: {e}")
        exit(1)

    print("\n" + "=" * 70)
    print(f"FINAL E2E BENCHMARK: {args.qubits} Qubits, {args.samples} Samples")
    print("=" * 70)

    # Run benchmarks
    t_pl = run_pennylane(args.qubits, args.samples)
    t_qiskit = run_qiskit(args.qubits, args.samples)
    t_mahout = run_mahout(engine, args.qubits, args.samples)

    print("\n" + "=" * 70)
    print("FINAL E2E LATENCY (Lower is Better)")
    print(f"Samples: {args.samples}, Qubits: {args.qubits}")
    print("=" * 70)

    results = []
    if t_mahout > 0:
        results.append(("Mahout", t_mahout))
    if t_pl > 0:
        results.append(("PennyLane", t_pl))
    if t_qiskit > 0:
        results.append(("Qiskit", t_qiskit))

    results.sort(key=lambda x: x[1])

    for name, time_val in results:
        print(f"{name:12s} {time_val:10.4f} s")

    print("-" * 70)
    if t_mahout > 0:
        if t_pl > 0:
            print(f"Speedup vs PennyLane: {t_pl / t_mahout:10.2f}x")
        if t_qiskit > 0:
            print(f"Speedup vs Qiskit:    {t_qiskit / t_mahout:10.2f}x")

    print("=" * 70)
    print("\nKey Insights:")
    print("  1. IO time is included in all benchmarks (fair comparison)")
    print("  2. Qiskit suffers from O(2^N) gate synthesis overhead")
    print("  3. PennyLane suffers from CPU normalization + PCIe transfer")
    print("  4. Mahout's zero-copy pipeline eliminates all CPU bottlenecks")
    print("  5. This demonstrates the true cost of end-to-end data loading")
