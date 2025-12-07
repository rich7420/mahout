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
Real-world SOP Benchmark (Qiskit vs PennyLane vs Mahout).

Simulates a real Data Loader loop. Compares three modes:

1. Qiskit SOP (Baseline):
   - Pandas read -> Numpy -> CPU L2 Norm -> Statevector -> Transpile overhead
   - Represents the "Circuit Simulation" approach.

2. PennyLane SOP (Competitor):
   - Pandas read -> Numpy -> AmplitudeEmbedding (CPU Norm & State Prep) -> PCIe Transfer -> GPU
   - Represents the standard QML approach.

3. Mahout SOP (The Innovation):
   - Arrow read -> GPU Async Pipeline (Normalization + Encoding)
   - Demonstrates Rust + GPU Zero-Copy architecture.

only memory benchmark
"""

import time
import argparse
import torch
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector
from mahout_qdp import QdpEngine

# Try importing PennyLane
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def benchmark_baseline_qiskit(parquet_file, num_qubits, batch_size=128):
    """
    Baseline: Traditional Qiskit data loading workflow.
    """
    print(f"\n[Baseline] Running Qiskit SOP on {parquet_file}...")

    start_time = time.perf_counter()

    # 1. IO: Read Parquet
    df = pd.read_parquet(parquet_file)
    raw_data = np.stack(df["feature_vector"].values)

    vectors_processed = 0
    state_prep_time = 0

    # Simulate Training Loop
    for i in range(0, len(raw_data), batch_size):
        batch = raw_data[i : i + batch_size]

        t0 = time.perf_counter()

        # 3. CPU Normalization
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_batch = batch / norms

        # 4. State Preparation (One by one)
        batch_states = []
        for vec in normalized_batch:
            # Pure CPU compute
            sv = Statevector(vec)
            # In a real loop, we would act on this sv
            batch_states.append(sv.data)

        # 5. GPU Transfer (Fairness Fix: Match Mahout/PennyLane)
        # To match Mahout's output (GPU Tensor), we must move results to GPU
        # This pays the PCIe tax, making the comparison fair
        _ = torch.tensor(np.array(batch_states), device="cuda")
        torch.cuda.synchronize()

        state_prep_time += time.perf_counter() - t0
        vectors_processed += len(batch)

    total_time = time.perf_counter() - start_time
    throughput = vectors_processed / total_time

    print(f"  Processed {vectors_processed} vectors")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  State Prep (Compute): {state_prep_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} vectors/sec")
    return throughput


def benchmark_pennylane_sop(parquet_file, num_qubits, batch_size=128):
    """
    Competitor: PennyLane standard workflow.

    The Fairness Standard:
    Since Mahout provides data on GPU, PennyLane must also pay the cost
    of moving data to GPU (H2D Copy) to be comparable in a PyTorch QNN pipeline.
    """
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0

    print(f"\n[PennyLane] Running Standard SOP on {parquet_file}...")

    # Define standard CPU device (fastest for simple embedding)
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface=None)
    def circuit(features):
        # PennyLane performs L2 Normalization on CPU here if normalize=True
        qml.AmplitudeEmbedding(
            features=features, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    start_time = time.perf_counter()

    # 1. IO: Read Parquet (Standard Pandas)
    # This is often the bottleneck Mahout avoids via Arrow
    df = pd.read_parquet(parquet_file)

    # 2. Preprocessing
    # The expensive "explode" operation to get numpy matrix
    raw_data = np.stack(df["feature_vector"].values)

    vectors_processed = 0
    compute_time = 0

    # Simulate Training Loop
    for i in range(0, len(raw_data), batch_size):
        batch = raw_data[i : i + batch_size]

        t0 = time.perf_counter()

        # 3. Embedding (CPU Compute & Normalization)
        # We use a list comprehension as strictly idiomatic PennyLane batching
        # on default.qubit often requires this or specific broadcast settings.
        batch_states = [circuit(vec) for vec in batch]

        # 4. GPU Transfer (The Fairness Fix)
        # To match Mahout's output (GPU Tensor), we must move results to GPU
        # This pays the PCIe tax
        _ = torch.tensor(np.array(batch_states), device="cuda")

        # Force sync to measure true latency
        torch.cuda.synchronize()

        compute_time += time.perf_counter() - t0
        vectors_processed += len(batch)

    total_time = time.perf_counter() - start_time
    throughput = vectors_processed / total_time

    print(f"  Processed {vectors_processed} vectors")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  Compute + Transfer: {compute_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} vectors/sec")
    return throughput


def benchmark_mahout_gpu(parquet_file, num_qubits, engine, batch_size=128):
    """
    Mahout: Direct Parquet -> GPU workflow (Zero-Copy Arrow Path).
    """
    print(f"\n[Mahout] Running GPU SOP on {parquet_file}...")

    start_time = time.perf_counter()

    # Direct encode from Parquet (zero-copy Arrow path)
    # Mahout's engine.encode_from_parquet handles:
    # 1. Arrow IO (Chunked reading)
    # 2. GPU L2 Normalization (Kernel)
    # 3. Batch processing on GPU
    t0 = time.perf_counter()

    # Note: In the final library, encode_from_parquet handles the looping internally
    # or returns a generator. Here we simulate the optimal "one-shot" load
    # or chunked load supported by the Rust backend.
    qtensor = engine.encode_from_parquet(parquet_file, num_qubits, "amplitude")

    # Zero-cost conversion to PyTorch
    torch_tensor = torch.from_dlpack(qtensor)

    # Wait for async GPU work
    torch.cuda.synchronize()

    encode_time = time.perf_counter() - t0

    # Calculate stats
    single_vector_size = 1 << num_qubits
    total_elements = torch_tensor.shape[0]
    vectors_processed = total_elements // single_vector_size

    total_time = time.perf_counter() - start_time
    throughput = vectors_processed / total_time

    print(f"  Processed {vectors_processed} vectors")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  GPU Encode (Compute + IO): {encode_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} vectors/sec")
    return throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-world QML SOP benchmark")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to generated parquet file"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        required=True,
        help="Expected qubits (must match file data)",
    )
    args = parser.parse_args()

    # 1. Initialize Mahout
    try:
        engine = QdpEngine(0, precision="float32")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # 2. Run Benchmarks
    baseline_rate = benchmark_baseline_qiskit(args.file, args.qubits)
    pl_rate = benchmark_pennylane_sop(args.file, args.qubits)
    mahout_rate = benchmark_mahout_gpu(args.file, args.qubits, engine)

    # 3. Summary
    print("\n" + "=" * 60)
    print(f"FINAL THROUGHPUT COMPARISON ({args.qubits} Qubits)")
    print("=" * 60)
    print(f"Qiskit Baseline: {baseline_rate:10.2f} vec/sec")
    print(f"PennyLane SOP:   {pl_rate:10.2f} vec/sec")
    print(f"Mahout QDP:      {mahout_rate:10.2f} vec/sec")
    print("-" * 60)

    if pl_rate > 0:
        print(f"Speedup vs PennyLane: {mahout_rate / pl_rate:.2f}x")
    if baseline_rate > 0:
        print(f"Speedup vs Qiskit:    {mahout_rate / baseline_rate:.2f}x")
