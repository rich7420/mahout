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
End-to-End "Honest" QNN Benchmark.

This benchmark removes all "simulated optimizations" for PennyLane.
It forces PennyLane to perform the actual embedding and state preparation
logic on the CPU (standard usage), revealing the true cost of
hybrid quantum-classical data loading.

Scenario:
1. PennyLane: Reads Parquet -> CPU Norm -> CPU State Prep -> PCIe -> GPU Model
2. Mahout: Reads Parquet -> GPU Async Pipeline -> GPU Model
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from mahout_qdp import QdpEngine

# Try importing PennyLane
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# Configuration
BATCH_SIZE = 128
EPOCHS = 3
HIDDEN_DIM = 16
DATA_FILE = "training_data_honest.parquet"


def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    _ = process.memory_info().rss / 1024 / 1024  # MB
    if torch.cuda.is_available():
        _ = torch.cuda.memory_allocated(0) / 1024 / 1024  # MB
    # print(f"[{tag}] RAM: {ram:.2f} MB | VRAM: {vram:.2f} MB")


class HybridModel(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.fc1 = nn.Linear(1 << n_qubits, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def generate_data(n_qubits, n_samples):
    if os.path.exists(DATA_FILE):
        # Simple check if file seems large enough
        if os.path.getsize(DATA_FILE) > 1024:
            return

    print(f"Generating synthetic parquet data: {DATA_FILE}...")
    dim = 1 << n_qubits
    chunk_size = 500
    schema = pa.schema([("feature_vector", pa.list_(pa.float64()))])

    with pq.ParquetWriter(DATA_FILE, schema) as writer:
        for start_idx in range(0, n_samples, chunk_size):
            current = min(chunk_size, n_samples - start_idx)
            data = np.random.rand(current, dim).astype(np.float64)
            # We do NOT normalize here. We let the framework do it.
            # This forces PennyLane to compute L2 norm on CPU.
            feature_vectors = [row.tolist() for row in data]
            arrays = pa.array(feature_vectors, type=pa.list_(pa.float64()))
            batch_table = pa.Table.from_arrays([arrays], names=["feature_vector"])
            writer.write_table(batch_table)

    print(f"  Generated {n_samples} samples with dimension {dim}")


def benchmark_pennylane_honest(n_qubits, n_samples):
    """PennyLane with actual QNode execution - no shortcuts"""
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0

    print("\n[PennyLane] Standard Workflow (Honest CPU Execution)...")

    # Standard CPU device (default.qubit is what 99% of users start with)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        # === The Bottleneck ===
        # PennyLane must calculate normalization and prepare state on CPU
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
        )
        return qml.state()

    model = HybridModel(n_qubits).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 2, (n_samples,)).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # 1. IO Phase (Pandas)
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data_cpu = np.stack(df["feature_vector"].values)

    print("  IO Finished. Starting Training Loop...")

    for epoch in range(EPOCHS):
        for i in range(0, n_samples, BATCH_SIZE):
            batch_data = torch.tensor(raw_data_cpu[i : i + BATCH_SIZE])  # CPU Tensor
            batch_labels = labels[i : i + BATCH_SIZE]

            optimizer.zero_grad()

            # 2. QNode Execution (CPU)
            # We must map the function over the batch if using default.qubit without batching support
            # Or use broadcasting if dimensions allow.
            # AmplitudeEmbedding usually supports broadcasting, let's try the fastest way:
            try:
                # Direct broadcast call (Fastest possible CPU path)
                state_cpu = circuit(batch_data)
            except Exception:
                # Fallback to loop if broadcasting fails (Slower but robust)
                state_cpu = torch.stack([circuit(x) for x in batch_data])

            # 3. Transfer to GPU (The Tax)
            state_gpu = state_cpu.to(
                "cuda", dtype=torch.float32
            )  # Using float32 for model

            # 4. Classical Forward
            output = model(state_gpu.abs())  # Simulated measurement processing
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    throughput = (n_samples * EPOCHS) / total_time
    print(f"  PennyLane Throughput: {throughput:.2f} samples/sec")
    return throughput


def benchmark_mahout_gpu(engine, n_qubits, n_samples):
    """Mahout GPU-accelerated workflow"""
    print("\n[Mahout] Accelerated Workflow (Direct GPU)...")

    model = HybridModel(n_qubits).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 2, (n_samples,)).cuda()

    state_dim = 1 << n_qubits

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # 1. One-Shot Load (Arrow -> GPU)
    # Using F32 Engine for maximum throughput
    # Mahout handles IO, Decompression, Normalization (GPU), and State Prep (GPU)
    qtensor = engine.encode_from_parquet(DATA_FILE, n_qubits, "amplitude")
    gpu_dataset = torch.from_dlpack(qtensor)

    # Reshape
    total_elements = gpu_dataset.numel()
    # Check if we have enough elements for reshape
    if total_elements % state_dim != 0:
        # Pad or truncate (should be handled by Rust, but safety check)
        valid_samples = total_elements // state_dim
        gpu_dataset = gpu_dataset[: valid_samples * state_dim]

    gpu_dataset = gpu_dataset.view(-1, state_dim)

    # Ensure data is ready
    torch.cuda.synchronize()
    print("  Data Loaded to GPU. Starting Training Loop...")

    for epoch in range(EPOCHS):
        # 2. Zero-Copy Loop
        # Everything is already on VRAM
        for i in range(0, n_samples, BATCH_SIZE):
            batch_tensor = gpu_dataset[i : i + BATCH_SIZE]
            if batch_tensor.shape[0] == 0:
                break

            batch_labels = labels[i : i + BATCH_SIZE]
            if batch_labels.shape[0] != batch_tensor.shape[0]:
                batch_labels = batch_labels[: batch_tensor.shape[0]]

            optimizer.zero_grad()

            # Forward (No transfer needed)
            output = model(batch_tensor.abs())
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    throughput = (n_samples * EPOCHS) / total_time
    print(f"  Mahout Throughput:    {throughput:.2f} samples/sec")
    return throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Honest End-to-End QNN Benchmark (No Shortcuts for PennyLane)"
    )
    # 18 Qubits = 262,144 dimensions.
    # PennyLane CPU will struggle with norm/state prep.
    # Mahout GPU F32 will fly.
    parser.add_argument(
        "--qubits", type=int, default=18, help="Number of qubits (18 recommended)"
    )
    parser.add_argument(
        "--samples", type=int, default=2000, help="Number of training samples"
    )
    args = parser.parse_args()

    # Generate fresh data
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    generate_data(args.qubits, args.samples)

    # Init Mahout (F32 Mode)
    try:
        # F32 is critical for 2x memory bandwidth
        engine = QdpEngine(0, precision="float32")
    except Exception as e:
        print(f"Mahout Init Error: {e}")
        exit(1)

    print("=" * 60)
    print(f"HONEST BENCHMARK: {args.qubits} Qubits, {args.samples} Samples")
    print("=" * 60)

    pl_rate = benchmark_pennylane_honest(args.qubits, args.samples)
    mahout_rate = benchmark_mahout_gpu(engine, args.qubits, args.samples)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"PennyLane: {pl_rate:10.2f} samples/sec")
    print(f"Mahout:    {mahout_rate:10.2f} samples/sec")
    print("-" * 60)

    if pl_rate > 0:
        speedup = mahout_rate / pl_rate
        print(f"Speedup:   {speedup:10.2f}x")
        if speedup > 50:
            print("\n🚀 Mahout achieves 50x+ speedup!")
            print("   This demonstrates the true cost of CPU-bound quantum operations.")
    else:
        print("Speedup:   Infinite (PennyLane Failed)")

    print("=" * 60)
    print("\nKey Insights:")
    print("  1. PennyLane forced to execute actual QNode on CPU")
    print("  2. CPU normalization + state prep is the bottleneck")
    print("  3. PCIe transfer adds significant overhead")
    print("  4. Mahout's GPU-native pipeline eliminates all CPU bottlenecks")
