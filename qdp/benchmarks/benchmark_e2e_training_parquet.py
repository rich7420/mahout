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
End-to-End QNN Training Benchmark using Parquet (PennyLane vs Mahout).

This benchmark demonstrates Mahout's true advantage by using Parquet I/O,
avoiding the Python List serialization bottleneck that occurs with .tolist().

Key differences:
1. PennyLane: Parquet -> Pandas -> NumPy -> CPU Normalization -> H2D -> GPU
2. Mahout: Parquet -> Rust Arrow (Zero-Copy) -> GPU (Direct)
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
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
DATA_FILE = "training_data.parquet"


class HybridModel(nn.Module):
    """A simple Hybrid QNN model for testing Data Loading + Forward Pass"""

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
    """Generate synthetic Parquet data file"""
    print(f"Generating synthetic parquet data: {DATA_FILE}...")
    dim = 1 << n_qubits
    # Generate chunks to avoid RAM explosion
    chunk_size = 500

    schema = pa.schema([("feature_vector", pa.list_(pa.float64()))])

    with pq.ParquetWriter(DATA_FILE, schema) as writer:
        for start_idx in range(0, n_samples, chunk_size):
            current = min(chunk_size, n_samples - start_idx)
            data = np.random.rand(current, dim).astype(np.float64)
            # Note: We don't normalize here - let each framework do it
            # This simulates raw data from disk

            # Convert to PyArrow format (same as generate_dataset.py)
            feature_vectors = [row.tolist() for row in data]
            arrays = pa.array(feature_vectors, type=pa.list_(pa.float64()))
            batch_table = pa.Table.from_arrays([arrays], names=["feature_vector"])
            writer.write_table(batch_table)

    print(f"  Generated {n_samples} samples with dimension {dim}")


def benchmark_pennylane_training(n_qubits, n_samples):
    """PennyLane training from Parquet (standard workflow)"""
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0

    print(f"\n[PennyLane] Training from Parquet ({n_qubits} Qubits)...")

    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
    except Exception:
        print("Warning: lightning.gpu not found, falling back to default.qubit (CPU)")
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
        return qml.state()

    model = HybridModel(n_qubits).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # IO Load (Simulating standard PyTorch Dataset __getitem__)
    # We read entire file to memory for simple iteration, but convert batch-by-batch
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data = np.stack(df["feature_vector"].values)
    labels = torch.randint(0, 2, (n_samples,)).cuda()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        for i in range(0, n_samples, BATCH_SIZE):
            # 1. Get Batch (CPU)
            batch_data = raw_data[i : i + BATCH_SIZE]
            batch_labels = labels[i : i + BATCH_SIZE]

            optimizer.zero_grad()

            # 2. Preprocess & Transfer (The Bottleneck)
            # Normalize on CPU
            norms = np.linalg.norm(batch_data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            batch_data = batch_data / norms

            # Transfer to GPU (H2D copy - expensive!)
            batch_tensor = torch.tensor(
                batch_data, device="cuda", dtype=torch.complex64
            )

            # 3. Forward
            output = model(batch_tensor.abs())
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    throughput = (n_samples * EPOCHS) / total_time
    print(f"  PennyLane: {throughput:.2f} samples/sec")
    return throughput


def benchmark_mahout_training(engine, n_qubits, n_samples):
    """Mahout training from Parquet (GPU-accelerated I/O)"""
    print(f"\n[Mahout] Training from Parquet ({n_qubits} Qubits)...")

    model = HybridModel(n_qubits).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 2, (n_samples,)).cuda()

    # Mahout doesn't need to preload data to RAM using Pandas
    # It streams from Parquet directly to GPU
    # NOTE: Our current Python binding for encode_from_parquet reads the WHOLE file.
    # For this benchmark to be a loop, we should technically use a chunked reader.
    # BUT, since we want to measure throughput, reading the whole file once and then
    # encoding it is a valid test of the "Data Loader" speed if we include IO time.

    # However, to be strictly comparable to the loop above, we will simulate
    # the behavior where `encode_from_parquet` is called per batch (if we split files)
    # OR we just measure the total throughput of processing the whole file.

    # Let's measure processing the whole dataset X times (Epochs)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        # In a real impl, we would iterate chunks.
        # Here, engine.encode_from_parquet reads the file and returns one giant GPU tensor.
        # This is essentially "Full Batch" loading or "Pre-loading to GPU".
        # This is VALID for Mahout because it supports large GPU allocations.

        # === Mahout Core Advantage ===
        # 1. IO + Encode + Transfer (All in one Rust call)
        #    Parquet -> Rust Arrow (Zero-Copy) -> GPU
        #    No Python List serialization!
        #    No CPU normalization overhead!
        qtensor = engine.encode_from_parquet(DATA_FILE, n_qubits, "amplitude")
        gpu_all_data = torch.from_dlpack(qtensor)

        # Reshape to (N, Dim)
        # Note: encode_from_parquet returns flat 1D array
        state_dim = 1 << n_qubits
        n_loaded = gpu_all_data.numel() // state_dim
        gpu_all_data = gpu_all_data.view(n_loaded, state_dim)

        # 2. Training Loop (on GPU - all data already on GPU!)
        for i in range(0, n_loaded, BATCH_SIZE):
            batch_tensor = gpu_all_data[i : i + BATCH_SIZE]
            batch_labels = labels[i : i + BATCH_SIZE]

            optimizer.zero_grad()
            output = model(batch_tensor.abs())
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    throughput = (n_samples * EPOCHS) / total_time
    print(f"  Mahout QDP: {throughput:.2f} samples/sec")
    return throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-End QNN Training Benchmark (Parquet I/O)"
    )
    parser.add_argument("--qubits", type=int, default=16, help="Number of qubits")
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove generated Parquet file after benchmark",
    )
    args = parser.parse_args()

    # Generate Data
    if os.path.exists(DATA_FILE):
        print(f"Using existing {DATA_FILE}")
    else:
        generate_data(args.qubits, args.samples)

    # Initialize Mahout (F32 Mode for max speed)
    try:
        engine = QdpEngine(0, precision="float32")
    except Exception as e:
        print(f"Error initializing Mahout: {e}")
        exit(1)

    # Run Benchmarks
    pl_rate = benchmark_pennylane_training(args.qubits, args.samples)
    mahout_rate = benchmark_mahout_training(engine, args.qubits, args.samples)

    # Cleanup
    if args.cleanup and os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        print(f"\nCleaned up {DATA_FILE}")

    print("\n" + "=" * 60)
    print("Training Throughput (Parquet I/O + Training)")
    print(f"  {args.qubits} Qubits, {args.samples} Samples, {EPOCHS} Epochs")
    print("=" * 60)
    print(f"PennyLane:   {pl_rate:10.2f} samples/sec")
    print(f"Mahout QDP:  {mahout_rate:10.2f} samples/sec")
    print("-" * 60)
    if pl_rate > 0:
        print(f"Speedup:     {mahout_rate / pl_rate:10.2f}x")
    print("=" * 60)
    print("\nWhy Mahout wins with Parquet?")
    print(" 1. Zero-Copy I/O: Parquet -> Rust Arrow -> GPU (no Python List!)")
    print(" 2. GPU-native normalization: Happens on GPU, not CPU")
    print(" 3. Memory Pool: Reuses staging buffers across epochs")
    print(" 4. Direct GPU allocation: No intermediate CPU copies")
