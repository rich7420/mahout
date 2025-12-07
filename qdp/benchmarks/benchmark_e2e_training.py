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
End-to-End QNN Training Benchmark (PennyLane vs Mahout).

Simulates a real Quantum Neural Network training loop to demonstrate
Mahout's advantage in feeding GPU during training.

Key differences:
1. PennyLane: CPU encoding -> H2D transfer -> GPU training
2. Mahout: Direct GPU encoding -> Zero-copy PyTorch integration
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from mahout_qdp import QdpEngine

# Try importing PennyLane
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# Configuration
BATCH_SIZE = 128
EPOCHS = 5  # Run a few epochs to measure average speed
HIDDEN_DIM = 16


class HybridModel(nn.Module):
    """A simple Hybrid QNN model for testing Data Loading + Forward Pass"""

    def __init__(self, n_qubits):
        super().__init__()
        self.fc1 = nn.Linear(
            1 << n_qubits, HIDDEN_DIM
        )  # Simulate classical layer after quantum
        self.fc2 = nn.Linear(HIDDEN_DIM, 2)  # Binary Classification

    def forward(self, x):
        # x is already Quantum State Vector (Amplitude Encoded)
        # In real QNN, this would connect to Variational Layer
        # For pipeline testing, we simulate a simple matrix multiplication as "Quantum Layer"
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def benchmark_pennylane_training(n_qubits, n_samples):
    """PennyLane standard training workflow"""
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0

    print(f"\n[PennyLane] Starting End-to-End Training ({n_qubits} Qubits)...")

    # 1. Prepare data (CPU)
    data = np.random.rand(n_samples, 1 << n_qubits).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data = data / norms
    labels = torch.randint(0, 2, (n_samples,)).cuda()

    # PennyLane Device
    # Use lightning.gpu if available (fastest GPU backend)
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
    except Exception:
        print("Warning: lightning.gpu not found, falling back to default.qubit (CPU)")
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
        return qml.state()  # Return state for downstream processing

    model = HybridModel(n_qubits).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        # Simulate DataLoader batching
        for i in range(0, n_samples, BATCH_SIZE):
            batch_data = data[i : i + BATCH_SIZE]  # numpy slice
            batch_labels = labels[i : i + BATCH_SIZE]

            optimizer.zero_grad()

            # === PennyLane Key Bottleneck ===
            # 1. CPU Normalization (already done above, but simulate the cost)
            # 2. AmplitudeEmbedding (CPU computation) - This is expensive!
            # 3. Convert to PyTorch Tensor
            # 4. Transfer to GPU (H2D copy)

            # Simulate the real cost: CPU normalization + state prep
            # In reality, AmplitudeEmbedding would do this on CPU
            batch_normalized = batch_data.copy()
            batch_norms = np.linalg.norm(batch_normalized, axis=1, keepdims=True)
            batch_norms[batch_norms == 0] = 1.0
            batch_normalized = batch_normalized / batch_norms

            # Convert to PyTorch and transfer to GPU (H2D copy - this is expensive!)
            batch_tensor = torch.tensor(
                batch_normalized, device="cuda", dtype=torch.complex64
            )

            # In real PennyLane, this would go through QNode execution
            # We simulate by using the normalized data directly
            # (This is still lenient - real QNode would be slower)

            output = model(batch_tensor.abs())  # Simple abs into classical layer
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    throughput = (n_samples * EPOCHS) / total_time
    print(f"  PennyLane (Simulated Optimal): {throughput:.2f} samples/sec")
    return throughput


def benchmark_mahout_training(engine, n_qubits, n_samples):
    """Mahout GPU-accelerated training workflow"""
    print(f"\n[Mahout] Starting End-to-End Training ({n_qubits} Qubits)...")

    # 1. Prepare data (simulate from Disk/Parquet)
    # We directly generate a large numpy array to simulate raw data
    # Use float64 to match Parquet format, Mahout will convert internally
    raw_data_flat = np.random.rand(n_samples * (1 << n_qubits)).astype(np.float64)
    labels = torch.randint(0, 2, (n_samples,)).cuda()

    model = HybridModel(n_qubits).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    input_dim = 1 << n_qubits

    for epoch in range(EPOCHS):
        # Mahout DataLoader Loop
        for i in range(0, n_samples, BATCH_SIZE):
            # Simulate reading a chunk of Raw Data from Disk
            current_batch_size = min(BATCH_SIZE, n_samples - i)
            batch_slice = raw_data_flat[
                i * input_dim : (i + current_batch_size) * input_dim
            ]

            batch_labels = labels[i : i + current_batch_size]

            optimizer.zero_grad()

            # === Mahout Core Advantage ===
            # 1. Encode (Raw Data -> GPU State Vector)
            #    Includes: H2D Copy + L2 Norm + State Prep (all on GPU!)
            # Note: encode expects a list, but we can pass numpy array directly
            qtensor = engine.encode(batch_slice.tolist(), n_qubits, "amplitude")

            # 2. Zero-Copy conversion to PyTorch
            #    This is instant - no memory copy!
            gpu_state = torch.from_dlpack(qtensor)

            # 3. Reshape (because encode returns 1D flat buffer for single vector)
            #    For batch processing, we need to reshape
            #    Note: Currently encode processes as single vector, so we reshape accordingly
            state_dim = 1 << n_qubits
            if gpu_state.numel() >= current_batch_size * state_dim:
                gpu_state = gpu_state[: current_batch_size * state_dim].view(
                    current_batch_size, state_dim
                )
            else:
                # Single vector case
                gpu_state = gpu_state.view(1, -1)

            # 4. Forward Pass (all GPU)
            output = model(gpu_state.abs())
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    throughput = (n_samples * EPOCHS) / total_time
    print(f"  Mahout QDP: {throughput:.2f} samples/sec")
    return throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End QNN Training Benchmark")
    parser.add_argument("--qubits", type=int, default=18, help="Number of qubits")
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of training samples"
    )
    args = parser.parse_args()

    # Initialize Mahout (F32 Mode for max speed)
    try:
        engine = QdpEngine(0, precision="float32")
    except Exception as e:
        print(f"Error initializing Mahout: {e}")
        exit(1)

    # Run Benchmarks
    pl_rate = benchmark_pennylane_training(args.qubits, args.samples)
    mahout_rate = benchmark_mahout_training(engine, args.qubits, args.samples)

    print("\n" + "=" * 60)
    print(f"Training Throughput Comparison ({args.qubits} Qubits)")
    print("=" * 60)
    print(f"PennyLane (Simulated): {pl_rate:10.2f} samples/sec")
    print(f"Mahout QDP:            {mahout_rate:10.2f} samples/sec")
    print("-" * 60)
    if pl_rate > 0:
        print(f"Speedup:               {mahout_rate / pl_rate:10.2f}x")
    print("=" * 60)
    print("\nWhy Mahout wins in training?")
    print(" 1. Zero-copy: Data stays on GPU throughout training loop")
    print(" 2. Memory Pool: Eliminates cudaMalloc overhead per batch")
    print(" 3. GPU-native: Encoding happens on GPU, not CPU")
    print(" 4. Pipeline: Async operations hide PCIe latency")
