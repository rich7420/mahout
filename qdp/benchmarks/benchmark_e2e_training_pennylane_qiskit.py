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
The Ultimate QML Benchmark: Mahout vs PennyLane vs Qiskit.

This benchmark exposes the "Elephant in the Room":
Classic frameworks (Qiskit/PennyLane) are not designed for High-Dimension Data Loading.

1. Qiskit SOP (The Victim):
   - Uses `qc.initialize()` which tries to synthesize gates (O(2^N) complexity).
   - Suffers from Python loop overhead + Math overhead + Transpilation.

2. PennyLane SOP (The Struggle):
   - Uses `AmplitudeEmbedding` which does CPU Normalization.
   - Suffers from PCIe bottleneck.

3. Mahout SOP (The Apex Predator):
   - Uses Zero-Copy Parquet + GPU Async Pipeline.
   - Bypasses CPU computation entirely.
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

# --- Import Competitors ---
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

# Configuration
BATCH_SIZE = 64
EPOCHS = 2
HIDDEN_DIM = 16
DATA_FILE = "training_data_v4.parquet"


# --- Model & Utils ---
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
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 1024:
        return

    print(f"Generating synthetic parquet data: {DATA_FILE}...")
    dim = 1 << n_qubits
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

    print(f"  Generated {n_samples} samples with dimension {dim}")


# -----------------------------------------------------------------------------
# 1. Qiskit SOP (The Hardest Way)
# -----------------------------------------------------------------------------
def benchmark_qiskit_honest(n_qubits, n_samples):
    if not HAS_QISKIT:
        print("\n[Qiskit] Not installed, skipping.")
        return 0.0

    print(f"\n[Qiskit] Standard SOP ({n_qubits} Qubits)...")
    print("  * Warning: Qiskit `initialize()` is O(2^N) compute bound.")
    print("  * This might be very slow.")

    model = HybridModel(n_qubits).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 2, (n_samples,)).cuda()

    # IO Load
    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data_cpu = np.stack(df["feature_vector"].values)

    # Simulator
    backend = AerSimulator(method="statevector")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        for i in range(0, n_samples, BATCH_SIZE):
            batch_data = raw_data_cpu[i : i + BATCH_SIZE]
            batch_labels = labels[i : i + BATCH_SIZE]
            optimizer.zero_grad()

            # === Qiskit Bottleneck ===
            batch_states = []

            # Loop overhead (Qiskit doesn't batch initialize)
            for vec_idx, vec in enumerate(batch_data):
                # 1. CPU Norm
                norm = np.linalg.norm(vec)
                vec = vec / (norm if norm > 0 else 1.0)

                # 2. Circuit Construction (Expensive!)
                # Qiskit calculates gate decomposition here
                qc = QuantumCircuit(n_qubits)
                qc.initialize(vec, range(n_qubits))
                qc.save_statevector()

                # 3. Transpile & Run (Expensive!)
                # Even with reuse, parameter binding is slow for this size
                t_qc = transpile(qc, backend)
                job = backend.run(t_qc)

                # 4. Extract Result (CPU RAM)
                result = job.result().get_statevector().data
                batch_states.append(result)

                # Progress indicator because Qiskit is slow
                if (vec_idx + 1) % 10 == 0:
                    print(
                        f"    Qiskit processed {vec_idx + 1}/{len(batch_data)} vectors...",
                        end="\r",
                    )

            # 5. Stack & Transfer to GPU (The Tax)
            # Complex128 -> GPU
            gpu_tensor = torch.tensor(
                np.array(batch_states), device="cuda", dtype=torch.complex64
            )

            # 6. Forward
            output = model(gpu_tensor.abs())
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    throughput = (n_samples * EPOCHS) / total_time
    print(f"\n  Qiskit Throughput:    {throughput:.2f} samples/sec")
    return throughput


# -----------------------------------------------------------------------------
# 2. PennyLane SOP
# -----------------------------------------------------------------------------
def benchmark_pennylane_honest(n_qubits, n_samples):
    if not HAS_PENNYLANE:
        print("\n[PennyLane] Not installed, skipping.")
        return 0.0

    print(f"\n[PennyLane] Standard SOP ({n_qubits} Qubits)...")

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
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

    import pandas as pd

    df = pd.read_parquet(DATA_FILE)
    raw_data_cpu = np.stack(df["feature_vector"].values)

    for epoch in range(EPOCHS):
        for i in range(0, n_samples, BATCH_SIZE):
            batch_data = torch.tensor(raw_data_cpu[i : i + BATCH_SIZE])
            batch_labels = labels[i : i + BATCH_SIZE]
            optimizer.zero_grad()

            # PennyLane CPU Execution
            try:
                state_cpu = circuit(batch_data)
            except Exception:
                state_cpu = torch.stack([circuit(x) for x in batch_data])

            # H2D Copy
            state_gpu = state_cpu.to("cuda", dtype=torch.float64)

            output = model(state_gpu.abs())
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    throughput = (n_samples * EPOCHS) / total_time
    print(f"  PennyLane Throughput: {throughput:.2f} samples/sec")
    return throughput


# -----------------------------------------------------------------------------
# 3. Mahout QDP SOP
# -----------------------------------------------------------------------------
def benchmark_mahout_gpu(engine, n_qubits, n_samples):
    print(f"\n[Mahout] Accelerated SOP ({n_qubits} Qubits)...")

    model = HybridModel(n_qubits).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 2, (n_samples,)).cuda()

    state_dim = 1 << n_qubits

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # One-Shot Load (Arrow -> GPU)
    qtensor = engine.encode_from_parquet(DATA_FILE, n_qubits, "amplitude")
    gpu_dataset = torch.from_dlpack(qtensor)

    # Handle padding/reshaping
    total_elements = gpu_dataset.numel()
    valid_samples = total_elements // state_dim
    gpu_dataset = gpu_dataset[: valid_samples * state_dim].view(-1, state_dim)

    # Training Loop (Pure VRAM)
    for epoch in range(EPOCHS):
        for i in range(0, n_samples, BATCH_SIZE):
            if i >= valid_samples:
                break
            batch_tensor = gpu_dataset[i : i + BATCH_SIZE]
            batch_labels = labels[i : i + BATCH_SIZE]

            if batch_labels.shape[0] != batch_tensor.shape[0]:
                batch_labels = batch_labels[: batch_tensor.shape[0]]

            optimizer.zero_grad()
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
        description="Ultimate QML Benchmark: Mahout vs PennyLane vs Qiskit"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=16,
        help="Number of qubits (16 recommended, 18+ may be very slow for Qiskit)",
    )
    parser.add_argument(
        "--samples", type=int, default=500, help="Number of training samples"
    )
    args = parser.parse_args()

    # Regenerate data to match qubits (Essential for fairness)
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    generate_data(args.qubits, args.samples)

    try:
        engine = QdpEngine(0)
    except Exception as e:
        print(f"Mahout Init Error: {e}")
        exit(1)

    print("=" * 70)
    print(f"ULTIMATE QML BENCHMARK: {args.qubits} Qubits, {args.samples} Samples")
    print("=" * 70)

    pl_rate = benchmark_pennylane_honest(args.qubits, args.samples)
    qiskit_rate = benchmark_qiskit_honest(args.qubits, args.samples)
    mahout_rate = benchmark_mahout_gpu(engine, args.qubits, args.samples)

    print("\n" + "=" * 70)
    print("FINAL STANDINGS (Throughput: Higher is Better)")
    print("=" * 70)

    # Sort by throughput
    results = []
    if mahout_rate > 0:
        results.append(("Mahout QDP", mahout_rate))
    if pl_rate > 0:
        results.append(("PennyLane", pl_rate))
    if qiskit_rate > 0:
        results.append(("Qiskit", qiskit_rate))

    results.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, rate) in enumerate(results, 1):
        print(f"{rank}. {name:15s} {rate:10.2f} samples/sec")

    print("-" * 70)

    if pl_rate > 0 and mahout_rate > 0:
        print(f"Mahout vs PennyLane: {mahout_rate / pl_rate:10.2f}x")
    if qiskit_rate > 0 and mahout_rate > 0:
        print(f"Mahout vs Qiskit:    {mahout_rate / qiskit_rate:10.2f}x")
    if qiskit_rate > 0 and pl_rate > 0:
        print(f"PennyLane vs Qiskit: {pl_rate / qiskit_rate:10.2f}x")

    print("=" * 70)
    print("\nKey Insights:")
    print("  1. Qiskit's `initialize()` requires O(2^N) gate synthesis")
    print("  2. PennyLane's CPU normalization + PCIe transfer is the bottleneck")
    print("  3. Mahout's GPU-native pipeline eliminates all CPU overhead")
    print("  4. This demonstrates why GPU acceleration is essential for QML")
