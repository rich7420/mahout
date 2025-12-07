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
Synthetic benchmark: Scaling & Fidelity tests.

Tests raw computation capability by comparing state vector preparation
time across different qubit counts (10-22 qubits).
"""

import time
import numpy as np
import torch
import qiskit
from qiskit.quantum_info import Statevector
import pennylane as qml
import pandas as pd
import matplotlib.pyplot as plt
from mahout_qdp import QdpEngine

# Constants
QUBIT_RANGE = range(10, 21, 1)  # 10, 12, ... 20 (20 qubits = 1M elements, stable)
TRIALS = 5


def benchmark_qiskit_statevector(data):
    """Benchmark Qiskit's direct Statevector loading (Memory Bound)."""
    start = time.perf_counter()
    _ = Statevector(data)
    return time.perf_counter() - start


def benchmark_qiskit_initialize(data, num_qubits):
    """Benchmark Qiskit's Initialize instruction (Compute Bound - Gate Decomposition)."""
    # Note: This is usually extremely slow for >15 qubits
    if num_qubits > 14:
        return None
    qc = qiskit.QuantumCircuit(num_qubits)
    start = time.perf_counter()
    qc.initialize(data, range(num_qubits))
    return time.perf_counter() - start


def benchmark_pennylane(data, num_qubits):
    """Benchmark PennyLane AmplitudeEmbedding (Framework Bound)."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(features=data, wires=range(num_qubits), normalize=True)
        return qml.state()

    start = time.perf_counter()
    _ = circuit()
    return time.perf_counter() - start


def benchmark_mahout(engine, data, num_qubits):
    """Benchmark Mahout QDP (GPU/PCIe Bound)."""
    start = time.perf_counter()
    # Returns DLPack capsule (zero-copy to Torch)
    qtensor = engine.encode(data, num_qubits, "amplitude")
    # We include the DLPack conversion cost to be fair
    _ = torch.from_dlpack(qtensor)
    torch.cuda.synchronize()  # Ensure GPU is done
    return time.perf_counter() - start


def calculate_fidelity_mahout(engine, data, num_qubits):
    """Calculate fidelity between expected and actual state vectors."""
    # Normalize input cpu data
    expected = np.array(data) / np.linalg.norm(data)

    qtensor = engine.encode(data, num_qubits, "amplitude")
    gpu_tensor = torch.from_dlpack(qtensor)

    # Move to CPU for comparison
    actual = gpu_tensor.cpu().numpy()

    # Fidelity = |<psi|phi>|^2
    overlap = np.vdot(expected, actual)  # Complex dot product
    return abs(overlap) ** 2


def run_benchmarks():
    print("Initializing Mahout Engine on GPU 0...")
    try:
        engine = QdpEngine(0)

        # Warm-up GPU to ensure CUDA context is fully initialized
        print("Warming up GPU...")
        dummy_data = [0.5, 0.5, 0.5, 0.5]
        _ = engine.encode(dummy_data, 2, "amplitude")
        torch.cuda.synchronize()
        print("Warm-up complete.\n")

    except Exception as e:
        print(f"Error initializing Mahout: {e}")
        return

    results = []
    print(
        f"{'Qubits':<10} | {'Dim':<10} | {'Mahout (ms)':<15} | {'Qiskit SV (ms)':<15} | {'PennyLane (ms)':<15} | {'Fidelity':<10}"
    )
    print("-" * 90)

    for n in QUBIT_RANGE:
        dim = 1 << n
        # Generate random data
        data = np.random.rand(dim).astype(np.float64).tolist()

        # 1. Mahout
        try:
            mahout_times = []
            for _ in range(TRIALS):
                mahout_times.append(benchmark_mahout(engine, data, n))
            mahout_avg = np.mean(mahout_times) * 1000

            # Fidelity Check
            fid = calculate_fidelity_mahout(engine, data, n)
        except Exception as e:
            print(f"    Error: Mahout failed for {n} qubits: {e}")
            mahout_avg = float("nan")
            fid = 0.0

        # 2. Qiskit Statevector
        qiskit_times = []
        for _ in range(TRIALS):
            qiskit_times.append(benchmark_qiskit_statevector(data))
        qiskit_avg = np.mean(qiskit_times) * 1000

        # 3. PennyLane
        # Skip PennyLane for very large qubits if it gets too slow
        if n <= 18:  # Reduced threshold as PennyLane becomes very slow
            try:
                pl_times = []
                for _ in range(max(1, TRIALS // 2)):  # Fewer trials for slow methods
                    pl_times.append(benchmark_pennylane(data, n))
                pl_avg = np.mean(pl_times) * 1000
            except Exception as e:
                print(f"    Warning: PennyLane failed for {n} qubits: {e}")
                pl_avg = float("nan")
        else:
            pl_avg = float("nan")

        print(
            f"{n:<10} | {dim:<10} | {mahout_avg:<15.4f} | {qiskit_avg:<15.4f} | {pl_avg:<15.4f} | {fid:<10.6f}"
        )

        results.append(
            {
                "qubits": n,
                "mahout": mahout_avg,
                "qiskit": qiskit_avg,
                "pennylane": pl_avg,
                "fidelity": fid,
            }
        )

    # Plotting
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df["qubits"], df["mahout"], marker="o", label="Mahout QDP (GPU)")
    plt.plot(df["qubits"], df["qiskit"], marker="x", label="Qiskit Statevector (CPU)")
    plt.plot(df["qubits"], df["pennylane"], marker="s", label="PennyLane (CPU/Default)")

    plt.yscale("log")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Time (ms) [Log Scale]")
    plt.title("Quantum State Preparation Latency vs Qubits")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig("benchmark_synthetic.png")
    print("\nPlot saved to benchmark_synthetic.png")


if __name__ == "__main__":
    run_benchmarks()
