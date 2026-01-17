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
MNIST Image Classification Benchmark: QDP vs PennyLane

This benchmark demonstrates the performance advantage of QDP's GPU-accelerated
amplitude encoding compared to PennyLane's CPU-based approach.

Dataset: MNIST Hand-written Digits
- Input: 28x28 grayscale images (784 features)
- Quantum bits: 10 qubits (2^10 = 1024, sufficient for 784 pixels with padding)
- Encoding: Amplitude encoding with L2 normalization

Key Performance Factors:
1. Batch Processing: QDP processes entire batch in single GPU kernel launch
2. GPU Parallelism: Normalization and padding done in parallel on GPU
3. Zero-Copy Integration: Direct GPU memory sharing via DLPack
4. Memory Efficiency: Single allocation for batch vs multiple allocations
"""

import time
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import gc

# QDP imports
from _qdp import QdpEngine

# Optional: PennyLane for comparison
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("Warning: PennyLane not available. Skipping PennyLane benchmark.")

# Configuration
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_QUBITS = 10
DEFAULT_NUM_BATCHES = 5  # Number of batches to process for averaging


def clean_gpu_cache():
    """Clear GPU cache and Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def prepare_mnist_data(batch_size, num_batches=1):
    """
    Download and prepare MNIST dataset.

    Args:
        batch_size: Number of samples per batch
        num_batches: Number of batches to prepare

    Returns:
        List of (images, labels) tuples, where images are numpy arrays
    """
    print(f"Downloading MNIST dataset (batch_size={batch_size})...")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Note: We don't normalize here because amplitude encoding
            # requires specific L2 normalization that QDP/PennyLane will handle
        ]
    )

    # Suppress download progress bar by redirecting stdout temporarily
    import sys
    from io import StringIO

    # Download training set (suppress progress output)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    finally:
        sys.stdout = old_stdout

    data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Collect batches
    batches = []
    for i, (images, labels) in enumerate(data_loader):
        if i >= num_batches:
            break
        # Flatten images: (batch_size, 28, 28) -> (batch_size, 784)
        X_batch = images.view(batch_size, -1).numpy().astype(np.float64)
        batches.append((X_batch, labels))

    print(f"Prepared {len(batches)} batches of shape {batches[0][0].shape}")
    return batches


def benchmark_pennylane(batches, n_qubits):
    """
    Benchmark PennyLane's AmplitudeEmbedding (CPU-based).

    Note: PennyLane's AmplitudeEmbedding doesn't natively support batch processing,
    so we need to loop over samples, which is a significant bottleneck.

    Args:
        batches: List of (images, labels) tuples
        n_qubits: Number of qubits for encoding

    Returns:
        Total time, average time per sample, list of encoded states
    """
    if not HAS_PENNYLANE:
        return None, None, None

    print("\n" + "=" * 60)
    print("Benchmark: PennyLane (AmplitudeEmbedding)")
    print("=" * 60)

    # Use lightning.qubit for better performance (C++ backend)
    # Still slower than QDP due to CPU-based normalization and lack of batching
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        # Fallback to default.qubit if lightning.qubit is not available
        print("Warning: lightning.qubit not available, using default.qubit")
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(features):
        """Quantum circuit for amplitude encoding."""
        qml.AmplitudeEmbedding(
            features=features, wires=range(n_qubits), pad_with=0.0, normalize=True
        )
        return qml.state()  # Return state vector

    clean_gpu_cache()

    all_states = []
    start_time = time.time()

    # Process each batch
    for batch_idx, (X_batch, labels) in enumerate(batches):
        batch_states = []

        # PennyLane doesn't support batch processing natively for AmplitudeEmbedding,
        # so we need to loop over each sample
        # Note: Some PennyLane versions may support batching, but it's not reliable
        for i in range(X_batch.shape[0]):
            state = circuit(features=X_batch[i])
            batch_states.append(state)

        # Convert to numpy array
        # qml.state() returns a 1D array of complex numbers
        batch_tensor = np.array(batch_states)
        all_states.append(batch_tensor)

        if (batch_idx + 1) % 1 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Processed batch {batch_idx + 1}/{len(batches)} "
                f"({elapsed:.2f}s elapsed)"
            )

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    total_samples = sum(batch[0].shape[0] for batch in batches)
    avg_time_per_sample = total_time / total_samples

    print("\nResults:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Total samples: {total_samples}")
    print(f"  Average time per sample: {avg_time_per_sample * 1000:.2f} ms")
    print(f"  Throughput: {total_samples / total_time:.2f} samples/second")

    return total_time, avg_time_per_sample, all_states


def benchmark_qdp(batches, n_qubits, precision="float32", device_id=0):
    """
    Benchmark QDP's GPU-accelerated amplitude encoding.

    Args:
        batches: List of (images, labels) tuples
        n_qubits: Number of qubits for encoding
        precision: Output precision ("float32" or "float64")
        device_id: CUDA device ID

    Returns:
        Total time, average time per sample, list of PyTorch tensors
    """
    print("\n" + "=" * 60)
    print(f"Benchmark: QDP (GPU-Accelerated, precision={precision})")
    print("=" * 60)

    # Initialize QDP engine
    engine = QdpEngine(device_id=device_id, precision=precision)

    clean_gpu_cache()

    all_tensors = []
    start_time = time.time()

    # Process each batch
    for batch_idx, (X_batch, labels) in enumerate(batches):
        # QDP's encode() automatically detects 2D NumPy arrays and uses batch encoding
        # This is a single GPU kernel launch for the entire batch!
        qtensor = engine.encode(
            X_batch, num_qubits=n_qubits, encoding_method="amplitude"
        )

        # Convert to PyTorch tensor via DLPack (zero-copy)
        torch_tensor = torch.from_dlpack(qtensor)

        # Ensure GPU synchronization for accurate timing
        torch.cuda.synchronize()

        all_tensors.append(torch_tensor)

        if (batch_idx + 1) % 1 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Processed batch {batch_idx + 1}/{len(batches)} "
                f"({elapsed:.2f}s elapsed)"
            )
            print(f"    Output shape: {torch_tensor.shape}")
            print(f"    Output dtype: {torch_tensor.dtype}")
            print(f"    Device: {torch_tensor.device}")

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    total_samples = sum(batch[0].shape[0] for batch in batches)
    avg_time_per_sample = total_time / total_samples

    print("\nResults:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Total samples: {total_samples}")
    print(f"  Average time per sample: {avg_time_per_sample * 1000:.2f} ms")
    print(f"  Throughput: {total_samples / total_time:.2f} samples/second")

    return total_time, avg_time_per_sample, all_tensors


def verify_outputs(qdp_tensors, pennylane_states, n_qubits):
    """
    Verify that QDP and PennyLane produce similar outputs.

    Args:
        qdp_tensors: List of PyTorch tensors from QDP
        pennylane_states: List of numpy arrays from PennyLane
        n_qubits: Number of qubits
    """
    if pennylane_states is None:
        print("\nSkipping verification (PennyLane not available)")
        return

    print("\n" + "=" * 60)
    print("Verification: Comparing QDP and PennyLane outputs")
    print("=" * 60)

    # Compare first batch
    qdp_batch = qdp_tensors[0].cpu().numpy()
    pl_batch = pennylane_states[0]

    print(f"QDP output shape: {qdp_batch.shape}")
    print(f"PennyLane output shape: {pl_batch.shape}")

    # Check normalization (both should be L2 normalized)
    # For complex arrays, we need to compute the norm correctly
    qdp_norms = np.linalg.norm(qdp_batch, axis=1)  # Works for complex arrays
    pl_norms = np.linalg.norm(pl_batch, axis=1)  # Works for complex arrays

    print("\nNormalization check:")
    print(
        f"  QDP norms (should be ~1.0): mean={qdp_norms.mean():.6f}, "
        f"std={qdp_norms.std():.6f}"
    )
    print(
        f"  PennyLane norms (should be ~1.0): mean={pl_norms.mean():.6f}, "
        f"std={pl_norms.std():.6f}"
    )

    # Compare first sample
    # Note: QDP returns complex64 (float32 precision), PennyLane returns complex128
    # We need to convert QDP's complex64 to complex128 for fair comparison
    qdp_sample = qdp_batch[0].astype(np.complex128)
    pl_sample = pl_batch[0].astype(np.complex128)  # Ensure same dtype

    # Calculate relative difference
    diff = np.abs(qdp_sample - pl_sample)
    rel_diff = diff / (np.abs(pl_sample) + 1e-10)

    print("\nFirst sample comparison:")
    print(f"  Max absolute difference: {diff.max():.6e}")
    print(f"  Mean relative difference: {rel_diff.mean():.6e}")
    print(f"  Max relative difference: {rel_diff.max():.6e}")

    # Check if outputs are similar (within numerical precision)
    max_diff = diff.max()
    if max_diff < 1e-4:
        print(f"\n✓ Outputs match within tolerance (max diff: {max_diff:.6e})")
    else:
        print(f"\n⚠ Outputs differ (max diff: {max_diff:.6e})")
        print("  Note: Some difference is expected due to:")
        print("    - Different precision (QDP: float32, PennyLane: float64)")
        print("    - Different normalization implementations")
        print("    - GPU vs CPU numerical differences")


def main():
    parser = argparse.ArgumentParser(
        description="MNIST Benchmark: QDP vs PennyLane Amplitude Encoding"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=DEFAULT_N_QUBITS,
        help=f"Number of qubits (default: {DEFAULT_N_QUBITS}, 2^{DEFAULT_N_QUBITS}={1 << DEFAULT_N_QUBITS})",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=DEFAULT_NUM_BATCHES,
        help=f"Number of batches to process (default: {DEFAULT_NUM_BATCHES})",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Output precision (default: float32)",
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--skip-pennylane",
        action="store_true",
        help="Skip PennyLane benchmark (faster execution)",
    )
    parser.add_argument(
        "--skip-verification", action="store_true", help="Skip output verification"
    )

    args = parser.parse_args()

    # Validate qubits
    state_len = 1 << args.n_qubits
    mnist_features = 28 * 28  # 784
    if state_len < mnist_features:
        print(
            f"Warning: State length ({state_len}) < MNIST features ({mnist_features})"
        )
        print("         Padding will be used, but consider increasing --n-qubits")

    print("=" * 60)
    print("MNIST Amplitude Encoding Benchmark")
    print("=" * 60)
    print("Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of qubits: {args.n_qubits} (state length: {state_len})")
    print(f"  Number of batches: {args.num_batches}")
    print(f"  Precision: {args.precision}")
    print(f"  CUDA device: {args.device_id}")
    print("=" * 60)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. QDP requires GPU.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(args.device_id)}")

    # Prepare data
    batches = prepare_mnist_data(args.batch_size, args.num_batches)

    # Benchmark QDP
    qdp_time, qdp_avg, qdp_tensors = benchmark_qdp(
        batches, args.n_qubits, args.precision, args.device_id
    )

    # Benchmark PennyLane (optional)
    pl_time, pl_avg, pl_states = None, None, None
    if not args.skip_pennylane and HAS_PENNYLANE:
        pl_time, pl_avg, pl_states = benchmark_pennylane(batches, args.n_qubits)

    # Verification
    if not args.skip_verification:
        verify_outputs(qdp_tensors, pl_states, args.n_qubits)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("QDP Performance:")
    print(f"  Total time: {qdp_time:.4f} seconds")
    print(f"  Average per sample: {qdp_avg * 1000:.2f} ms")

    if pl_time is not None:
        speedup = pl_time / qdp_time
        print("\nPennyLane Performance:")
        print(f"  Total time: {pl_time:.4f} seconds")
        print(f"  Average per sample: {pl_avg * 1000:.2f} ms")
        print(f"\n🚀 QDP is {speedup:.2f}x faster than PennyLane!")

        if speedup > 50:
            print("  This demonstrates the power of GPU-accelerated batch processing!")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
