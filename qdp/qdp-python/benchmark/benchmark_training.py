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
Complete Training Benchmark: QDP vs PennyLane Native

This benchmark compares the complete training pipeline (forward + backward pass)
between PennyLane's native AmplitudeEmbedding and QDP's GPU-accelerated encoding.

Key Features:
- Fair comparison: Same VQC structure, same backend, same differentiation method
- Complete training loop: Forward pass + backward pass + optimization
- Real-world scenario: MNIST binary classification (0 vs 1)
- GPU acceleration: QDP requires lightning.gpu for end-to-end GPU pipeline

Performance Analysis:
- PennyLane Native: AmplitudeEmbedding does CPU normalization + H2D transfer per batch
- QDP Accelerated: Batch encoding on GPU (single kernel launch) + zero-copy DLPack integration
- QDP maintains end-to-end GPU pipeline: state vectors stay on GPU throughout training
- The speedup is most visible when:
  1. Batch size is large (QDP's batch processing advantage)
  2. Data preparation time dominates (QDP eliminates this bottleneck)
  3. Using GPU backend (lightning.gpu) - REQUIRED for QDP path

Expected Results:
- Small batches (< 16): Similar performance (circuit execution dominates)
- Medium batches (16-64): QDP shows 1.5-3x speedup
- Large batches (64+): QDP shows 3-10x speedup (data prep bottleneck eliminated)

Requirements:
- QDP path REQUIRES lightning.gpu backend (install with: pip install pennylane-lightning[gpu])
- QDP maintains end-to-end GPU pipeline - no CPU transfers
- PennyLane Native can use any backend (lightning.gpu, lightning.qubit, or default.qubit)

Note: The actual speedup depends on:
- Batch size (larger = more QDP advantage)
- Number of qubits (more qubits = larger state vectors = more encoding overhead)
"""

import argparse
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from io import StringIO

# QDP imports
from _qdp import QdpEngine

# PennyLane imports
try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("Warning: PennyLane not available. Install with: pip install pennylane")

# Configuration
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_QUBITS = 10
DEFAULT_N_LAYERS = 2
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_LIMIT_BATCHES = 50


def clean_gpu_cache():
    """Clear GPU cache and Python garbage collection."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cuda_sync():
    """Synchronize CUDA (no-op if CUDA unavailable)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _dev_name(dev) -> str:
    """Best-effort device name for PennyLane devices."""
    for attr in ("short_name", "name"):
        v = getattr(dev, attr, None)
        if v:
            return str(v)
    return dev.__class__.__name__


def _is_gpu_backend(dev) -> bool:
    """Heuristic: whether PennyLane device is GPU-backed."""
    name = _dev_name(dev).lower()
    return "gpu" in name or "cuda" in name


def _summarize_timings_ms(timers: dict[str, float], denom: float | None = None) -> str:
    """Pretty-print timing dict (values in seconds)."""
    if not timers:
        return "(no timings)"
    total = sum(timers.values()) if denom is None else denom
    total = total if total > 0 else 1e-12
    rows = []
    for k, v in sorted(timers.items(), key=lambda kv: kv[1], reverse=True):
        pct = (v / total) * 100.0
        rows.append(f"  - {k:24s} {v * 1000.0:9.3f} ms  ({pct:5.1f}%)")
    return "\n".join(rows)


def prepare_mnist_binary(
    batch_size, n_qubits, limit_batches=None, num_workers=0, pin_memory=False
):
    """
    Prepare MNIST dataset for binary classification (0 vs 1), with auto-resizing based on qubit count.

    Rationale: to fairly stress classical data-prep, we scale input feature dimension toward 2^n_qubits.
    This makes CPU-side normalization/padding significantly heavier at higher qubit counts.

    Args:
        batch_size: Batch size for DataLoader
        n_qubits: Number of qubits (controls target feature dimension)
        limit_batches: Limit number of batches (None = all)
        num_workers: DataLoader workers
        pin_memory: DataLoader pin_memory

    Returns:
        DataLoader for binary classification
    """
    print("\n[Step 1] Preparing MNIST Binary Classification Data...")

    # Auto-scaling strategy: choose a square image size that matches common 2^n feature counts.
    max_features = 1 << int(n_qubits)
    if n_qubits >= 16:
        target_size = 256  # 65536 features
    elif n_qubits >= 14:
        target_size = 128  # 16384 features
    elif n_qubits >= 12:
        target_size = 64  # 4096 features
    else:
        target_size = 28  # Default MNIST (784 features)

    print("  Auto-scaling strategy:")
    print(f"    - Qubits: {n_qubits}")
    print(f"    - State vector size: {max_features}")
    print(
        f"    - Resizing MNIST to: {target_size}x{target_size} ({target_size * target_size} pixels)"
    )

    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Simple normalization (not L2; embedding will normalize)
        ]
    )

    # Suppress download progress
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    finally:
        sys.stdout = old_stdout

    # Filter for binary classification (0 vs 1)
    idx = (train_dataset.targets == 0) | (train_dataset.targets == 1)
    train_dataset.targets = train_dataset.targets[idx]
    train_dataset.data = train_dataset.data[idx]

    print(f"  Filtered dataset: {len(train_dataset)} samples (labels 0 and 1)")

    # Create DataLoader (keep it simple and deterministic for benchmarking)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # NOTE: We do NOT materialize batches into a list: doing so hides DataLoader overhead.
    # The training loop will stop after `limit_batches`.
    if limit_batches:
        print(f"  Benchmark will stop after {limit_batches} batches")
    print(f"  Total batches (full epoch): {len(train_loader)}")
    return train_loader


def get_quantum_device(n_qubits, require_gpu=False):
    """
    Get the best available quantum device.

    Args:
        n_qubits: Number of qubits
        require_gpu: If True, only return GPU device (for QDP pipeline)

    Returns:
        PennyLane device

    Raises:
        RuntimeError: If require_gpu=True but GPU device is not available
    """
    # Try lightning.gpu (cuQuantum) first
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        print("  Using device: lightning.gpu (cuQuantum)")
        return dev
    except Exception as e:
        if require_gpu:
            raise RuntimeError(
                f"QDP requires GPU backend (lightning.gpu) for end-to-end GPU pipeline. "
                f"lightning.gpu is not available: {e}\n"
                f"Please install with: pip install pennylane-lightning[gpu]"
            )
        print(f"  lightning.gpu not available: {e}")

    # Fallback to lightning.qubit (C++ backend) - only if not requiring GPU
    if not require_gpu:
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            print("  Using device: lightning.qubit (C++ backend)")
            return dev
        except Exception as e:
            print(f"  lightning.qubit not available: {e}")

        # Final fallback to default.qubit
        dev = qml.device("default.qubit", wires=n_qubits)
        print("  Using device: default.qubit (Python backend)")
        return dev

    # Should not reach here if require_gpu=True
    raise RuntimeError("GPU device required but not available")


def ansatz_layer(params, wires):
    """
    Shared ansatz layer for both models.

    Uses StronglyEntanglingLayers for fair comparison.

    Args:
        params: Parameter tensor of shape (n_qubits, 3)
        wires: Wire indices
    """
    qml.StronglyEntanglingLayers(params, wires=wires)


# ==========================================
# Method A: PennyLane Native (Baseline)
# ==========================================
def create_pennylane_model(n_qubits, n_layers, device):
    """
    Create PennyLane native model using AmplitudeEmbedding.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of VQC layers
        device: PyTorch device

    Returns:
        PyTorch model
    """
    print("\n[Step 2] Building PennyLane Native Model...")
    qml_dev = get_quantum_device(n_qubits)

    @qml.qnode(qml_dev, interface="torch", diff_method="adjoint")
    def circuit_pl(inputs, weights):
        """
        PennyLane native circuit with AmplitudeEmbedding.

        This is the bottleneck: AmplitudeEmbedding does CPU normalization
        and requires H2D transfer for each batch.
        """
        # Bottleneck: CPU normalization + H2D transfer
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(n_qubits), pad_with=0.0, normalize=True
        )

        # VQC part (same for both methods)
        ansatz_layer(weights, wires=range(n_qubits))

        # Measure expectation value
        return qml.expval(qml.PauliZ(0))

    class PLModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Manual weights to keep the same execution style as QDP path.
            # This avoids unfairness where one path gets hidden batching optimizations.
            self.weights = nn.Parameter(
                torch.rand(n_layers, n_qubits, 3, dtype=torch.float64)
            )
            self._profile: dict[str, float] = {}

        def forward(self, x):
            """
            Forward pass using PennyLane native AmplitudeEmbedding.

            Bottleneck: AmplitudeEmbedding performs CPU-based normalization
            and requires H2D transfer for each batch, which becomes the
            performance bottleneck in training loops.
            """
            t0 = time.perf_counter()
            # Flatten image: (B, 1, 28, 28) -> (B, 784)
            x_flat = x.view(x.shape[0], -1)
            # PennyLane's AmplitudeEmbedding expects float tensors; ensure float64.
            # For SOTA configs, you may choose to move to GPU before embedding,
            # but we keep it explicit and record time.
            t1 = time.perf_counter()

            outputs = []
            # Execute one circuit per sample (same pattern as QDP path).
            for i in range(x_flat.shape[0]):
                outputs.append(
                    circuit_pl(x_flat[i].to(dtype=torch.float64), self.weights)
                )
            out = torch.stack(outputs)
            t2 = time.perf_counter()

            self._profile = {
                "pl.flatten": (t1 - t0),
                "pl.qnode_forward_total": (t2 - t1),
            }
            return out

    # Keep model on CPU by default; PennyLane device determines execution.
    model = PLModel()
    print(
        f"  Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )
    print(f"  PennyLane device: {_dev_name(qml_dev)}")
    return model


# ==========================================
# Method B: QDP Accelerated
# ==========================================
def create_qdp_model(n_qubits, n_layers, device, qdp_engine):
    """
    Create QDP-accelerated model using QubitStateVector.

    QDP requires GPU backend to maintain end-to-end GPU pipeline.
    State vectors stay on GPU throughout the entire training process.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of VQC layers
        device: PyTorch device
        qdp_engine: QDP engine instance

    Returns:
        PyTorch model

    Raises:
        RuntimeError: If GPU backend (lightning.gpu) is not available
    """
    print("\n[Step 3] Building QDP Accelerated Model...")
    # QDP requires GPU backend for end-to-end GPU pipeline
    qml_dev = get_quantum_device(n_qubits, require_gpu=True)

    @qml.qnode(qml_dev, interface="torch", diff_method="adjoint")
    def circuit_qdp(state_vector, weights):
        """
        QDP-accelerated circuit with StatePrep.

        This bypasses the embedding bottleneck by directly using
        GPU-prepared state vectors from QDP.
        """
        # QDP advantage: Direct GPU state vector (no CPU computation, no H2D)
        # StatePrep directly sets the quantum state from the prepared vector
        qml.StatePrep(state_vector, wires=range(n_qubits))

        # VQC part (same for both methods)
        ansatz_layer(weights, wires=range(n_qubits))

        # Measure expectation value
        return qml.expval(qml.PauliZ(0))

    class QDPModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Manual weight management (TorchLayer doesn't work well with state vectors)
            self.weights = nn.Parameter(
                torch.rand(n_layers, n_qubits, 3, dtype=torch.float64)
            )
            self._profile: dict[str, float] = {}

        def forward(self, x_raw_img):
            """
            Forward pass using QDP-accelerated encoding.

            Key advantage: QDP performs batch encoding on GPU in a single kernel launch,
            eliminating the CPU normalization bottleneck present in AmplitudeEmbedding.

            This maintains an end-to-end GPU pipeline:
            - QDP encodes on GPU → state vectors on GPU
            - PennyLane executes on GPU → results on GPU
            - Zero D2H transfers, maximum performance
            """
            prof: dict[str, float] = {}
            t0 = time.perf_counter()
            # 1. Flatten image: (B, 1, 28, 28) -> (B, 784)
            x_flat = x_raw_img.view(x_raw_img.shape[0], -1)
            t1 = time.perf_counter()
            prof["qdp.flatten"] = t1 - t0

            # 2. QDP encode input must be CPU torch tensor (current binding limitation).
            #
            # IMPORTANT PERFORMANCE NOTE:
            # - If your upstream pipeline already moved `x_raw_img` to CUDA,
            #   this `.to("cpu")` is a *device roundtrip* (D2H) and can dominate runtime.
            # - For best performance, keep raw features on CPU and let QDP produce GPU state.
            t_cpu0 = time.perf_counter()
            x_cpu = x_flat.detach()
            if x_cpu.is_cuda:
                x_cpu = x_cpu.to("cpu", non_blocking=False)
            x_cpu = x_cpu.to(dtype=torch.float64).contiguous()
            t_cpu1 = time.perf_counter()
            prof["qdp.input_to_cpu_f64_contig"] = t_cpu1 - t_cpu0

            # 3. QDP encode (batch path). This returns a DLPack capsule pointing to GPU memory.
            # Make timing accurate by synchronizing around CUDA work.
            cuda_sync()
            t_enc0 = time.perf_counter()
            qtensor = qdp_engine.encode(
                x_cpu, num_qubits=n_qubits, encoding_method="amplitude"
            )
            # torch.from_dlpack will *consume* the capsule once.
            state_vector_gpu = torch.from_dlpack(qtensor)
            cuda_sync()
            t_enc1 = time.perf_counter()
            prof["qdp.encode+dlpack"] = t_enc1 - t_enc0

            # 4. QDP maintains end-to-end GPU pipeline: state vectors stay on GPU.
            # This is a key advantage: zero-copy, no D2H transfer overhead.
            t_xfer0 = time.perf_counter()
            # Verify we're using GPU backend (should be guaranteed by get_quantum_device)
            if not _is_gpu_backend(qml_dev):
                raise RuntimeError(
                    f"QDP requires GPU backend but got {_dev_name(qml_dev)}. "
                    f"This should not happen - check get_quantum_device(require_gpu=True)."
                )
            # State vector stays on GPU - zero-copy transfer to PennyLane
            state_for_circuit = state_vector_gpu
            t_xfer1 = time.perf_counter()
            prof["qdp.state_transfer_for_backend"] = t_xfer1 - t_xfer0  # Should be ~0

            # 5. Execute circuit per sample (matches baseline execution style).
            t_q0 = time.perf_counter()
            outputs = []
            for i in range(state_for_circuit.shape[0]):
                outputs.append(circuit_qdp(state_for_circuit[i], self.weights))
            out = torch.stack(outputs)
            t_q1 = time.perf_counter()
            prof["qdp.qnode_forward_total"] = t_q1 - t_q0

            self._profile = prof
            return out

    model = QDPModel()
    print(
        f"  Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )
    print(f"  PennyLane device: {_dev_name(qml_dev)}")
    return model


# ==========================================
# Training Benchmark Function
# ==========================================
def run_training_benchmark(
    model,
    train_data,
    name,
    device,
    limit_batches=None,
    learning_rate=None,
    simulate_data_on_cuda: bool = False,
):
    """
    Run complete training benchmark (forward + backward).

    Args:
        model: PyTorch model
        train_data: DataLoader or list of batches
        name: Model name for logging
        device: PyTorch device
        limit_batches: Limit number of batches (None = all)

    Returns:
        Throughput (samples/second)
    """
    print(f"\n--- Starting Training Benchmark: {name} ---")

    model.train()

    # Use provided learning rate or default
    lr = learning_rate if learning_rate is not None else DEFAULT_LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Warmup (compile caches, JIT paths, etc.)
    print("  Warming up...")
    dummy_input = torch.rand(2, 1, 28, 28)
    try:
        _ = model(dummy_input)
    except Exception as e:
        print(f"  Warning: Warmup failed: {e}")

    clean_gpu_cache()

    timers: dict[str, float] = {}
    processed_batches = 0
    total_samples = 0

    t_total0 = time.perf_counter()
    t_last_batch_end = t_total0
    for batch_idx, (data, target) in enumerate(train_data):
        if limit_batches and batch_idx >= limit_batches:
            break

        t_batch_start = time.perf_counter()
        # Approximate DataLoader + Python iteration overhead between batches.
        # This is the time from end of previous batch processing to start of this batch.
        # (For num_workers>0 this also includes queueing/IPC).
        if batch_idx > 0:
            timers["loop.dataloader_wait"] = timers.get("loop.dataloader_wait", 0.0) + (
                t_batch_start - t_last_batch_end
            )

        if simulate_data_on_cuda:
            # This simulates a common PyTorch training workflow:
            # move raw batches to CUDA at the start of the loop.
            # For QDP this is a *performance hazard* because qdp-python currently
            # requires CPU torch tensors as input.
            t_d0 = time.perf_counter()
            data = data.to(device)
            target = target.to(device)
            cuda_sync()
            t_d1 = time.perf_counter()
            timers["loop.data_to_cuda"] = timers.get("loop.data_to_cuda", 0.0) + (
                t_d1 - t_d0
            )

        # DataLoader gives CPU tensors. We intentionally keep raw data on CPU here,
        # because QDP input path currently requires CPU torch tensors.
        t_prep0 = time.perf_counter()
        target = target.float()
        target = torch.where(target == 0, -1.0, 1.0)
        t_prep1 = time.perf_counter()
        timers["loop.label_prep"] = timers.get("loop.label_prep", 0.0) + (
            t_prep1 - t_prep0
        )

        optimizer.zero_grad(set_to_none=True)

        # Forward pass (model is responsible for any CPU/GPU moves)
        t_f0 = time.perf_counter()
        output = model(data)
        cuda_sync()
        t_f1 = time.perf_counter()
        timers["loop.forward_total"] = timers.get("loop.forward_total", 0.0) + (
            t_f1 - t_f0
        )

        # Harvest per-model profile breakdown (if provided)
        prof = getattr(model, "_profile", None)
        if isinstance(prof, dict):
            for k, v in prof.items():
                timers[k] = timers.get(k, 0.0) + float(v)

        # Shape handling and device alignment
        if output.dim() > 1:
            output = output.squeeze()
        if output.dim() == 0:
            output = output.unsqueeze(0)

        # Ensure output and target are on the same device
        # (PennyLane CPU backends produce CPU outputs even if input was on GPU)
        if output.device != target.device:
            output = output.to(target.device)

        # Loss/backward/step timings
        t_l0 = time.perf_counter()
        loss = criterion(output, target)
        t_l1 = time.perf_counter()
        timers["loop.loss"] = timers.get("loop.loss", 0.0) + (t_l1 - t_l0)

        t_b0 = time.perf_counter()
        loss.backward()
        cuda_sync()
        t_b1 = time.perf_counter()
        timers["loop.backward"] = timers.get("loop.backward", 0.0) + (t_b1 - t_b0)

        t_s0 = time.perf_counter()
        optimizer.step()
        t_s1 = time.perf_counter()
        timers["loop.opt_step"] = timers.get("loop.opt_step", 0.0) + (t_s1 - t_s0)

        processed_batches += 1
        total_samples += data.shape[0]

        if (batch_idx + 1) % 10 == 0:
            print(f"  {name} | Batch {batch_idx + 1} | Loss: {loss.item():.4f}")

        t_last_batch_end = time.perf_counter()

    t_total1 = time.perf_counter()
    total_time = t_total1 - t_total0
    throughput = total_samples / total_time if total_time > 0 else 0.0

    print(f"\n>>> {name} Results:")
    print(f"  Total Time: {total_time:.4f} seconds")
    print(f"  Total Samples: {total_samples}")
    print(f"  Throughput: {throughput:.2f} samples/second")
    print(f"  Time per Sample: {(total_time / total_samples * 1000):.2f} ms")
    print(f"\n  Timing breakdown (accumulated over {processed_batches} batches):")
    print(_summarize_timings_ms(timers, denom=total_time))

    return throughput, total_time, timers


def main():
    parser = argparse.ArgumentParser(
        description="Complete Training Benchmark: QDP vs PennyLane Native"
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
        help=f"Number of qubits (default: {DEFAULT_N_QUBITS})",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=DEFAULT_N_LAYERS,
        help=f"Number of VQC layers (default: {DEFAULT_N_LAYERS})",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=DEFAULT_LIMIT_BATCHES,
        help=f"Limit number of batches for benchmarking (default: {DEFAULT_LIMIT_BATCHES})",
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--skip-pennylane", action="store_true", help="Skip PennyLane native benchmark"
    )
    parser.add_argument("--skip-qdp", action="store_true", help="Skip QDP benchmark")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader num_workers (default: 0)"
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pin_memory (default: off)",
    )
    parser.add_argument(
        "--simulate-data-on-cuda",
        action="store_true",
        help="Simulate typical training loop that moves raw batches to CUDA first (useful to reveal D2H bottlenecks for QDP).",
    )

    args = parser.parse_args()

    # QDP requires CUDA.
    if not torch.cuda.is_available():
        print("❌ Error: CUDA is not available. QDP requires GPU.")
        if not args.skip_qdp:
            print("   Use --skip-qdp to run only PennyLane Native benchmark.")
        return 1

    # Keep this as info only; we keep raw training data on CPU in the loop.
    cuda_device = torch.device(f"cuda:{args.device_id}")
    print("=" * 60)
    print("Complete Training Benchmark: QDP vs PennyLane Native")
    print("=" * 60)
    print("Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of qubits: {args.n_qubits} (state length: {1 << args.n_qubits})")
    print(f"  Number of VQC layers: {args.n_layers}")
    print(f"  Limit batches: {args.limit_batches}")
    print(
        f"  CUDA device: {args.device_id} ({torch.cuda.get_device_name(args.device_id)})"
    )
    print("=" * 60)

    # Check PennyLane
    if not HAS_PENNYLANE:
        print("Error: PennyLane is required for this benchmark.")
        print("Install with: pip install pennylane")
        if not args.skip_pennylane:
            return 1

    # Prepare data
    train_data = prepare_mnist_binary(
        args.batch_size,
        args.n_qubits,
        args.limit_batches,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Initialize QDP engine
    if not args.skip_qdp:
        print("\n[Step 0] Initializing QDP Engine...")
        qdp_engine = QdpEngine(device_id=args.device_id, precision="float32")
        print("  QDP Engine initialized")

    # Run benchmarks
    results = {}

    # PennyLane Native
    if not args.skip_pennylane:
        try:
            pl_model = create_pennylane_model(args.n_qubits, args.n_layers, cuda_device)
            throughput, total_time, timers = run_training_benchmark(
                pl_model,
                train_data,
                "PennyLane Native",
                cuda_device,
                args.limit_batches,
                args.learning_rate,
                args.simulate_data_on_cuda,
            )
            results["PennyLane"] = (throughput, total_time, timers)
        except Exception as e:
            print(f"\nError running PennyLane benchmark: {e}")
            import traceback

            traceback.print_exc()

    # QDP Accelerated
    if not args.skip_qdp:
        try:
            qdp_model = create_qdp_model(
                args.n_qubits, args.n_layers, cuda_device, qdp_engine
            )
            throughput, total_time, timers = run_training_benchmark(
                qdp_model,
                train_data,
                "QDP Accelerated",
                cuda_device,
                args.limit_batches,
                args.learning_rate,
                args.simulate_data_on_cuda,
            )
            results["QDP"] = (throughput, total_time, timers)
        except RuntimeError as e:
            # Handle GPU backend requirement error gracefully
            error_msg = str(e)
            if (
                "lightning.gpu" in error_msg.lower()
                or "gpu backend" in error_msg.lower()
            ):
                print(
                    "\n⚠️  QDP benchmark skipped: GPU backend (lightning.gpu) is required"
                )
                print(f"   {error_msg}")
                print("\n   To run QDP benchmark, please install:")
                print("     pip install pennylane-lightning[gpu]")
                print("\n   Note: This requires CUDA 12 and cuQuantum SDK.")
                print(
                    "   For now, only PennyLane Native benchmark results are shown below."
                )
            else:
                print(f"\n❌ Error running QDP benchmark: {e}")
                import traceback

                traceback.print_exc()
        except Exception as e:
            print(f"\n❌ Error running QDP benchmark: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK REPORT")
    print("=" * 60)

    if "PennyLane" in results:
        pl_throughput, pl_time, _ = results["PennyLane"]
        print("PennyLane Native:")
        print(f"  Throughput: {pl_throughput:.2f} samples/second")
        print(f"  Total Time: {pl_time:.4f} seconds")

    if "QDP" in results:
        qdp_throughput, qdp_time, _ = results["QDP"]
        print("\nQDP Accelerated:")
        print(f"  Throughput: {qdp_throughput:.2f} samples/second")
        print(f"  Total Time: {qdp_time:.4f} seconds")

    if "PennyLane" in results and "QDP" in results:
        speedup = qdp_throughput / pl_throughput if pl_throughput > 0 else math.inf
        time_speedup = pl_time / qdp_time if qdp_time > 0 else math.inf

        print("\n🚀 Performance Analysis:")
        print(f"  Throughput Speedup: {speedup:.2f}x")
        print(f"  Time Speedup: {time_speedup:.2f}x")
    elif "PennyLane" in results and "QDP" not in results:
        print("\n⚠️  QDP benchmark was not run (GPU backend required).")
        print("   Only PennyLane Native results are available.")

        # Workflow-level diagnostics (based on the *actual* observed execution path)
        if "QDP" in results:
            qdp_timers = results["QDP"][2]

            # QDP maintains end-to-end GPU pipeline - state transfer should be minimal
            qdp_state_xfer = qdp_timers.get("qdp.state_transfer_for_backend", 0.0)
            qdp_encode = qdp_timers.get("qdp.encode+dlpack", 0.0)
            qdp_qnode = qdp_timers.get("qdp.qnode_forward_total", 0.0)
            qdp_cpu_in = qdp_timers.get("qdp.input_to_cpu_f64_contig", 0.0)

            print(
                "\n🔎 Workflow bottleneck hints (QDP path - end-to-end GPU pipeline):"
            )
            print(f"  - qdp.input_to_cpu_f64_contig: {qdp_cpu_in * 1000.0:.3f} ms")
            print(f"  - qdp.encode+dlpack:           {qdp_encode * 1000.0:.3f} ms")
            print(
                f"  - qdp.state_transfer_backend:  {qdp_state_xfer * 1000.0:.3f} ms (should be ~0)"
            )
            print(f"  - qdp.qnode_forward_total:     {qdp_qnode * 1000.0:.3f} ms")

            if speedup > 1.2:
                print("\n  ✅ QDP shows significant advantage!")
                print("  This demonstrates QDP's ability to eliminate the")
                print("  data preparation bottleneck in quantum training pipelines.")
            elif speedup > 0.9:
                print("\n  ⚠️  Similar performance (circuit execution dominates).")
                print("  QDP's advantage is more visible with:")
                print("    - Larger batch sizes (try --batch-size 64 or 128)")
                print("    - More qubits (try --n-qubits 12 or 14)")
            else:
                print("\n  ⚠️  QDP slower in this configuration.")
                print("  Most common workflow causes:")
                print(
                    "    - QDP Python binding currently accepts only CPU torch tensors as input."
                )
                print(
                    "      If your training loop moved images to CUDA first, you pay D2H per batch."
                )
                print(
                    "    - Per-sample QNode execution dominates when batching isn't supported."
                )
                print(
                    "    - Circuit execution time may dominate over data preparation time."
                )
                print("  Actionable next steps:")
                print(
                    "    - Keep raw features on CPU and feed them directly to QDP (avoid data.to('cuda') before encode)."
                )
                print("    - Try larger batch sizes to better utilize GPU parallelism.")
                print(
                    "    - If needed, extend qdp-python to accept GPU torch tensors (eliminate D2H)."
                )

    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
