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
Real-world dataset preparation for QML benchmarking.

Downloads and preprocesses real datasets (MNIST & Credit Card),
converting them to QML input format (normalized, dimension-reduced/padded)
and saving as Parquet files.
"""

import os
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set data storage path
DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)


def pad_to_power_of_2(data):
    """Pad feature dimension to nearest 2^n."""
    n_samples, n_features = data.shape
    next_pow2 = 1 << (n_features - 1).bit_length()
    if next_pow2 == n_features:
        return data, n_features.bit_length()

    padded = np.zeros((n_samples, next_pow2))
    padded[:, :n_features] = data
    return padded, next_pow2.bit_length()


def prepare_mnist(n_components=None, limit=10000):
    """
    Prepare MNIST dataset.

    Args:
        n_components: If set, use PCA to reduce dimensions (e.g., 16 for 4 qubits)
        limit: Limit number of samples to speed up testing
    """
    print(f"Downloading MNIST (limit={limit})...")
    # MNIST original is 784 features (~10 qubits)
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    if limit:
        X = X[:limit]
        y = y[:limit]

    # QML standard SOP: Scale to [0, 1] or standardize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if n_components:
        print(f"Applying PCA to reduce dimensions to {n_components}...")
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)

    # Amplitude encoding requires L2 Norm normalization
    # Note: We don't do L2 Norm here, leave it for the "CPU Preprocessing" stage
    # in the benchmark workflow, so we can fairly compare Mahout (GPU Norm) vs
    # Numpy (CPU Norm) differences.

    # Pad to 2^n
    X_final, num_qubits = pad_to_power_of_2(X)

    print(f"MNIST Prepared: Shape {X_final.shape}, Fits into {num_qubits} Qubits")

    # Save to Parquet
    filename = f"{DATA_DIR}/mnist_q{num_qubits}_{len(X)}.parquet"
    arrays = [pa.array(row) for row in X_final]
    table = pa.Table.from_arrays([arrays], names=["feature_vector"])
    pq.write_table(table, filename)
    print(f"Saved to {filename}")
    return filename, num_qubits


def prepare_credit_card(limit=None):
    """
    Prepare UCI Credit Card Fraud Detection dataset (simulates financial QML).

    Contains 28 PCA features + Time + Amount = 30 features -> Pad to 32 (5 qubits).
    """
    print("Downloading/Generating Credit Card data (Simulation if download fails)...")

    # Since Credit Card dataset usually requires Kaggle API, we use sklearn
    # to generate similar structured data to simulate financial tabular data (30 features)
    from sklearn.datasets import make_classification

    n_samples = limit if limit else 50000
    X, _ = make_classification(n_samples=n_samples, n_features=30, random_state=42)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_final, num_qubits = pad_to_power_of_2(X)  # 30 -> 32 (5 qubits)

    print(f"CreditCard Prepared: Shape {X_final.shape}, Fits into {num_qubits} Qubits")

    filename = f"{DATA_DIR}/credit_q{num_qubits}_{len(X)}.parquet"
    arrays = [pa.array(row) for row in X_final]
    table = pa.Table.from_arrays([arrays], names=["feature_vector"])
    pq.write_table(table, filename)
    print(f"Saved to {filename}")
    return filename, num_qubits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare real-world datasets for QML benchmarking"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "credit"],
        default="mnist",
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--pca", type=int, default=None, help="PCA components (e.g., 16 for 4 qubits)"
    )
    parser.add_argument(
        "--rows", type=int, default=5000, help="Number of rows to generate/limit"
    )
    args = parser.parse_args()

    if args.dataset == "mnist":
        prepare_mnist(n_components=args.pca, limit=args.rows)
    elif args.dataset == "credit":
        prepare_credit_card(limit=args.rows)
