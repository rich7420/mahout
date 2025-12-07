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
Dataset generation utility for stress testing QML pipelines.

Generates Parquet files with high-dimensional feature vectors (up to 2^22 features)
to test IO bottlenecks and encoding throughput.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import argparse


def generate_parquet(num_qubits, num_rows, output_file):
    """
    Generates a Parquet file with 'num_rows' vectors, each having 2^num_qubits elements.

    This simulates a pre-processed QML dataset (e.g., PCA reduced images or time-series).
    """
    vector_dim = 1 << num_qubits
    print(f"Generating dataset for {num_qubits} qubits (dim={vector_dim})...")
    print(
        f"Rows: {num_rows}, Est. Raw Size: {num_rows * vector_dim * 8 / (1024**3):.2f} GB"
    )

    # We generate in chunks to avoid blowing up RAM before writing to disk
    chunk_size = 100

    schema = pa.schema([("feature_vector", pa.list_(pa.float64()))])

    with pq.ParquetWriter(output_file, schema) as writer:
        for start_idx in range(0, num_rows, chunk_size):
            current_batch = min(chunk_size, num_rows - start_idx)
            # Generate random data
            data_chunk = np.random.rand(current_batch, vector_dim).astype(np.float64)

            # Normalize (simulate L2 norm requirement for amplitude encoding)
            norms = np.linalg.norm(data_chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            data_chunk /= norms

            # Convert to PyArrow format
            # Store as list of arrays (each row is a list of floats)
            feature_vectors = [row.tolist() for row in data_chunk]
            arrays = pa.array(feature_vectors, type=pa.list_(pa.float64()))
            batch_table = pa.Table.from_arrays([arrays], names=["feature_vector"])

            writer.write_table(batch_table)
            print(
                f"  Wrote batch of {current_batch} rows (total: {start_idx + current_batch}/{num_rows})..."
            )

    print(f"Successfully created {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic Parquet datasets for QML benchmarking"
    )
    parser.add_argument(
        "--qubits", type=int, default=18, help="Number of qubits (vector size 2^n)"
    )
    parser.add_argument("--rows", type=int, default=1000, help="Number of samples")
    parser.add_argument(
        "--out", type=str, default="benchmarks_data.parquet", help="Output file path"
    )
    args = parser.parse_args()

    generate_parquet(args.qubits, args.rows, args.out)
