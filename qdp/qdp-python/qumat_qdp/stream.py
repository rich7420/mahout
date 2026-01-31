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
Optimized pipeline API: encode_stream (batch-level overlap).

This module is part of the qumat_qdp package. When the engine exposes
encode_batch_submit (Linux, Rust pool), encode_stream submits batches to the
Rust pool and get()s in order so the GPU stays fed. Otherwise a single worker
thread is used.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np


def _normalize_batch(
    batch: np.ndarray, encoding_method: str = "amplitude"
) -> np.ndarray:
    """L2-normalize batch for amplitude encoding; no-op for angle/basis."""
    if encoding_method in ("basis", "angle"):
        return batch
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return batch / norms


def encode_stream(
    engine,
    batch_iterator: Iterator[np.ndarray],
    num_qubits: int,
    encoding_method: str,
    *,
    in_flight: int = 4,
):
    """
    Yield encoded QuantumTensor for each batch in order (async pipeline).

    When the engine has encode_batch_submit (Linux): submits batches to the
    Rust-side pool (submit many, get in order) so the GPU has multiple batches
    in flight. Otherwise uses a single worker thread + queue.

    Args:
        engine: QdpEngine instance (must have .encode(batch, num_qubits, encoding_method),
            and optionally .encode_batch_submit for pool path).
        batch_iterator: Iterator of 2D NumPy arrays (batch_size, sample_size).
        num_qubits: Number of qubits.
        encoding_method: "amplitude", "angle", or "basis".
        in_flight: Max number of batches in the pipeline (default 4); backpressure.

    Yields:
        QuantumTensor (DLPack) for each batch, in order.

    Raises:
        Any exception from encode / encode_batch_submit is re-raised when yielding.

    Notes:
        - Empty iterator: yields nothing and exits cleanly.
        - Generator supports single use only; do not iterate twice.
    """
    submit_fn = getattr(engine, "encode_batch_submit", None)
    if submit_fn is not None:
        yield from _encode_stream_pool(
            engine, submit_fn, batch_iterator, num_qubits, encoding_method, in_flight
        )
        return

    input_queue: queue.Queue[tuple[int | None, np.ndarray | None]] = queue.Queue(
        maxsize=in_flight
    )
    output_queue: queue.Queue[tuple] = queue.Queue()
    buffer: dict[int, object] = {}
    next_yield = 0
    batch_count = 0

    def worker() -> None:
        while True:
            item = input_queue.get()
            i, batch = item
            if i is None:
                break
            try:
                qt = engine.encode(batch, num_qubits, encoding_method)
                output_queue.put((i, qt))
            except Exception as e:
                output_queue.put((i, None, e))

    thread = threading.Thread(target=worker, daemon=False)
    thread.start()
    sentinel_put = False
    try:
        for i, batch in enumerate(batch_iterator):
            batch_count = i + 1
            normalized = np.ascontiguousarray(
                _normalize_batch(batch, encoding_method), dtype=np.float64
            )
            input_queue.put((i, normalized))
            while next_yield not in buffer:
                item = output_queue.get()
                if len(item) == 3 and item[1] is None:
                    buffer[item[0]] = (None, item[2])
                else:
                    buffer[item[0]] = item[1]
            while next_yield in buffer:
                val = buffer.pop(next_yield)
                next_yield += 1
                if isinstance(val, tuple) and val[0] is None:
                    raise val[1]
                yield val
        sentinel_put = True
        input_queue.put((None, None))
        while next_yield < batch_count:
            while next_yield not in buffer:
                item = output_queue.get()
                if len(item) == 3 and item[1] is None:
                    buffer[item[0]] = (None, item[2])
                else:
                    buffer[item[0]] = item[1]
            while next_yield in buffer:
                val = buffer.pop(next_yield)
                next_yield += 1
                if isinstance(val, tuple) and val[0] is None:
                    raise val[1]
                yield val
    finally:
        if not sentinel_put:
            try:
                input_queue.put((None, None))
            except Exception:
                pass
        thread.join()


def _encode_stream_pool(
    engine,
    submit_fn,
    batch_iterator: Iterator[np.ndarray],
    num_qubits: int,
    encoding_method: str,
    in_flight: int,
) -> Iterator[object]:
    """
    Rust pool path: submit batches via encode_batch_submit, get() in order.
    Keeps up to in_flight batches in the pool so the GPU stays fed.
    """
    handles: list[Any] = []
    next_yield = 0

    for batch in batch_iterator:
        normalized = np.ascontiguousarray(
            _normalize_batch(batch, encoding_method), dtype=np.float64
        )
        while len(handles) - next_yield >= in_flight:
            qt = handles[next_yield].get()
            next_yield += 1
            yield qt
        handles.append(submit_fn(normalized, num_qubits, encoding_method))

    for i in range(next_yield, len(handles)):
        yield handles[i].get()
