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
Behavior tests for encode_stream (§5.11.11).

Tests: empty iterator, single/multi-batch order, error propagation.
Uses a mock engine (no CUDA) so tests run without GPU.
"""

from __future__ import annotations

import unittest

import numpy as np

from qumat_qdp import encode_stream


class MockEngine:
    """Mock engine that returns (call_index,) so we can assert order."""

    def __init__(self, raise_on_call: int | None = None):
        self.call_count = 0
        self.raise_on_call = raise_on_call  # 1-based: raise on Nth call

    def encode(self, batch: np.ndarray, num_qubits: int, encoding_method: str):
        self.call_count += 1
        if self.raise_on_call is not None and self.call_count == self.raise_on_call:
            raise ValueError("mock encode failure")
        return (self.call_count - 1,)


def _batches(n: int, size: int = 2, dim: int = 4) -> list[np.ndarray]:
    return [np.zeros((size, dim), dtype=np.float64) for _ in range(n)]


class TestEncodeStream(unittest.TestCase):
    def test_empty_iterator(self):
        """Empty iterator: encode_stream yields 0 times and exits (§5.11.1 #9)."""
        engine = MockEngine()
        out = list(encode_stream(engine, iter([]), 2, "amplitude", in_flight=2))
        self.assertEqual(out, [])
        self.assertEqual(engine.call_count, 0)

    def test_single_batch_order(self):
        """Single batch: one yield in order."""
        engine = MockEngine()
        batches = _batches(1)
        out = list(encode_stream(engine, iter(batches), 2, "amplitude", in_flight=2))
        self.assertEqual(out, [(0,)])
        self.assertEqual(engine.call_count, 1)

    def test_multi_batch_order(self):
        """Multiple batches: results match input order (FIFO) (§5.11.1 #2)."""
        engine = MockEngine()
        batches = _batches(5)
        out = list(encode_stream(engine, iter(batches), 2, "amplitude", in_flight=2))
        self.assertEqual(out, [(0,), (1,), (2,), (3,), (4,)])
        self.assertEqual(engine.call_count, 5)

    def test_error_propagation(self):
        """Worker encode() error is re-raised when yielding that batch (§5.11.1 #4, #9)."""
        engine = MockEngine(raise_on_call=2)  # fail on 2nd batch
        batches = _batches(3)
        gen = encode_stream(engine, iter(batches), 2, "amplitude", in_flight=2)
        first = next(gen)
        self.assertEqual(first, (0,))
        with self.assertRaises(ValueError) as ctx:
            next(gen)
        self.assertIn("mock encode failure", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
