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

"""Tests for TensorFlow tensor input support in QDP Python bindings."""

import pytest


@pytest.mark.gpu
def test_encode_tensorflow_tensor_cpu_1d():
    """Encode from TensorFlow eager 1D tensor (single sample)."""
    tf = pytest.importorskip("tensorflow")
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    from _qdp import QdpEngine

    engine = QdpEngine(0)
    x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float64)

    qtensor = engine.encode(x, 2, "amplitude")
    t = torch.from_dlpack(qtensor)

    assert t.is_cuda
    assert t.shape == (1, 4)


@pytest.mark.gpu
def test_encode_tensorflow_tensor_cpu_2d_batch():
    """Encode from TensorFlow eager 2D tensor (batch)."""
    tf = pytest.importorskip("tensorflow")
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    from _qdp import QdpEngine

    engine = QdpEngine(0)
    x = tf.constant(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=tf.float64,
    )

    qtensor = engine.encode(x, 2, "amplitude")
    t = torch.from_dlpack(qtensor)

    assert t.is_cuda
    assert t.shape == (3, 4)


@pytest.mark.gpu
def test_encode_tensorflow_tensor_dtype_error():
    """TensorFlow tensors must be float64 for current bindings."""
    tf = pytest.importorskip("tensorflow")
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    from _qdp import QdpEngine

    engine = QdpEngine(0)
    x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)

    with pytest.raises(RuntimeError, match="dtype must be float64"):
        engine.encode(x, 2, "amplitude")


@pytest.mark.gpu
def test_encode_tensorflow_tensor_rank_error():
    """Only 1D/2D TensorFlow tensors are supported."""
    tf = pytest.importorskip("tensorflow")
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("GPU required for QdpEngine")

    from _qdp import QdpEngine

    engine = QdpEngine(0)
    x = tf.constant(1.0, dtype=tf.float64)  # scalar, rank 0

    with pytest.raises(RuntimeError, match="Unsupported TensorFlow tensor shape"):
        engine.encode(x, 2, "amplitude")
