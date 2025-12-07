//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use qdp_core::QdpEngine as CoreEngine;
use qdp_core::dlpack::DLManagedTensor;
use qdp_core::gpu::encodings::{AmplitudeEncoder, QuantumEncoder};

/// Quantum tensor wrapper implementing DLPack protocol
///
/// This class wraps a GPU-allocated quantum state vector and implements
/// the DLPack protocol for zero-copy integration with PyTorch and other
/// array libraries.
///
/// Example:
///     >>> engine = QdpEngine(device_id=0)
///     >>> qtensor = engine.encode([1.0, 2.0, 3.0], num_qubits=2, encoding_method="amplitude")
///     >>> torch_tensor = torch.from_dlpack(qtensor)
#[pyclass]
struct QuantumTensor {
    ptr: *mut DLManagedTensor,
    consumed: bool,
}

#[pymethods]
impl QuantumTensor {
    /// Implements DLPack protocol - returns PyCapsule for PyTorch
    ///
    /// This method is called by torch.from_dlpack() to get the GPU memory pointer.
    /// The capsule can only be consumed once to prevent double-free errors.
    ///
    /// Args:
    ///     stream: Optional CUDA stream pointer (for DLPack 0.8+)
    ///
    /// Returns:
    ///     PyCapsule containing DLManagedTensor pointer
    ///
    /// Raises:
    ///     RuntimeError: If the tensor has already been consumed
    #[pyo3(signature = (stream=None))]
    fn __dlpack__<'py>(&mut self, py: Python<'py>, stream: Option<i64>) -> PyResult<Py<PyAny>> {
        let _ = stream;  // Suppress unused variable warning
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor already consumed (can only be used once)"
            ));
        }

        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("Invalid DLPack tensor pointer"));
        }

        // Mark as consumed to prevent double-free
        self.consumed = true;

        // Create PyCapsule using FFI
        // PyTorch will call the deleter stored in DLManagedTensor.deleter
        // Use a static C string for the capsule name to avoid lifetime issues
        const DLTENSOR_NAME: &[u8] = b"dltensor\0";

        unsafe {
            // Create PyCapsule without a destructor
            // PyTorch will manually call the deleter from DLManagedTensor
            let capsule_ptr = ffi::PyCapsule_New(
                self.ptr as *mut std::ffi::c_void,
                DLTENSOR_NAME.as_ptr() as *const i8,
                None  // No destructor - PyTorch handles it
            );

            if capsule_ptr.is_null() {
                return Err(PyRuntimeError::new_err("Failed to create PyCapsule"));
            }

            Ok(Py::from_owned_ptr(py, capsule_ptr))
        }
    }

    /// Returns DLPack device information
    ///
    /// Returns:
    ///     Tuple of (device_type, device_id) where device_type=2 for CUDA
    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        // DLDeviceType::kDLCUDA = 2, device_id = 0
        Ok((2, 0))
    }
}

impl Drop for QuantumTensor {
    fn drop(&mut self) {
        // Only free if not consumed by __dlpack__
        // If consumed, PyTorch/consumer will call the deleter
        if !self.consumed && !self.ptr.is_null() {
            unsafe {
                // Defensive check: qdp-core always provides a deleter
                debug_assert!(
                    (*self.ptr).deleter.is_some(),
                    "DLManagedTensor from qdp-core should always have a deleter"
                );

                // Call the DLPack deleter to free memory
                if let Some(deleter) = (*self.ptr).deleter {
                    deleter(self.ptr);
                }
            }
        }
    }
}

// Safety: QuantumTensor can be sent between threads
// The DLManagedTensor pointer management is thread-safe via Arc in the deleter
unsafe impl Send for QuantumTensor {}
unsafe impl Sync for QuantumTensor {}

/// Wrapper Enum to hold either f32 or f64 engine
enum EngineVariant {
    F32(CoreEngine<f32>),
    F64(CoreEngine<f64>),
}

/// PyO3 wrapper for QdpEngine
///
/// Provides Python bindings for GPU-accelerated quantum state encoding.
/// Now supports precision selection (float32 or float64).
#[pyclass]
struct QdpEngine {
    inner: EngineVariant,
}

#[pymethods]
impl QdpEngine {
    /// Initialize QDP engine on specified GPU device
    ///
    /// Args:
    ///     device_id: CUDA device ID (typically 0)
    ///     precision: Floating-point precision ("float32", "f32", "float64", or "f64")
    ///
    /// Returns:
    ///     QdpEngine instance
    ///
    /// Raises:
    ///     RuntimeError: If CUDA device initialization fails or precision is invalid
    ///
    /// Example:
    ///     >>> engine = QdpEngine(0, precision="float32")  # Fast, half memory
    ///     >>> engine = QdpEngine(0, precision="float64")  # High precision
    #[new]
    #[pyo3(signature = (device_id=0, precision="float64"))]
    fn new(device_id: usize, precision: &str) -> PyResult<Self> {
        let variant = match precision {
            "float32" | "f32" => {
                let engine = CoreEngine::<f32>::new(device_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;
                EngineVariant::F32(engine)
            },
            "float64" | "f64" => {
                let engine = CoreEngine::<f64>::new(device_id)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;
                EngineVariant::F64(engine)
            },
            _ => return Err(PyRuntimeError::new_err(
                "Unsupported precision. Use 'float32' or 'float64'"
            )),
        };
        Ok(Self { inner: variant })
    }

    /// Encode classical data into quantum state
    ///
    /// Args:
    ///     data: Input data as list of floats
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude", "angle", or "basis")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack-compatible tensor for zero-copy PyTorch integration
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> qtensor = engine.encode([1.0, 2.0, 3.0, 4.0], num_qubits=2, encoding_method="amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(qtensor)
    ///
    /// TODO: Use numpy array input (`PyReadonlyArray1<f64>`) for zero-copy instead of `Vec<f64>`.
    fn encode(&self, data: Vec<f64>, num_qubits: usize, encoding_method: &str) -> PyResult<QuantumTensor> {
        // For now, only support amplitude encoding with the new generic API
        if encoding_method != "amplitude" {
            return Err(PyRuntimeError::new_err(
                format!("Only 'amplitude' encoding is supported with generic API. Got: {}", encoding_method)
            ));
        }

        let ptr = match &self.inner {
            EngineVariant::F64(engine) => {
                // For now, use the legacy API which has pool integrated
                // TODO: Expose pool access in QdpEngine
                let ptr = engine.encode(&data, num_qubits, encoding_method)
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                ptr
            },
            EngineVariant::F32(engine) => {
                // Convert f64 data to f32
                // Ideally input should be numpy array of f32 to avoid this conversion
                let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                let encoder = AmplitudeEncoder;
                // Use the persistent pool from the engine (reused across calls)
                let state = encoder.encode(engine.device(), engine.pool(), &data_f32, num_qubits)
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                state.to_dlpack()
            }
        };

        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Encode data directly from Parquet file (zero-copy Arrow path)
    ///
    /// This method reads Parquet chunks directly and encodes them without
    /// intermediate Python list conversion, providing maximum performance.
    ///
    /// Args:
    ///     path: Path to Parquet file
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude", "angle", or "basis")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack-compatible tensor for zero-copy PyTorch integration
    ///
    /// Raises:
    ///     RuntimeError: If encoding fails
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> qtensor = engine.encode_from_parquet("data.parquet", num_qubits=10, encoding_method="amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(qtensor)
    fn encode_from_parquet(&self, path: &str, num_qubits: usize, encoding_method: &str) -> PyResult<QuantumTensor> {
        let ptr = match &self.inner {
            EngineVariant::F64(engine) => {
                engine.encode_from_parquet(path, num_qubits, encoding_method)
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding from parquet failed: {}", e)))?
            },
            EngineVariant::F32(engine) => {
                // This now calls the generic implementation which handles Arrow->f32 conversion internally
                engine.encode_from_parquet(path, num_qubits, encoding_method)
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding from parquet failed: {}", e)))?
            }
        };

        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }
}

/// Mahout QDP Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn mahout_qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QdpEngine>()?;
    m.add_class::<QuantumTensor>()?;
    Ok(())
}
