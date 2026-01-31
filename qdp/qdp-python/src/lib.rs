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

use std::cell::RefCell;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use qdp_core::dlpack::DLManagedTensor;
#[cfg(target_os = "linux")]
use qdp_core::gpu::BatchHandle as CoreBatchHandle;
#[cfg(target_os = "linux")]
use qdp_core::gpu::EncodeHandle as CoreEncodeHandle;
use qdp_core::{Precision, QdpEngine as CoreEngine};

/// Result of coalesce batcher flush: (batch QuantumTensor, sample boundaries). Reduces type complexity.
type CoalesceResult = (Py<QuantumTensor>, Vec<(usize, usize)>);

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
        let _ = stream; // Suppress unused variable warning
        if self.consumed {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor already consumed (can only be used once)",
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
                None, // No destructor - PyTorch handles it
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
        if self.ptr.is_null() {
            return Err(PyRuntimeError::new_err("Invalid DLPack tensor pointer"));
        }

        unsafe {
            let tensor = &(*self.ptr).dl_tensor;
            // device_type is an enum, convert to integer
            // kDLCUDA = 2, kDLCPU = 1
            // Ref: https://github.com/dmlc/dlpack/blob/6ea9b3eb64c881f614cd4537f95f0e125a35555c/include/dlpack/dlpack.h#L76-L80
            let device_type = match tensor.device.device_type {
                qdp_core::dlpack::DLDeviceType::kDLCUDA => 2,
                qdp_core::dlpack::DLDeviceType::kDLCPU => 1,
            };
            // Read device_id from DLPack tensor metadata
            Ok((device_type, tensor.device.device_id))
        }
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

/// Handle for deferred result of encode_async (§3.9). Call `.get()` to block and receive the tensor.
#[cfg(target_os = "linux")]
#[pyclass]
struct EncodeHandle {
    inner: Option<CoreEncodeHandle>,
}

#[cfg(target_os = "linux")]
#[pymethods]
impl EncodeHandle {
    /// Block until the result is ready; returns the encoded QuantumTensor.
    /// Preserves order: result corresponds to the request submitted with encode_async.
    fn get(&mut self) -> PyResult<QuantumTensor> {
        let inner = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("EncodeHandle already consumed (get() can only be called once)")
        })?;
        let ptr = inner
            .get()
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }
}

#[cfg(target_os = "linux")]
unsafe impl Send for EncodeHandle {}
#[cfg(target_os = "linux")]
unsafe impl Sync for EncodeHandle {}

/// Handle for non-blocking batch submit (Rust pool). Call `.get()` to block and receive QuantumTensor.
#[cfg(target_os = "linux")]
#[pyclass]
struct BatchHandle {
    inner: Option<CoreBatchHandle>,
}

#[cfg(target_os = "linux")]
#[pymethods]
impl BatchHandle {
    /// Block until the batch result is ready; returns the encoded QuantumTensor.
    fn get(&mut self) -> PyResult<QuantumTensor> {
        let inner = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("BatchHandle already consumed (get() can only be called once)")
        })?;
        let ptr = inner
            .get()
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }
}

#[cfg(target_os = "linux")]
unsafe impl Send for BatchHandle {}
#[cfg(target_os = "linux")]
unsafe impl Sync for BatchHandle {}

/// Scope-based coalescing: collect submits in `with` block, flush on exit (§3.4, §3.6).
///
/// Use `with engine.coalesce(sample_size, num_qubits, encoding_method) as batcher:`
/// then `batcher.submit(array)` / `batcher.submit_batch(...)`; on exit, flush runs
/// (one copy to Pinned → run_dual_stream_pipeline) and result is in `batcher.get_result()`.
#[pyclass(unsendable)]
struct CoalesceBatcher {
    engine: Py<PyAny>,
    sample_size: usize,
    num_qubits: usize,
    encoding_method: String,
    arrays: RefCell<Vec<Py<PyAny>>>,
    result: RefCell<Option<CoalesceResult>>,
}

#[pymethods]
impl CoalesceBatcher {
    /// Context manager entry; returns self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// On normal exit: flush (encode_list) and store (QuantumTensor, boundaries) in result.
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let arrays = self.arrays.borrow();
        if arrays.is_empty() {
            return Err(PyRuntimeError::new_err(
                "coalesce: flush called with empty queue (no submit before exit)",
            ));
        }
        let list = PyList::empty(py);
        for arr in arrays.iter() {
            list.append(arr.bind(py))?;
        }
        drop(arrays);
        let engine = self
            .engine
            .bind(py)
            .cast::<QdpEngine>()
            .map_err(|_| PyRuntimeError::new_err("coalesce: engine type mismatch"))?;
        let ret = engine.call_method1(
            "encode_list",
            (
                list,
                self.sample_size,
                self.num_qubits,
                self.encoding_method.as_str(),
            ),
        )?;
        let item0 = ret.get_item(0)?;
        let bound_batch = item0.cast::<QuantumTensor>()?;
        let batch_py = bound_batch.clone().unbind();
        let boundaries = ret.get_item(1)?.extract::<Vec<(usize, usize)>>()?;
        *self.result.borrow_mut() = Some((batch_py, boundaries.clone()));
        self.arrays.borrow_mut().clear();
        Ok(false)
    }

    /// Submit one sample (1D float64 array of length sample_size).
    fn submit(&self, _py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<()> {
        let arr = array.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
            PyRuntimeError::new_err("coalesce.submit: argument must be 1D NumPy array (float64)")
        })?;
        if arr.len() != self.sample_size {
            return Err(PyRuntimeError::new_err(format!(
                "coalesce.submit: array length {} != sample_size {}",
                arr.len(),
                self.sample_size
            )));
        }
        // Store Python reference: Bound -> Py<PyAny> via unbind (PyO3 0.27).
        self.arrays.borrow_mut().push(array.clone().unbind());
        Ok(())
    }

    /// Submit a batch (2D array: num_samples × sample_size); stores row views (no copy).
    fn submit_batch(
        &self,
        _py: Python<'_>,
        batch_data: &Bound<'_, PyAny>,
        num_samples: usize,
        sample_size: usize,
    ) -> PyResult<()> {
        if sample_size != self.sample_size {
            return Err(PyRuntimeError::new_err(format!(
                "coalesce.submit_batch: sample_size {} != batcher sample_size {}",
                sample_size, self.sample_size
            )));
        }
        let arr = batch_data.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
            PyRuntimeError::new_err(
                "coalesce.submit_batch: argument must be 2D NumPy array (float64)",
            )
        })?;
        let shape = arr.shape();
        if shape[0] != num_samples || shape[1] != sample_size {
            return Err(PyRuntimeError::new_err(format!(
                "coalesce.submit_batch: shape {:?} != (num_samples={}, sample_size={})",
                shape, num_samples, sample_size
            )));
        }
        for i in 0..num_samples {
            let row = batch_data.get_item(i)?;
            self.arrays.borrow_mut().push(row.clone().unbind());
        }
        Ok(())
    }

    /// Result after exiting the context: (QuantumTensor, boundaries). Raises if not yet flushed.
    fn get_result(&self) -> PyResult<CoalesceResult> {
        self.result.borrow_mut().take().ok_or_else(|| {
            PyRuntimeError::new_err("coalesce: result not available (exit the 'with' block first)")
        })
    }
}

/// Helper to detect PyTorch tensor
fn is_pytorch_tensor(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let type_obj = obj.get_type();
    let name = type_obj.name()?;
    if name != "Tensor" {
        return Ok(false);
    }
    let module = type_obj.module()?;
    let module_name = module.to_str()?;
    Ok(module_name == "torch")
}

/// Helper to validate CPU tensor
fn validate_tensor(tensor: &Bound<'_, PyAny>) -> PyResult<()> {
    if !is_pytorch_tensor(tensor)? {
        return Err(PyRuntimeError::new_err("Object is not a PyTorch Tensor"));
    }

    let device = tensor.getattr("device")?;
    let device_type: String = device.getattr("type")?.extract()?;

    if device_type != "cpu" {
        return Err(PyRuntimeError::new_err(format!(
            "Only CPU tensors are currently supported for this path. Got device: {}",
            device_type
        )));
    }

    Ok(())
}

/// Check if a PyTorch tensor is on a CUDA device
fn is_cuda_tensor(tensor: &Bound<'_, PyAny>) -> PyResult<bool> {
    let device = tensor.getattr("device")?;
    let device_type: String = device.getattr("type")?.extract()?;
    Ok(device_type == "cuda")
}

/// Validate array/tensor shape (must be 1D or 2D)
///
/// Args:
///     ndim: Number of dimensions
///     context: Context string for error message (e.g., "array", "tensor", "CUDA tensor")
///
/// Returns:
///     Ok(()) if shape is valid (1D or 2D), otherwise returns an error
fn validate_shape(ndim: usize, context: &str) -> PyResult<()> {
    match ndim {
        1 | 2 => Ok(()),
        _ => {
            let item_type = if context.contains("array") {
                "array"
            } else {
                "tensor"
            };
            Err(PyRuntimeError::new_err(format!(
                "Unsupported {} shape: {}D. Expected 1D {} for single sample \
                 encoding or 2D {} (batch_size, features) for batch encoding.",
                context, ndim, item_type, item_type
            )))
        }
    }
}

/// Get the CUDA device index from a PyTorch tensor
fn get_tensor_device_id(tensor: &Bound<'_, PyAny>) -> PyResult<i32> {
    let device = tensor.getattr("device")?;
    let device_index: i32 = device.getattr("index")?.extract()?;
    Ok(device_index)
}

/// Validate a CUDA tensor for direct GPU encoding
/// Checks: dtype=float64, contiguous, non-empty, device_id matches engine
fn validate_cuda_tensor_for_encoding(
    tensor: &Bound<'_, PyAny>,
    expected_device_id: usize,
    encoding_method: &str,
) -> PyResult<()> {
    let method = encoding_method.to_ascii_lowercase();
    // Check encoding method support (currently amplitude and angle are supported for CUDA tensors)
    if method != "amplitude" && method != "angle" {
        return Err(PyRuntimeError::new_err(format!(
            "CUDA tensor encoding currently only supports 'amplitude' and 'angle' methods, got '{}'. \
             Use tensor.cpu() to convert to CPU tensor for other encoding methods.",
            encoding_method
        )));
    }

    // Check dtype is float64
    let dtype = tensor.getattr("dtype")?;
    let dtype_str: String = dtype.str()?.extract()?;
    if !dtype_str.contains("float64") {
        return Err(PyRuntimeError::new_err(format!(
            "CUDA tensor must have dtype float64, got {}. Use tensor.to(torch.float64)",
            dtype_str
        )));
    }

    // Check contiguous
    let is_contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
    if !is_contiguous {
        return Err(PyRuntimeError::new_err(
            "CUDA tensor must be contiguous. Use tensor.contiguous()",
        ));
    }

    // Check non-empty
    let numel: usize = tensor.call_method0("numel")?.extract()?;
    if numel == 0 {
        return Err(PyRuntimeError::new_err("CUDA tensor cannot be empty"));
    }

    // Check device matches engine
    let tensor_device_id = get_tensor_device_id(tensor)?;
    if tensor_device_id as usize != expected_device_id {
        return Err(PyRuntimeError::new_err(format!(
            "Device mismatch: tensor is on cuda:{}, but engine is on cuda:{}. \
             Move tensor with tensor.to('cuda:{}')",
            tensor_device_id, expected_device_id, expected_device_id
        )));
    }

    Ok(())
}

/// DLPack tensor information extracted from a PyCapsule
///
/// This struct owns the DLManagedTensor pointer and ensures proper cleanup
/// via the DLPack deleter when dropped (RAII pattern).
struct DLPackTensorInfo {
    /// Raw DLManagedTensor pointer from PyTorch DLPack capsule
    /// This is owned by this struct and will be freed via deleter on drop
    managed_ptr: *mut DLManagedTensor,
    /// Data pointer inside dl_tensor (GPU memory, owned by managed_ptr)
    data_ptr: *const f64,
    shape: Vec<i64>,
    /// CUDA device ID from DLPack metadata.
    /// Used for defensive validation against PyTorch API device ID.
    device_id: i32,
}

impl Drop for DLPackTensorInfo {
    fn drop(&mut self) {
        unsafe {
            if !self.managed_ptr.is_null() {
                // Per DLPack protocol: consumer must call deleter exactly once
                if let Some(deleter) = (*self.managed_ptr).deleter {
                    deleter(self.managed_ptr);
                }
                // Prevent double-free
                self.managed_ptr = std::ptr::null_mut();
            }
        }
    }
}

/// Extract GPU pointer from PyTorch tensor's __dlpack__() capsule
///
/// Uses the DLPack protocol to obtain a zero-copy view of the tensor's GPU memory.
/// The returned `DLPackTensorInfo` owns the DLManagedTensor and will automatically
/// call the deleter when dropped, ensuring proper resource cleanup.
///
/// # Safety
/// The returned `data_ptr` points to GPU memory owned by the source tensor.
/// The caller must ensure the source tensor remains alive and unmodified
/// for the entire duration that `data_ptr` is in use. Python's GIL ensures
/// the tensor won't be garbage collected during `encode()`, but the caller
/// must not deallocate or resize the tensor while encoding is in progress.
fn extract_dlpack_tensor(_py: Python<'_>, tensor: &Bound<'_, PyAny>) -> PyResult<DLPackTensorInfo> {
    // Call tensor.__dlpack__() to get PyCapsule
    // Note: PyTorch's __dlpack__ uses the default stream when called without arguments
    let capsule = tensor.call_method0("__dlpack__")?;

    // Extract the DLManagedTensor pointer from the capsule
    const DLTENSOR_NAME: &[u8] = b"dltensor\0";

    unsafe {
        let capsule_ptr = capsule.as_ptr();
        let managed_ptr =
            ffi::PyCapsule_GetPointer(capsule_ptr, DLTENSOR_NAME.as_ptr() as *const i8)
                as *mut DLManagedTensor;

        if managed_ptr.is_null() {
            return Err(PyRuntimeError::new_err(
                "Failed to extract DLManagedTensor from PyCapsule",
            ));
        }

        let dl_tensor = &(*managed_ptr).dl_tensor;

        // Extract data pointer with null check
        if dl_tensor.data.is_null() {
            return Err(PyRuntimeError::new_err(
                "DLPack tensor has null data pointer",
            ));
        }
        let data_ptr = dl_tensor.data as *const f64;

        // Extract shape
        let ndim = dl_tensor.ndim as usize;
        let shape = if ndim > 0 && !dl_tensor.shape.is_null() {
            std::slice::from_raw_parts(dl_tensor.shape, ndim).to_vec()
        } else {
            vec![]
        };

        // Extract device_id
        let device_id = dl_tensor.device.device_id;

        // Rename the capsule to "used_dltensor" as per DLPack protocol
        // This prevents PyTorch from trying to delete it when the capsule is garbage collected
        const USED_DLTENSOR_NAME: &[u8] = b"used_dltensor\0";
        ffi::PyCapsule_SetName(capsule_ptr, USED_DLTENSOR_NAME.as_ptr() as *const i8);

        Ok(DLPackTensorInfo {
            managed_ptr,
            data_ptr,
            shape,
            device_id,
        })
    }
}

/// PyO3 wrapper for QdpEngine
///
/// Provides Python bindings for GPU-accelerated quantum state encoding.
#[pyclass]
struct QdpEngine {
    engine: CoreEngine,
}

#[pymethods]
impl QdpEngine {
    /// Initialize QDP engine on specified GPU device
    ///
    /// Args:
    ///     device_id: CUDA device ID (typically 0)
    ///     precision: Output precision ("float32" default, or "float64")
    ///
    /// Returns:
    ///     QdpEngine instance
    ///
    /// Raises:
    ///     RuntimeError: If CUDA device initialization fails
    #[new]
    #[pyo3(signature = (device_id=0, precision="float32"))]
    fn new(device_id: usize, precision: &str) -> PyResult<Self> {
        let precision = match precision.to_ascii_lowercase().as_str() {
            "float32" | "f32" | "float" => Precision::Float32,
            "float64" | "f64" | "double" => Precision::Float64,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unsupported precision '{}'. Use 'float32' (default) or 'float64'.",
                    other
                )));
            }
        };

        let engine = CoreEngine::new_with_precision(device_id, precision)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize: {}", e)))?;
        Ok(Self { engine })
    }

    /// Encode classical data into quantum state (auto-detects input type)
    ///
    /// Args:
    ///     data: Input data - supports:
    ///         - Python list: [1.0, 2.0, 3.0, 4.0]
    ///         - NumPy array: 1D (single sample) or 2D (batch) array
    ///         - PyTorch tensor: CPU tensor (float64 recommended; will be copied to GPU)
    ///         - String path: .parquet, .arrow, .feather, .npy, .pt, .pth, .pb file
    ///         - pathlib.Path: Path object (converted via os.fspath())
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy ("amplitude" default, "angle", or "basis")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack-compatible tensor for zero-copy PyTorch integration
    ///         Shape: [batch_size, 2^num_qubits]
    ///
    /// Example:
    ///     >>> engine = QdpEngine(0)
    ///     >>> # From list
    ///     >>> tensor = engine.encode([1.0, 2.0, 3.0, 4.0], 2)
    ///     >>> # From NumPy batch
    ///     >>> tensor = engine.encode(np.random.randn(64, 4), 2)
    ///     >>> # From file path string
    ///     >>> tensor = engine.encode("data.parquet", 10)
    ///     >>> # From pathlib.Path
    ///     >>> from pathlib import Path
    ///     >>> tensor = engine.encode(Path("data.npy"), 10)
    #[pyo3(signature = (data, num_qubits, encoding_method="amplitude"))]
    fn encode(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        // Check if it's a string path
        if let Ok(path) = data.extract::<String>() {
            return self.encode_from_file(&path, num_qubits, encoding_method);
        }

        // Check if it's a pathlib.Path or os.PathLike object (has __fspath__ method)
        if data.hasattr("__fspath__")? {
            let path: String = data.call_method0("__fspath__")?.extract()?;
            return self.encode_from_file(&path, num_qubits, encoding_method);
        }

        // Check if it's a NumPy array
        if data.hasattr("__array_interface__")? {
            return self.encode_from_numpy(data, num_qubits, encoding_method);
        }

        // Check if it's a PyTorch tensor
        if is_pytorch_tensor(data)? {
            return self.encode_from_pytorch(data, num_qubits, encoding_method);
        }

        // Fallback: try to extract as Vec<f64> (Python list)
        self.encode_from_list(data, num_qubits, encoding_method)
    }

    /// Encode batch from raw bytes (float64, native/ little-endian order).
    ///
    /// For concurrent GPU submissions: pass `arr.tobytes()` so the heavy work
    /// (interpret bytes as f64 + encode_batch) runs inside GIL release; the initial
    /// bytes copy still holds GIL. Prefer encode_list for large batches when order allows.
    #[pyo3(signature = (data, num_samples, sample_size, num_qubits, encoding_method="amplitude"))]
    fn encode_batch_from_bytes(
        &self,
        data: &Bound<'_, PyAny>,
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let py_bytes = data.cast::<PyBytes>().map_err(|_| {
            PyRuntimeError::new_err(
                "encode_batch_from_bytes: data must be bytes (e.g. arr.tobytes())",
            )
        })?;
        let bytes_slice = py_bytes.as_bytes();
        let len_bytes = bytes_slice.len();
        if len_bytes % 8 != 0 {
            return Err(PyRuntimeError::new_err(
                "encode_batch_from_bytes: bytes length must be multiple of 8 (float64)",
            ));
        }
        let n_f64 = len_bytes / 8;
        if n_f64 != num_samples * sample_size {
            return Err(PyRuntimeError::new_err(format!(
                "encode_batch_from_bytes: bytes length {} ({} f64) != num_samples * sample_size ({} * {})",
                len_bytes, n_f64, num_samples, sample_size
            )));
        }
        // Owned copy so the detach closure is Send (no raw pointers).
        let bytes_copy: Vec<u8> = bytes_slice.to_vec();
        let enc = encoding_method.to_string();
        let ptr_raw: usize = data.py().detach(|| {
            let vec_f64: Vec<f64> = bytes_copy
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            self.engine
                .encode_batch(
                    vec_f64.as_slice(),
                    num_samples,
                    sample_size,
                    num_qubits,
                    enc.as_str(),
                )
                .map(|p| p as usize)
                .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))
        })?;
        Ok(QuantumTensor {
            ptr: ptr_raw as *mut _,
            consumed: false,
        })
    }

    /// Encode a list of arrays in one coalesced batch (one FFI, pipeline).
    ///
    /// Reduces FFI and Python object overhead vs N separate encode() calls (§3.9, §3.11.2).
    /// Each array must be 1D float64 of length `sample_size`.
    ///
    /// Args:
    ///     list_of_arrays: List of 1D NumPy arrays (float64), each of length sample_size
    ///     sample_size: Number of elements per sample
    ///     num_qubits: Number of qubits
    ///     encoding_method: "amplitude", "angle", or "basis"
    ///
    /// Returns:
    ///     Tuple of (QuantumTensor, boundaries). QuantumTensor is the merged batch (DLPack);
    ///     boundaries is a list of (start_sample, num_samples), one (i, 1) per input array in order.
    ///
    /// Example:
    ///     >>> engine = QdpEngine(0)
    ///     >>> arrays = [np.random.randn(64).astype(np.float64) for _ in range(10)]
    ///     >>> batch, boundaries = engine.encode_list(arrays, 64, 4, "amplitude")
    ///     >>> len(boundaries) == 10
    ///     True
    ///
    /// Scope-based coalescing (§3.4, §3.6): use `with engine.coalesce(...) as batcher:`
    /// then `batcher.submit(array)` / `batcher.submit_batch(...)`; on exit, flush runs
    /// and `batcher.get_result()` returns (QuantumTensor, boundaries).
    #[pyo3(signature = (sample_size, num_qubits, encoding_method="amplitude"))]
    fn coalesce(
        slf: PyRef<'_, Self>,
        _py: Python<'_>,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<CoalesceBatcher> {
        // PyRef<'_, QdpEngine> -> Py<QdpEngine> (From), then -> Py<PyAny> (Into); see PyO3 0.27 types.
        let engine_py: Py<QdpEngine> = slf.into();
        Ok(CoalesceBatcher {
            engine: engine_py.into(),
            sample_size,
            num_qubits,
            encoding_method: encoding_method.to_string(),
            arrays: RefCell::new(Vec::new()),
            result: RefCell::new(None),
        })
    }

    #[pyo3(signature = (list_of_arrays, sample_size, num_qubits, encoding_method="amplitude"))]
    fn encode_list(
        &self,
        _py: Python<'_>,
        list_of_arrays: &Bound<'_, PyAny>,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<(QuantumTensor, Vec<(usize, usize)>)> {
        let list = list_of_arrays.cast::<PyList>().map_err(|_| {
            PyRuntimeError::new_err(
                "encode_list: argument must be a list of 1D NumPy arrays (float64)",
            )
        })?;
        if list.len() == 0 {
            return Err(PyRuntimeError::new_err(
                "encode_list: list must not be empty",
            ));
        }
        let mut arrays: Vec<PyReadonlyArray1<f64>> = Vec::with_capacity(list.len());
        for (i, item) in list.iter().enumerate() {
            let arr: PyReadonlyArray1<f64> =
                item.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
                    PyRuntimeError::new_err(
                        "encode_list: each item must be a 1D NumPy array with dtype float64",
                    )
                })?;
            if arr.len() != sample_size {
                return Err(PyRuntimeError::new_err(format!(
                    "encode_list: array at index {} has length {}, expected sample_size {}",
                    i,
                    arr.len(),
                    sample_size
                )));
            }
            arrays.push(arr);
        }
        let mut refs: Vec<&[f64]> = Vec::with_capacity(arrays.len());
        for a in &arrays {
            refs.push(a.as_slice().map_err(|_| {
                PyRuntimeError::new_err("encode_list: each array must be contiguous (C-order)")
            })?);
        }
        let (ptr, boundaries) = self
            .engine
            .encode_list(&refs, sample_size, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
        Ok((
            QuantumTensor {
                ptr,
                consumed: false,
            },
            boundaries,
        ))
    }

    /// Submit encode task without blocking; returns a handle to retrieve result later (§3.9).
    ///
    /// Enables Dynamic Batching: submit multiple requests, then call `handle.get()` to receive
    /// results in order. Scheduler coalesces tasks and runs pipeline once per batch.
    ///
    /// Args:
    ///     data: Single sample - 1D NumPy array (float64) or list of floats; length must equal 2^num_qubits for amplitude encoding
    ///     num_qubits: Number of qubits
    ///     encoding_method: "amplitude", "angle", or "basis"
    ///
    /// Returns:
    ///     EncodeHandle; call `.get()` to block and get QuantumTensor.
    ///
    /// Example:
    ///     >>> engine = QdpEngine(0)
    ///     >>> futures = [engine.encode_async(np.array([...], dtype=np.float64), 4, "amplitude") for _ in range(10)]
    ///     >>> results = [f.get() for f in futures]
    #[cfg(target_os = "linux")]
    #[pyo3(signature = (data, num_qubits, encoding_method="amplitude"))]
    fn encode_async(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<EncodeHandle> {
        let data_slice: Vec<f64> = if data.hasattr("__array_interface__")? {
            let arr = data.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
                PyRuntimeError::new_err(
                    "encode_async: data must be 1D NumPy array with dtype float64",
                )
            })?;
            arr.as_slice()
                .map_err(|_| {
                    PyRuntimeError::new_err("encode_async: array must be contiguous (C-order)")
                })?
                .to_vec()
        } else {
            data.extract::<Vec<f64>>().map_err(|_| {
                PyRuntimeError::new_err(
                    "encode_async: data must be 1D NumPy array (float64) or list of floats",
                )
            })?
        };
        let handle = self
            .engine
            .encode_async(&data_slice, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(EncodeHandle {
            inner: Some(handle),
        })
    }

    /// Submit one batch without blocking; returns BatchHandle. Call `.get()` to receive QuantumTensor.
    /// Use for encode_stream: submit multiple batches so the Rust pool keeps the GPU fed, then get() in order.
    #[cfg(target_os = "linux")]
    #[pyo3(signature = (data, num_qubits, encoding_method="amplitude"))]
    fn encode_batch_submit(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<BatchHandle> {
        let array_2d = data.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
            PyRuntimeError::new_err("encode_batch_submit: data must be 2D NumPy array (float64)")
        })?;
        let shape = array_2d.shape();
        let num_samples = shape[0];
        let sample_size = shape[1];
        let data_slice = array_2d.as_slice().map_err(|_| {
            PyRuntimeError::new_err("encode_batch_submit: array must be contiguous (C-order)")
        })?;
        // Zero-copy: pass slice to Rust; buffer must stay alive until handle.get() returns.
        let handle = self
            .engine
            .encode_batch_submit(
                data_slice,
                num_samples,
                sample_size,
                num_qubits,
                encoding_method,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
        Ok(BatchHandle {
            inner: Some(handle),
        })
    }

    /// Encode from NumPy array (1D or 2D)
    fn encode_from_numpy(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ndim: usize = data.getattr("ndim")?.extract()?;
        validate_shape(ndim, "array")?;

        match ndim {
            1 => {
                // 1D array: single sample encoding (zero-copy if already contiguous)
                let array_1d = data.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to extract 1D NumPy array. Ensure dtype is float64.",
                    )
                })?;
                let data_slice = array_1d.as_slice().map_err(|_| {
                    PyRuntimeError::new_err("NumPy array must be contiguous (C-order)")
                })?;
                let ptr = self
                    .engine
                    .encode(data_slice, num_qubits, encoding_method)
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                Ok(QuantumTensor {
                    ptr,
                    consumed: false,
                })
            }
            2 => {
                // 2D array: batch encoding. Zero-copy: pass slice to Rust (cudaHostRegister path).
                // Caller keeps array alive for the duration of encode_batch (blocking).
                let array_2d = data.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to extract 2D NumPy array. Ensure dtype is float64.",
                    )
                })?;
                let shape = array_2d.shape();
                let num_samples = shape[0];
                let sample_size = shape[1];
                let data_slice = array_2d.as_slice().map_err(|_| {
                    PyRuntimeError::new_err("NumPy array must be contiguous (C-order)")
                })?;
                let ptr = self
                    .engine
                    .encode_batch(
                        data_slice,
                        num_samples,
                        sample_size,
                        num_qubits,
                        encoding_method,
                    )
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                Ok(QuantumTensor {
                    ptr,
                    consumed: false,
                })
            }
            _ => unreachable!("validate_shape() should have caught invalid ndim"),
        }
    }

    /// Encode from PyTorch tensor (1D or 2D)
    fn encode_from_pytorch(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        // Check if it's a CUDA tensor - use zero-copy GPU encoding via DLPack
        if is_cuda_tensor(data)? {
            // Validate CUDA tensor for direct GPU encoding
            validate_cuda_tensor_for_encoding(
                data,
                self.engine.device().ordinal(),
                encoding_method,
            )?;

            // Extract GPU pointer via DLPack (RAII wrapper ensures deleter is called)
            let dlpack_info = extract_dlpack_tensor(data.py(), data)?;

            // ensure PyTorch API and DLPack metadata agree on device ID
            let pytorch_device_id = get_tensor_device_id(data)?;
            if dlpack_info.device_id != pytorch_device_id {
                return Err(PyRuntimeError::new_err(format!(
                    "Device ID mismatch: PyTorch reports device {}, but DLPack metadata reports {}. \
                     This indicates an inconsistency between PyTorch and DLPack device information.",
                    pytorch_device_id, dlpack_info.device_id
                )));
            }

            let ndim: usize = data.call_method0("dim")?.extract()?;
            validate_shape(ndim, "CUDA tensor")?;

            match ndim {
                1 => {
                    // 1D CUDA tensor: single sample encoding
                    let input_len = dlpack_info.shape[0] as usize;
                    // SAFETY: dlpack_info.data_ptr was validated via DLPack protocol from a
                    // valid PyTorch CUDA tensor. The tensor remains alive during this call
                    // (held by Python's GIL), and we validated dtype/contiguity/device above.
                    // The DLPackTensorInfo RAII wrapper will call deleter when dropped.
                    let ptr = unsafe {
                        self.engine
                            .encode_from_gpu_ptr(
                                dlpack_info.data_ptr,
                                input_len,
                                num_qubits,
                                encoding_method,
                            )
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("Encoding failed: {}", e))
                            })?
                    };
                    return Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    });
                }
                2 => {
                    // 2D CUDA tensor: batch encoding
                    let num_samples = dlpack_info.shape[0] as usize;
                    let sample_size = dlpack_info.shape[1] as usize;
                    // SAFETY: Same as above - pointer from validated DLPack tensor
                    let ptr = unsafe {
                        self.engine
                            .encode_batch_from_gpu_ptr(
                                dlpack_info.data_ptr,
                                num_samples,
                                sample_size,
                                num_qubits,
                                encoding_method,
                            )
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!("Encoding failed: {}", e))
                            })?
                    };
                    return Ok(QuantumTensor {
                        ptr,
                        consumed: false,
                    });
                }
                _ => unreachable!("validate_shape() should have caught invalid ndim"),
            }
        }

        // CPU tensor path
        validate_tensor(data)?;
        // PERF: Avoid Tensor -> Python list -> Vec deep copies.
        //
        // For CPU tensors, `tensor.detach().numpy()` returns a NumPy view that shares the same
        // underlying memory (zero-copy) when the tensor is C-contiguous. We can then borrow a
        // `&[f64]` directly via pyo3-numpy.
        let ndim: usize = data.call_method0("dim")?.extract()?;
        validate_shape(ndim, "tensor")?;
        let numpy_view = data
            .call_method0("detach")?
            .call_method0("numpy")
            .map_err(|_| {
                PyRuntimeError::new_err(
                    "Failed to convert torch.Tensor to NumPy view. Ensure the tensor is on CPU \
                     and does not require grad (try: tensor = tensor.detach().cpu())",
                )
            })?;

        match ndim {
            1 => {
                // 1D tensor: single sample encoding
                let array_1d = numpy_view.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to extract NumPy view as float64 array. Ensure dtype is float64 \
                             (try: tensor = tensor.to(torch.float64))",
                    )
                })?;
                let data_slice = array_1d.as_slice().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Tensor must be contiguous (C-order) to get zero-copy slice \
                         (try: tensor = tensor.contiguous())",
                    )
                })?;
                let ptr = self
                    .engine
                    .encode(data_slice, num_qubits, encoding_method)
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                Ok(QuantumTensor {
                    ptr,
                    consumed: false,
                })
            }
            2 => {
                // 2D tensor: batch encoding
                let array_2d = numpy_view.extract::<PyReadonlyArray2<f64>>().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Failed to extract NumPy view as float64 array. Ensure dtype is float64 \
                             (try: tensor = tensor.to(torch.float64))",
                    )
                })?;
                let shape = array_2d.shape();
                let num_samples = shape[0];
                let sample_size = shape[1];
                let data_slice = array_2d.as_slice().map_err(|_| {
                    PyRuntimeError::new_err(
                        "Tensor must be contiguous (C-order) to get zero-copy slice \
                         (try: tensor = tensor.contiguous())",
                    )
                })?;
                let ptr = self
                    .engine
                    .encode_batch(
                        data_slice,
                        num_samples,
                        sample_size,
                        num_qubits,
                        encoding_method,
                    )
                    .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
                Ok(QuantumTensor {
                    ptr,
                    consumed: false,
                })
            }
            _ => unreachable!("validate_shape() should have caught invalid ndim"),
        }
    }

    /// Encode from Python list
    fn encode_from_list(
        &self,
        data: &Bound<'_, PyAny>,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let vec_data = data.extract::<Vec<f64>>().map_err(|_| {
            PyRuntimeError::new_err(
                "Unsupported data type. Expected: list, NumPy array, PyTorch tensor, or file path",
            )
        })?;
        let ptr = self
            .engine
            .encode(&vec_data, num_qubits, encoding_method)
            .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Internal helper to encode from file based on extension
    fn encode_from_file(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ptr = if path.ends_with(".parquet") {
            self.engine
                .encode_from_parquet(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from parquet failed: {}", e))
                })?
        } else if path.ends_with(".arrow") || path.ends_with(".feather") {
            self.engine
                .encode_from_arrow_ipc(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from Arrow IPC failed: {}", e))
                })?
        } else if path.ends_with(".npy") {
            self.engine
                .encode_from_numpy(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from NumPy failed: {}", e))
                })?
        } else if path.ends_with(".pt") || path.ends_with(".pth") {
            self.engine
                .encode_from_torch(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from PyTorch failed: {}", e))
                })?
        } else if path.ends_with(".pb") {
            self.engine
                .encode_from_tensorflow(path, num_qubits, encoding_method)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Encoding from TensorFlow failed: {}", e))
                })?
        } else {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported file format. Expected .parquet, .arrow, .feather, .npy, .pt, .pth, or .pb, got: {}",
                path
            )));
        };

        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }

    /// Encode from TensorFlow TensorProto file
    ///
    /// Args:
    ///     path: Path to TensorProto file (.pb)
    ///     num_qubits: Number of qubits for encoding
    ///     encoding_method: Encoding strategy (currently only "amplitude")
    ///
    /// Returns:
    ///     QuantumTensor: DLPack tensor containing all encoded states
    ///
    /// Example:
    ///     >>> engine = QdpEngine(device_id=0)
    ///     >>> batched = engine.encode_from_tensorflow("data.pb", 16, "amplitude")
    ///     >>> torch_tensor = torch.from_dlpack(batched)  # Shape: [200, 65536]
    fn encode_from_tensorflow(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> PyResult<QuantumTensor> {
        let ptr = self
            .engine
            .encode_from_tensorflow(path, num_qubits, encoding_method)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Encoding from TensorFlow failed: {}", e))
            })?;
        Ok(QuantumTensor {
            ptr,
            consumed: false,
        })
    }
}

/// Quantum Data Plane (QDP) Python module
///
/// GPU-accelerated quantum data encoding with DLPack integration.
#[pymodule]
fn _qdp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize Rust logging system - respect RUST_LOG environment variable
    // Ref: https://docs.rs/env_logger/latest/env_logger/
    // try_init() won't fail if logger is already initialized (e.g., by another library)
    // This allows Rust log messages to be visible when RUST_LOG is set
    let _ = env_logger::Builder::from_default_env().try_init();

    m.add_class::<QdpEngine>()?;
    m.add_class::<QuantumTensor>()?;
    m.add_class::<CoalesceBatcher>()?;
    #[cfg(target_os = "linux")]
    m.add_class::<EncodeHandle>()?;
    #[cfg(target_os = "linux")]
    m.add_class::<BatchHandle>()?;
    Ok(())
}
