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

// Allow unused_unsafe: CUDA FFI and kernel functions are unsafe in CUDA builds but safe stubs in no-CUDA builds.
// The compiler can't statically determine which path is taken.
#![allow(unused_unsafe)]

pub mod dlpack;
#[cfg(target_os = "linux")]
mod encoding;
pub mod error;
pub mod gpu;
pub mod io;
mod platform;
pub mod preprocessing;
pub mod reader;
pub mod readers;
pub mod tf_proto;
#[macro_use]
mod profiling;

pub use error::{MahoutError, Result, cuda_error_to_string};
pub use gpu::memory::Precision;

#[cfg(target_os = "linux")]
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::sync::Mutex;

use crate::dlpack::DLManagedTensor;
use crate::gpu::get_encoder;
use cudarc::driver::CudaDevice;

/// Main entry point for Mahout QDP
///
/// Manages GPU context and dispatches encoding tasks.
/// Provides unified interface for device management, memory allocation, and DLPack.
pub struct QdpEngine {
    device: Arc<CudaDevice>,
    precision: Precision,
    #[cfg(target_os = "linux")]
    pub(crate) schedulers: Arc<Mutex<crate::gpu::SchedulerMap>>,
    /// Default path for batch encode: N workers, each runs encode_batch_via_pipeline (no sync branch).
    #[cfg(target_os = "linux")]
    pub(crate) batch_pool: Arc<crate::gpu::BatchPool>,
}

impl QdpEngine {
    /// Initialize engine on GPU device
    ///
    /// # Arguments
    /// * `device_id` - CUDA device ID (typically 0)
    pub fn new(device_id: usize) -> Result<Self> {
        Self::new_with_precision(device_id, Precision::Float32)
    }

    /// Initialize engine with explicit precision.
    ///
    /// On Linux, creates a batch pool (single GPU master thread, bounded queue) so encode_batch
    /// uses the pool by default (single path, no sync branch).
    pub fn new_with_precision(device_id: usize, precision: Precision) -> Result<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            MahoutError::Cuda(format!(
                "Failed to initialize CUDA device {}: {:?}",
                device_id, e
            ))
        })?;
        #[cfg(target_os = "linux")]
        let batch_pool = Arc::new(crate::gpu::BatchPool::new(
            device.clone(),
            precision,
            crate::gpu::BatchPool::num_workers_from_env(),
        ));
        Ok(Self {
            device, // CudaDevice::new already returns Arc<CudaDevice> in cudarc 0.11
            precision,
            #[cfg(target_os = "linux")]
            schedulers: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(target_os = "linux")]
            batch_pool,
        })
    }

    /// Encode classical data into quantum state
    ///
    /// Selects encoding strategy, executes on GPU, returns DLPack pointer.
    ///
    /// # Arguments
    /// * `data` - Input data
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// DLPack pointer for zero-copy PyTorch integration
    ///
    /// # Safety
    /// Pointer freed by DLPack deleter, do not free manually.
    pub fn encode(
        &self,
        data: &[f64],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::Encode");

        let encoder = get_encoder(encoding_method)?;
        let state_vector = encoder.encode(&self.device, data, num_qubits)?;
        let state_vector = state_vector.to_precision(&self.device, self.precision)?;
        let dlpack_ptr = {
            crate::profile_scope!("DLPack::Wrap");
            state_vector.to_dlpack()
        };
        Ok(dlpack_ptr)
    }

    /// Get CUDA device reference for advanced operations
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Encode multiple samples in a single fused kernel (most efficient)
    ///
    /// Allocates one large GPU buffer and launches a single batch kernel.
    /// This is faster than encode_batch() as it reduces allocation and kernel launch overhead.
    ///
    /// # Arguments
    /// * `batch_data` - Flattened batch data (all samples concatenated)
    /// * `num_samples` - Number of samples in the batch
    /// * `sample_size` - Size of each sample
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
    ///
    /// Uses the batch pool by default (N workers, each runs encode_batch_via_pipeline).
    /// Single path: no sync branch; pool keeps GPU fed (Triton/vLLM/DALI-style).
    #[cfg(target_os = "linux")]
    pub fn encode_batch(
        &self,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeBatch");
        self.batch_pool
            .submit_handle(
                crate::gpu::BatchJobData::Borrowed {
                    ptr: batch_data.as_ptr(),
                    len: batch_data.len(),
                },
                num_samples,
                sample_size,
                num_qubits,
                encoding_method.to_string(),
            )?
            .get()
    }

    /// Same as encode_batch when not Linux (no pool).
    #[cfg(not(target_os = "linux"))]
    pub fn encode_batch(
        &self,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeBatch");
        let encoder = get_encoder(encoding_method)?;
        let state_vector = encoder.encode_batch_via_pipeline(
            &self.device,
            batch_data,
            num_samples,
            sample_size,
            num_qubits,
        )?;
        let state_vector = state_vector.to_precision(&self.device, self.precision)?;
        Ok(state_vector.to_dlpack())
    }

    /// Submit batch without blocking; returns handle. Call get() to receive DLPack. Use for encode_stream (submit many, get in order).
    #[cfg(target_os = "linux")]
    pub fn encode_batch_submit(
        &self,
        batch_data: &[f64],
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<crate::gpu::BatchHandle> {
        self.batch_pool.submit_handle(
            crate::gpu::BatchJobData::Borrowed {
                ptr: batch_data.as_ptr(),
                len: batch_data.len(),
            },
            num_samples,
            sample_size,
            num_qubits,
            encoding_method.to_string(),
        )
    }

    /// Encode multiple slices in one coalesced batch (run_coalesced + pipeline).
    ///
    /// One FFI call: coalesces all slices into one Pinned buffer, runs
    /// encode_batch_via_pipeline, returns merged batch DLPack and per-slice boundaries.
    /// Reduces FFI and Python object overhead vs N separate encode() calls (§3.9, §3.11.2).
    ///
    /// Ref: docs/optimization/REQUEST_COALESCING_REFERENCE_AND_DESIGN.md §3.9 (encode_list), §3.11.2
    ///
    /// # Arguments
    /// * `slices` - One slice per sample; each slice must have length `sample_size`
    /// * `sample_size` - Number of f64 elements per sample
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// `(dlpack_ptr, boundaries)` where `boundaries` is `[(start_sample, num_samples); N]`
    /// with one `(i, 1)` per slice in order.
    pub fn encode_list(
        &self,
        slices: &[&[f64]],
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<(*mut DLManagedTensor, Vec<(usize, usize)>)> {
        crate::profile_scope!("Mahout::EncodeList");

        let encoder = get_encoder(encoding_method)?;
        let config = crate::gpu::CoalescerConfig::default();
        let (batch_state, boundaries) = crate::gpu::encode_list(
            &self.device,
            encoder.as_ref(),
            sample_size,
            num_qubits,
            config,
            slices,
        )?;
        let state_vector = batch_state.to_precision(&self.device, self.precision)?;
        let dlpack_ptr = state_vector.to_dlpack();
        Ok((dlpack_ptr, boundaries))
    }

    /// Submit encode task without blocking; returns a handle to retrieve result later (§3.9).
    ///
    /// Enables Dynamic Batching: submit multiple requests, then call `handle.get()` to receive
    /// results in order. Scheduler coalesces tasks and runs pipeline once per batch.
    ///
    /// # Arguments
    /// * `data` - Single sample (length must equal 2^num_qubits for amplitude encoding)
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// `EncodeHandle`; call `.get()` to block and get DLPack pointer.
    #[cfg(target_os = "linux")]
    pub fn encode_async(
        &self,
        data: &[f64],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<crate::gpu::EncodeHandle> {
        crate::profile_scope!("Mahout::EncodeAsync");

        let sample_size = data.len();
        get_encoder(encoding_method)?; // validate before taking lock
        let key = (sample_size, num_qubits, encoding_method.to_string());
        let mut guard = self.schedulers.lock().map_err(|e| {
            MahoutError::InvalidInput(format!("encode_async: scheduler lock poisoned: {}", e))
        })?;
        let scheduler = guard.entry(key.clone()).or_insert_with(|| {
            let encoder = get_encoder(&key.2).expect("encoding_method already validated");
            crate::gpu::EncodeScheduler::new(
                self.device.clone(),
                encoder,
                key.0,
                key.1,
                crate::gpu::SchedulerConfig::default(),
            )
        });
        let (tx, rx) = std::sync::mpsc::channel();
        let task = crate::gpu::EncodeTask {
            data: data.to_vec(),
            num_samples: 1,
            sample_size,
            tx,
        };
        scheduler.submit(task)?;
        Ok(crate::gpu::EncodeHandle::new(
            rx,
            self.device.clone(),
            self.precision,
            num_qubits,
        ))
    }

    /// Streaming Parquet encoder with multi-threaded IO
    ///
    /// Uses Producer-Consumer pattern: IO thread reads Parquet while GPU processes data.
    /// Double-buffered (ping-pong) for maximum pipeline overlap.
    ///
    /// # Arguments
    /// * `path` - Path to Parquet file with List<Float64> column
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// DLPack pointer to encoded states [num_samples, 2^num_qubits]
    pub fn encode_from_parquet(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        platform::encode_from_parquet(self, path, num_qubits, encoding_method)
    }

    /// Load data from Arrow IPC file and encode into quantum state
    ///
    /// Supports:
    /// - FixedSizeList<Float64> - fastest, all samples same size
    /// - List<Float64> - flexible, variable sample sizes
    ///
    /// # Arguments
    /// * `path` - Path to Arrow IPC file (.arrow or .feather)
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
    pub fn encode_from_arrow_ipc(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromArrowIPC");

        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadArrowIPCBatch");
            crate::io::read_arrow_ipc_batch(path)?
        };

        self.encode_batch(
            &batch_data,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
        )
    }

    /// Load data from NumPy .npy file and encode into quantum state
    ///
    /// Supports 2D arrays with shape `[num_samples, sample_size]` and dtype `float64`.
    ///
    /// # Arguments
    /// * `path` - Path to NumPy .npy file
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
    pub fn encode_from_numpy(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromNumpy");

        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadNumpyBatch");
            crate::io::read_numpy_batch(path)?
        };

        self.encode_batch(
            &batch_data,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
        )
    }

    /// Load data from PyTorch .pt/.pth file and encode into quantum state
    ///
    /// Supports 1D or 2D tensors saved with `torch.save`.
    /// Requires the `pytorch` feature to be enabled.
    ///
    /// # Arguments
    /// * `path` - Path to PyTorch tensor file (.pt/.pth)
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
    pub fn encode_from_torch(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromTorch");

        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadTorchBatch");
            crate::io::read_torch_batch(path)?
        };

        self.encode_batch(
            &batch_data,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
        )
    }

    /// Load data from TensorFlow TensorProto file and encode into quantum state
    ///
    /// Supports Float64 tensors with shape [batch_size, feature_size] or [n].
    /// Uses efficient parsing with tensor_content when available.
    ///
    /// # Arguments
    /// * `path` - Path to TensorProto file (.pb)
    /// * `num_qubits` - Number of qubits
    /// * `encoding_method` - Strategy: "amplitude", "angle", or "basis"
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
    pub fn encode_from_tensorflow(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromTensorFlow");

        let (batch_data, num_samples, sample_size) = {
            crate::profile_scope!("IO::ReadTensorFlowBatch");
            crate::io::read_tensorflow_batch(path)?
        };

        self.encode_batch(
            &batch_data,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
        )
    }

    /// Encode from existing GPU pointer (zero-copy for CUDA tensors)
    ///
    /// This method enables zero-copy encoding from PyTorch CUDA tensors by accepting
    /// a raw GPU pointer directly, avoiding the GPU→CPU→GPU copy that would otherwise
    /// be required.
    ///
    /// TODO: Refactor to use QuantumEncoder trait (add `encode_from_gpu_ptr` to trait)
    /// to reduce duplication with AmplitudeEncoder::encode(). This would also make it
    /// easier to add GPU pointer support for other encoders (angle, basis) in the future.
    ///
    /// # Arguments
    /// * `input_d` - Device pointer to input data (f64 array on GPU)
    /// * `input_len` - Number of f64 elements in the input
    /// * `num_qubits` - Number of qubits for encoding
    /// * `encoding_method` - Strategy (currently "amplitude" and "angle" supported)
    ///
    /// # Returns
    /// DLPack pointer for zero-copy PyTorch integration
    ///
    /// # Safety
    /// The input pointer must:
    /// - Point to valid GPU memory on the same device as the engine
    /// - Contain at least `input_len` f64 elements
    /// - Remain valid for the duration of this call
    #[cfg(target_os = "linux")]
    pub unsafe fn encode_from_gpu_ptr(
        &self,
        input_d: *const f64,
        input_len: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromGpuPtr");

        if input_len == 0 {
            return Err(MahoutError::InvalidInput(
                "Input data cannot be empty".into(),
            ));
        }

        let state_len = 1usize << num_qubits;
        let method = encoding_method.to_ascii_lowercase();

        match method.as_str() {
            "amplitude" => {
                if input_len > state_len {
                    return Err(MahoutError::InvalidInput(format!(
                        "Input size {} exceeds state vector size {} (2^{} qubits)",
                        input_len, state_len, num_qubits
                    )));
                }

                let state_vector = {
                    crate::profile_scope!("GPU::Alloc");
                    gpu::GpuStateVector::new(&self.device, num_qubits)?
                };

                let inv_norm = {
                    crate::profile_scope!("GPU::NormFromPtr");
                    // SAFETY: input_d validity is guaranteed by the caller's safety contract
                    unsafe {
                        gpu::AmplitudeEncoder::calculate_inv_norm_gpu(
                            &self.device,
                            input_d,
                            input_len,
                        )?
                    }
                };

                let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
                    MahoutError::InvalidInput(
                        "State vector precision mismatch (expected float64 buffer)".to_string(),
                    )
                })?;

                {
                    crate::profile_scope!("GPU::KernelLaunch");
                    let ret = unsafe {
                        qdp_kernels::launch_amplitude_encode(
                            input_d,
                            state_ptr as *mut std::ffi::c_void,
                            input_len,
                            state_len,
                            inv_norm,
                            std::ptr::null_mut(), // default stream
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(format!(
                            "Amplitude encode kernel failed with CUDA error code: {} ({})",
                            ret,
                            cuda_error_to_string(ret)
                        )));
                    }
                }

                {
                    crate::profile_scope!("GPU::Synchronize");
                    self.device.synchronize().map_err(|e| {
                        MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e))
                    })?;
                }

                let state_vector = state_vector.to_precision(&self.device, self.precision)?;
                Ok(state_vector.to_dlpack())
            }
            "angle" => {
                if input_len != num_qubits {
                    return Err(MahoutError::InvalidInput(format!(
                        "Angle encoding expects {} values (one per qubit), got {}",
                        num_qubits, input_len
                    )));
                }

                let state_vector = {
                    crate::profile_scope!("GPU::Alloc");
                    gpu::GpuStateVector::new(&self.device, num_qubits)?
                };

                let state_ptr = state_vector.ptr_f64().ok_or_else(|| {
                    MahoutError::InvalidInput(
                        "State vector precision mismatch (expected float64 buffer)".to_string(),
                    )
                })?;

                {
                    crate::profile_scope!("GPU::KernelLaunch");
                    let ret = unsafe {
                        qdp_kernels::launch_angle_encode(
                            input_d,
                            state_ptr as *mut std::ffi::c_void,
                            state_len,
                            num_qubits as u32,
                            std::ptr::null_mut(), // default stream
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(format!(
                            "Angle encoding kernel failed with CUDA error code: {} ({})",
                            ret,
                            cuda_error_to_string(ret)
                        )));
                    }
                }

                {
                    crate::profile_scope!("GPU::Synchronize");
                    self.device.synchronize().map_err(|e| {
                        MahoutError::Cuda(format!("CUDA device synchronize failed: {:?}", e))
                    })?;
                }

                let state_vector = state_vector.to_precision(&self.device, self.precision)?;
                Ok(state_vector.to_dlpack())
            }
            _ => Err(MahoutError::NotImplemented(format!(
                "GPU pointer encoding currently only supports 'amplitude' and 'angle' methods, got '{}'",
                encoding_method
            ))),
        }
    }

    /// Encode batch from existing GPU pointer (zero-copy for CUDA tensors)
    ///
    /// This method enables zero-copy batch encoding from PyTorch CUDA tensors.
    ///
    /// TODO: Refactor to use QuantumEncoder trait (see `encode_from_gpu_ptr` TODO).
    ///
    /// # Arguments
    /// * `input_batch_d` - Device pointer to batch input data (flattened f64 array on GPU)
    /// * `num_samples` - Number of samples in the batch
    /// * `sample_size` - Size of each sample in f64 elements
    /// * `num_qubits` - Number of qubits for encoding
    /// * `encoding_method` - Strategy (currently "amplitude" and "angle" supported)
    ///
    /// # Returns
    /// Single DLPack pointer containing all encoded states (shape: [num_samples, 2^num_qubits])
    ///
    /// # Safety
    /// The input pointer must:
    /// - Point to valid GPU memory on the same device as the engine
    /// - Contain at least `num_samples * sample_size` f64 elements
    /// - Remain valid for the duration of this call
    #[cfg(target_os = "linux")]
    pub unsafe fn encode_batch_from_gpu_ptr(
        &self,
        input_batch_d: *const f64,
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeBatchFromGpuPtr");

        if num_samples == 0 {
            return Err(MahoutError::InvalidInput(
                "Number of samples cannot be zero".into(),
            ));
        }

        if sample_size == 0 {
            return Err(MahoutError::InvalidInput(
                "Sample size cannot be zero".into(),
            ));
        }

        let state_len = 1usize << num_qubits;
        let method = encoding_method.to_ascii_lowercase();

        match method.as_str() {
            "amplitude" => {
                if sample_size > state_len {
                    return Err(MahoutError::InvalidInput(format!(
                        "Sample size {} exceeds state vector size {} (2^{} qubits)",
                        sample_size, state_len, num_qubits
                    )));
                }

                let batch_state_vector = {
                    crate::profile_scope!("GPU::AllocBatch");
                    gpu::GpuStateVector::new_batch(&self.device, num_samples, num_qubits)?
                };

                let inv_norms_gpu = {
                    crate::profile_scope!("GPU::BatchNormKernel");
                    use cudarc::driver::DevicePtrMut;

                    let mut buffer = self.device.alloc_zeros::<f64>(num_samples).map_err(|e| {
                        MahoutError::MemoryAllocation(format!(
                            "Failed to allocate norm buffer: {:?}",
                            e
                        ))
                    })?;

                    let ret = unsafe {
                        qdp_kernels::launch_l2_norm_batch(
                            input_batch_d,
                            num_samples,
                            sample_size,
                            *buffer.device_ptr_mut() as *mut f64,
                            std::ptr::null_mut(), // default stream
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(format!(
                            "Norm reduction kernel failed with CUDA error code: {} ({})",
                            ret,
                            cuda_error_to_string(ret)
                        )));
                    }

                    buffer
                };

                {
                    crate::profile_scope!("GPU::NormValidation");
                    let host_inv_norms =
                        self.device.dtoh_sync_copy(&inv_norms_gpu).map_err(|e| {
                            MahoutError::Cuda(format!("Failed to copy norms to host: {:?}", e))
                        })?;

                    if host_inv_norms.iter().any(|v| !v.is_finite() || *v == 0.0) {
                        return Err(MahoutError::InvalidInput(
                            "One or more samples have zero or invalid norm".to_string(),
                        ));
                    }
                }

                {
                    crate::profile_scope!("GPU::BatchKernelLaunch");
                    use cudarc::driver::DevicePtr;

                    let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
                        MahoutError::InvalidInput(
                            "Batch state vector precision mismatch (expected float64 buffer)"
                                .to_string(),
                        )
                    })?;

                    let ret = unsafe {
                        qdp_kernels::launch_amplitude_encode_batch(
                            input_batch_d,
                            state_ptr as *mut std::ffi::c_void,
                            *inv_norms_gpu.device_ptr() as *const f64,
                            num_samples,
                            sample_size,
                            state_len,
                            std::ptr::null_mut(), // default stream
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(format!(
                            "Batch kernel launch failed with CUDA error code: {} ({})",
                            ret,
                            cuda_error_to_string(ret)
                        )));
                    }
                }

                {
                    crate::profile_scope!("GPU::Synchronize");
                    self.device
                        .synchronize()
                        .map_err(|e| MahoutError::Cuda(format!("Sync failed: {:?}", e)))?;
                }

                let batch_state_vector =
                    batch_state_vector.to_precision(&self.device, self.precision)?;
                Ok(batch_state_vector.to_dlpack())
            }
            "angle" => {
                use cudarc::driver::DevicePtrMut;

                if sample_size != num_qubits {
                    return Err(MahoutError::InvalidInput(format!(
                        "Angle encoding expects sample_size={} (one angle per qubit), got {}",
                        num_qubits, sample_size
                    )));
                }

                // Validate that all input angles are finite (no NaN/Inf), consistent with
                // CPU and host-side batch angle encoding paths.
                let angle_validation_buffer = {
                    crate::profile_scope!("GPU::AngleFiniteCheckBatch");

                    let mut buffer = self.device.alloc_zeros::<f64>(num_samples).map_err(|e| {
                        MahoutError::MemoryAllocation(format!(
                            "Failed to allocate angle validation buffer: {:?}",
                            e
                        ))
                    })?;

                    let ret = unsafe {
                        qdp_kernels::launch_l2_norm_batch(
                            input_batch_d,
                            num_samples,
                            sample_size,
                            *buffer.device_ptr_mut() as *mut f64,
                            std::ptr::null_mut(), // default stream
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(format!(
                            "Angle validation norm kernel failed with CUDA error code: {} ({})",
                            ret,
                            cuda_error_to_string(ret)
                        )));
                    }

                    buffer
                };

                {
                    crate::profile_scope!("GPU::AngleFiniteValidationHostCopy");
                    let host_norms = self
                        .device
                        .dtoh_sync_copy(&angle_validation_buffer)
                        .map_err(|e| {
                            MahoutError::Cuda(format!(
                                "Failed to copy angle validation norms to host: {:?}",
                                e
                            ))
                        })?;

                    if host_norms.iter().any(|v| !v.is_finite()) {
                        return Err(MahoutError::InvalidInput(
                            "Angle encoding batch contains non-finite values (NaN or Inf)"
                                .to_string(),
                        ));
                    }
                }

                let batch_state_vector = {
                    crate::profile_scope!("GPU::AllocBatch");
                    gpu::GpuStateVector::new_batch(&self.device, num_samples, num_qubits)?
                };

                let state_ptr = batch_state_vector.ptr_f64().ok_or_else(|| {
                    MahoutError::InvalidInput(
                        "Batch state vector precision mismatch (expected float64 buffer)"
                            .to_string(),
                    )
                })?;

                {
                    crate::profile_scope!("GPU::BatchKernelLaunch");
                    let ret = unsafe {
                        qdp_kernels::launch_angle_encode_batch(
                            input_batch_d,
                            state_ptr as *mut std::ffi::c_void,
                            num_samples,
                            state_len,
                            num_qubits as u32,
                            std::ptr::null_mut(), // default stream
                        )
                    };

                    if ret != 0 {
                        return Err(MahoutError::KernelLaunch(format!(
                            "Batch angle encoding kernel failed: {} ({})",
                            ret,
                            cuda_error_to_string(ret)
                        )));
                    }
                }

                {
                    crate::profile_scope!("GPU::Synchronize");
                    self.device
                        .synchronize()
                        .map_err(|e| MahoutError::Cuda(format!("Sync failed: {:?}", e)))?;
                }

                let batch_state_vector =
                    batch_state_vector.to_precision(&self.device, self.precision)?;
                Ok(batch_state_vector.to_dlpack())
            }
            _ => Err(MahoutError::NotImplemented(format!(
                "GPU pointer batch encoding currently only supports 'amplitude' and 'angle' methods, got '{}'",
                encoding_method
            ))),
        }
    }
}

// Re-export key types for convenience
pub use gpu::QuantumEncoder;
