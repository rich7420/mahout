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

// Amplitude encoding: direct state injection with L2 normalization
// Now uses Batch Kernel for maximum performance

use std::sync::Arc;
use cudarc::driver::{CudaDevice, DevicePtr};
use std::ffi::c_void;
use crate::error::{MahoutError, Result};
use crate::gpu::memory::GpuStateVector;
use crate::gpu::pool::StagingBufferPool;
use crate::types::QuantumFloat;
use super::{QuantumEncoder, QuantumEncoderLegacy};

#[cfg(target_os = "linux")]
use crate::gpu::pipeline::run_dual_stream_pipeline;

/// Amplitude encoding: data → normalized quantum amplitudes
///
/// Steps: GPU allocation → Host->Device copy → Batch Kernel (normalize + pad)
/// Fast: ~50-100x vs circuit-based methods
/// Now uses Batch Kernel for maximum throughput
pub struct AmplitudeEncoder;

// New generic implementation using Batch Kernel
impl<T: QuantumFloat> QuantumEncoder<T> for AmplitudeEncoder {
    fn encode(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        host_data: &[T],
        num_qubits: usize,
    ) -> Result<GpuStateVector<T>> {
        // 1. Validation & Pre-calc (CPU)
        if host_data.is_empty() {
            return Err(MahoutError::InvalidInput("Empty input".into()));
        }

        // For backward compatibility with single vector input, we treat as Batch=1
        let batch_size = 1;
        let input_dim = host_data.len();
        let state_dim = 1 << num_qubits;
        let total_elements = std::cmp::max(host_data.len(), state_dim);

        #[cfg(target_os = "linux")]
        {
            // 1. Allocate Output (State Vector) - This is for the user, cannot use Pool
            let state_vector = {
                crate::profile_scope!("GPU::Alloc");
                GpuStateVector::new_with_capacity(device, num_qubits, total_elements)?
            };

            // 2. Threshold Check
            // 1MB threshold for generic type
            const ASYNC_THRESHOLD_BYTES: usize = 1024 * 1024;
            let input_bytes = host_data.len() * std::mem::size_of::<T>();

            if input_bytes < ASYNC_THRESHOLD_BYTES {
                // === Synchronous Path (Small Data) ===
                let staging_buffer_u8 = {
                    crate::profile_scope!("Pool::Acquire");
                    pool.acquire(input_bytes)?
                };

                let input_dev_ptr = *staging_buffer_u8.device_ptr() as *mut T;

                {
                    crate::profile_scope!("GPU::H2DCopy");
                    unsafe {
                        unsafe extern "C" {
                            fn cudaMemcpy(
                                dst: *mut c_void,
                                src: *const c_void,
                                count: usize,
                                kind: u32,
                            ) -> i32;
                        }
                        const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;
                        let result = cudaMemcpy(
                            input_dev_ptr as *mut c_void,
                            host_data.as_ptr() as *const c_void,
                            input_bytes,
                            CUDA_MEMCPY_HOST_TO_DEVICE,
                        );
                        if result != 0 {
                            pool.release(staging_buffer_u8);
                            return Err(MahoutError::Cuda(format!("H2D copy failed: {}", result)));
                        }
                    }
                }

                unsafe {
                    crate::profile_scope!("GPU::KernelLaunch");
                    let ret = T::launch_batch_encode(
                        input_dev_ptr as *const T,
                        state_vector.ptr() as *mut c_void,
                        std::ptr::null(), // norms
                        batch_size,
                        input_dim,
                        state_dim,
                        std::ptr::null_mut()
                    );

                    if ret != 0 {
                        pool.release(staging_buffer_u8);
                        return Err(MahoutError::KernelLaunch(format!("CUDA Error: {}", ret)));
                    }
                }

                device.synchronize().map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
                pool.release(staging_buffer_u8);
                Ok(state_vector)

            } else {
                // === Async Path (Large Data) ===
                // Call generic pipeline
                Self::encode_async_pipeline(device, host_data, num_qubits, total_elements, &state_vector)?;
                Ok(state_vector)
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda("CUDA unavailable (non-Linux)".to_string()))
        }
    }

    fn name(&self) -> &'static str {
        "amplitude"
    }

    fn description(&self) -> &'static str {
        "Generic Amplitude Encoding with Batch Kernel"
    }
}

impl AmplitudeEncoder {
    #[cfg(target_os = "linux")]
    fn encode_async_pipeline<T: QuantumFloat>(
        device: &Arc<CudaDevice>,
        host_data: &[T],
        _num_qubits: usize,
        state_len: usize,
        state_vector: &GpuStateVector<T>,
    ) -> Result<()> {
        // Use generic run_dual_stream_pipeline
        run_dual_stream_pipeline(device, host_data, |stream, input_ptr, chunk_offset, chunk_len| {
            let state_ptr_offset = unsafe {
                state_vector.ptr().cast::<u8>()
                    .add(chunk_offset * std::mem::size_of::<T::Complex>())
                    .cast::<std::ffi::c_void>()
            };

            let ret = unsafe {
                // Use generic Batch Kernel
                T::launch_batch_encode(
                    input_ptr,
                    state_ptr_offset,
                    std::ptr::null(), // norms (not supported in pipelined chunks currently)
                    1, // batch_size inside this chunk (viewed as flat)
                    chunk_len, // input_dim
                    chunk_len, // state_dim (for this chunk)
                    stream.stream as *mut c_void,
                )
            };

            if ret != 0 {
                return Err(MahoutError::KernelLaunch(format!("Pipeline kernel failed: {}", ret)));
            }

            Ok(())
        })?;

        // Padding logic (T-aware)
        let data_len = host_data.len();
        if data_len < state_len {
            let padding_start = data_len;
            let padding_elements = state_len - padding_start;
            let padding_bytes = padding_elements * std::mem::size_of::<T::Complex>();

            let tail_ptr = unsafe {
                state_vector.ptr().add(padding_start) as *mut c_void
            };

            unsafe {
                unsafe extern "C" {
                    fn cudaMemsetAsync(
                        devPtr: *mut c_void,
                        val: i32,
                        count: usize,
                        stream: *mut c_void,
                    ) -> i32;
                }
                let res = cudaMemsetAsync(tail_ptr, 0, padding_bytes, std::ptr::null_mut());
                if res != 0 {
                    return Err(MahoutError::Cuda(format!("Padding failed: {}", res)));
                }
            }
            device.synchronize().map_err(|e| MahoutError::Cuda(format!("{:?}", e)))?;
        }

        Ok(())
    }
}

// Legacy implementation for backward compatibility
impl QuantumEncoderLegacy for AmplitudeEncoder {
    fn encode(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        host_data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector<f64>> {
        // Delegate to generic implementation
        <Self as QuantumEncoder<f64>>::encode(self, device, pool, host_data, num_qubits)
    }

    fn name(&self) -> &'static str {
        "amplitude"
    }

    fn description(&self) -> &'static str {
        "Amplitude encoding with Batch Kernel"
    }
}
