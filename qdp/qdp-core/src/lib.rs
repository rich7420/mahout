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

pub mod dlpack;
pub mod gpu;
pub mod error;
pub mod preprocessing;
pub mod io;
pub mod types;

#[macro_use]
mod profiling;

pub use error::{MahoutError, Result};

use std::sync::Arc;
use std::marker::PhantomData;
use arrow::array::Float64Array;
use cudarc::driver::CudaDevice;
use crate::dlpack::DLManagedTensor;
use crate::gpu::get_encoder_generic;
use crate::gpu::pool::StagingBufferPool;
use crate::types::QuantumFloat;

/// Main entry point for Mahout QDP (Generic version)
///
/// Manages GPU context and dispatches encoding tasks.
/// Provides unified interface for device management, memory allocation, and DLPack.
///
/// The generic parameter T specifies the floating-point precision (f32 or f64).
pub struct QdpEngine<T: QuantumFloat> {
    device: Arc<CudaDevice>,
    staging_pool: Arc<StagingBufferPool>,
    _marker: PhantomData<T>,
}

impl<T: QuantumFloat> QdpEngine<T> {
    /// Initialize engine on GPU device
    ///
    /// # Arguments
    /// * `device_id` - CUDA device ID (typically 0)
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| MahoutError::Cuda(format!("Failed to initialize CUDA device {}: {:?}", device_id, e)))?;
        let staging_pool = Arc::new(StagingBufferPool::new(device.clone()));
        Ok(Self {
            device,  // CudaDevice::new already returns Arc<CudaDevice> in cudarc 0.11
            staging_pool,
            _marker: PhantomData,
        })
    }

    /// Get CUDA device reference for advanced operations
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Access the staging buffer pool
    ///
    /// Returns a reference to the persistent memory pool used for H2D transfers.
    /// This pool is shared across all encoding operations to avoid repeated allocations.
    pub fn pool(&self) -> &Arc<StagingBufferPool> {
        &self.staging_pool
    }

    // === Generic Encoding Methods ===

    /// Encode from raw slice (already in correct type T)
    pub fn encode(
        &self,
        data: &[T],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::Encode");

        let encoder = get_encoder_generic::<T>(encoding_method)?;
        let state_vector = encoder.encode(&self.device, &self.staging_pool, data, num_qubits)?;
        let dlpack_ptr = {
            crate::profile_scope!("DLPack::Wrap");
            state_vector.to_dlpack()
        };
        Ok(dlpack_ptr)
    }

    /// Encode from Arrow Float64 chunks (handles casting to T if needed)
    pub fn encode_chunked(
        &self,
        chunks: &[Float64Array],
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeChunked");

        let encoder = get_encoder_generic::<T>(encoding_method)?;
        let state_vector = encoder.encode_chunked(&self.device, &self.staging_pool, chunks, num_qubits)?;
        let dlpack_ptr = {
            crate::profile_scope!("Mahout::CreateDLPack");
            state_vector.to_dlpack()
        };
        Ok(dlpack_ptr)
    }

    /// Encode from Parquet file path
    pub fn encode_from_parquet(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        crate::profile_scope!("Mahout::EncodeFromParquet");

        // Read Parquet as Float64 chunks (standard arrow behavior)
        let chunks = match crate::io::read_parquet_list_to_arrow(path) {
            Ok(chunks) => chunks,
            Err(_) => {
                crate::io::read_parquet_to_arrow(path)?
            }
        };

        // Delegate to chunked encoder
        self.encode_chunked(&chunks, num_qubits, encoding_method)
    }
}

// Re-export key types for convenience
pub use gpu::QuantumEncoder;
