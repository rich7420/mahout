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

//! Reusable pool of GPU device memory buffers for state vector allocation.
//!
//! Provides a caching allocator that avoids expensive `cudaMalloc`/`cudaFree`
//! calls on every encode operation. Buffers are keyed by `(size_elements, precision)`
//! and automatically returned to the pool when dropped.
//!
//! This mirrors the design of PyTorch's CUDA caching allocator at a smaller scale,
//! targeting the specific allocation patterns of quantum state encoding.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaDevice;

use crate::error::{MahoutError, Result};
use crate::gpu::memory::{BufferStorage, GpuBufferRaw, Precision};
use qdp_kernels::{CuComplex, CuDoubleComplex};

#[cfg(target_os = "linux")]
use crate::gpu::memory::ensure_device_memory_available;

/// Key for looking up cached buffers: (element count, precision).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PoolKey {
    size_elements: usize,
    precision: Precision,
}

/// Inner free-list storage. Each key maps to a stack of available buffers.
type FreeList = HashMap<PoolKey, Vec<BufferStorage>>;

/// Pool of reusable GPU device memory buffers.
///
/// Avoids repeated `cudaMalloc`/`cudaFree` overhead by caching freed buffers
/// and reusing them for subsequent allocations of the same size and precision.
///
/// # Thread Safety
/// The pool uses a `Mutex` for thread safety. All operations are O(1) amortized
/// since free-list lookups are hash-based and buffer reuse is stack-based.
pub struct DeviceBufferPool {
    free: Mutex<FreeList>,
    device: Arc<CudaDevice>,
    /// Maximum number of cached buffers across all keys.
    /// Prevents unbounded memory growth.
    max_cached: usize,
}

impl DeviceBufferPool {
    /// Create a new device buffer pool.
    ///
    /// # Arguments
    /// * `device` - CUDA device for allocations
    /// * `max_cached` - Maximum total number of cached buffers (0 = unlimited)
    pub fn new(device: Arc<CudaDevice>, max_cached: usize) -> Self {
        Self {
            free: Mutex::new(HashMap::new()),
            device,
            max_cached,
        }
    }

    /// Total number of cached (free) buffers across all keys.
    pub fn cached_count(&self) -> usize {
        let free = self.free.lock().unwrap_or_else(|p| p.into_inner());
        free.values().map(|v| v.len()).sum()
    }

    /// Acquire a buffer of the given size and precision.
    ///
    /// Returns a cached buffer if one is available, otherwise allocates a new one.
    /// This avoids `cudaMalloc` when a matching buffer exists in the pool.
    #[cfg(target_os = "linux")]
    pub fn acquire(&self, size_elements: usize, precision: Precision) -> Result<BufferStorage> {
        let key = PoolKey {
            size_elements,
            precision,
        };

        // Try to reuse a cached buffer
        {
            let mut free = self.free.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(buffers) = free.get_mut(&key)
                && let Some(buffer) = buffers.pop()
            {
                return Ok(buffer);
            }
        }

        // Cache miss: allocate a new buffer
        self.alloc_new(size_elements, precision)
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// If the pool has reached its maximum cached count, the buffer is dropped
    /// (triggering `cudaFree`) instead of being cached.
    pub fn release(&self, buffer: BufferStorage, size_elements: usize) {
        let key = PoolKey {
            size_elements,
            precision: buffer.precision(),
        };

        let mut free = self.free.lock().unwrap_or_else(|p| p.into_inner());

        // Check if we're at max capacity
        if self.max_cached > 0 {
            let total: usize = free.values().map(|v| v.len()).sum();
            if total >= self.max_cached {
                // Drop the buffer (cudaFree) instead of caching
                return;
            }
        }

        free.entry(key).or_default().push(buffer);
    }

    /// Clear all cached buffers, freeing GPU memory.
    pub fn clear(&self) {
        let mut free = self.free.lock().unwrap_or_else(|p| p.into_inner());
        free.clear();
    }

    /// Allocate a new buffer from the GPU.
    #[cfg(target_os = "linux")]
    fn alloc_new(&self, size_elements: usize, precision: Precision) -> Result<BufferStorage> {
        match precision {
            Precision::Float32 => {
                let requested_bytes = size_elements
                    .checked_mul(std::mem::size_of::<CuComplex>())
                    .ok_or_else(|| {
                        MahoutError::MemoryAllocation(format!(
                            "Requested GPU allocation size overflow (elements={})",
                            size_elements
                        ))
                    })?;

                ensure_device_memory_available(
                    requested_bytes,
                    "pooled state vector allocation (f32)",
                    None,
                )?;

                let slice =
                    unsafe { self.device.alloc::<CuComplex>(size_elements) }.map_err(|e| {
                        crate::gpu::memory::map_allocation_error(
                            requested_bytes,
                            "pooled state vector allocation (f32)",
                            None,
                            e,
                        )
                    })?;

                Ok(BufferStorage::F32(GpuBufferRaw { slice }))
            }
            Precision::Float64 => {
                let requested_bytes = size_elements
                    .checked_mul(std::mem::size_of::<CuDoubleComplex>())
                    .ok_or_else(|| {
                        MahoutError::MemoryAllocation(format!(
                            "Requested GPU allocation size overflow (elements={})",
                            size_elements
                        ))
                    })?;

                ensure_device_memory_available(
                    requested_bytes,
                    "pooled state vector allocation (f64)",
                    None,
                )?;

                let slice = unsafe { self.device.alloc::<CuDoubleComplex>(size_elements) }
                    .map_err(|e| {
                        crate::gpu::memory::map_allocation_error(
                            requested_bytes,
                            "pooled state vector allocation (f64)",
                            None,
                            e,
                        )
                    })?;

                Ok(BufferStorage::F64(GpuBufferRaw { slice }))
            }
        }
    }
}
