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

//! GPU Memory Pool (Stack Allocator & Staging Buffer Pool)
//!
//! Provides two allocation strategies:
//! 1. Stack Allocator: Pre-allocates a large GPU buffer and sub-allocates from it
//! 2. Staging Buffer Pool: Reuses temporary buffers for H2D copies

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use crate::error::{MahoutError, Result};

/// Stack-based GPU memory allocator
///
/// Pre-allocates a large buffer (e.g., 4GB) and manages sub-allocations
/// from it. This eliminates cudaMalloc overhead for frequent allocations.
pub struct GpuStackAllocator {
    device: Arc<CudaDevice>,
    pool: Mutex<PoolState>,
    pool_size_bytes: usize,
}

struct PoolState {
    buffer: Option<CudaSlice<u8>>,
    offset: usize,
    allocations: Vec<(usize, usize)>, // (offset, size) for tracking
}

impl GpuStackAllocator {
    /// Create a new stack allocator
    ///
    /// # Arguments
    /// * `device` - CUDA device
    /// * `pool_size_bytes` - Size of pre-allocated pool in bytes (default: 4GB)
    pub fn new(device: Arc<CudaDevice>, pool_size_bytes: usize) -> Result<Self> {
        let pool = Mutex::new(PoolState {
            buffer: None,
            offset: 0,
            allocations: Vec::new(),
        });

        Ok(Self {
            device,
            pool,
            pool_size_bytes,
        })
    }

    /// Initialize the pool by allocating the large buffer
    ///
    /// This should be called once during engine initialization.
    pub fn initialize(&self) -> Result<()> {
        let mut state = self.pool.lock().unwrap();

        if state.buffer.is_some() {
            return Ok(()); // Already initialized
        }

        let buffer = unsafe {
            self.device.alloc::<u8>(self.pool_size_bytes)
        }.map_err(|e| {
            MahoutError::MemoryAllocation(format!(
                "Failed to allocate {} bytes for GPU stack pool: {:?}",
                self.pool_size_bytes, e
            ))
        })?;

        state.buffer = Some(buffer);
        state.offset = 0;
        Ok(())
    }

    /// Allocate a slice from the pool
    ///
    /// Returns None if the pool is exhausted. The caller should fall back
    /// to regular allocation in this case.
    pub fn allocate<T>(&self, num_elements: usize) -> Result<Option<CudaSlice<T>>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        let size_bytes = num_elements * std::mem::size_of::<T>();
        let mut state = self.pool.lock().unwrap();

        let _buffer = match &state.buffer {
            Some(_b) => _b,
            None => return Ok(None), // Pool not initialized
        };

        // Check if we have enough space
        if state.offset + size_bytes > self.pool_size_bytes {
            return Ok(None); // Pool exhausted, fall back to regular alloc
        }

        // Calculate aligned offset (for type alignment)
        let align = std::mem::align_of::<T>();
        let aligned_offset = (state.offset + align - 1) & !(align - 1);

        if aligned_offset + size_bytes > self.pool_size_bytes {
            return Ok(None);
        }

        // Create a slice from the pool
        // Note: This is unsafe because we're creating a slice from raw bytes
        // In production, we'd need proper type conversion
        // For now, we return None and fall back to regular allocation
        // TODO: Implement proper sub-slicing from CudaSlice<u8>

        state.offset = aligned_offset + size_bytes;
        state.allocations.push((aligned_offset, size_bytes));

        // For now, return None to indicate we should use regular allocation
        // This is a placeholder - proper implementation requires unsafe pointer arithmetic
        Ok(None)
    }

    /// Reset the pool (mark all allocations as free)
    ///
    /// This should be called at the end of each batch or epoch.
    pub fn reset(&self) {
        let mut state = self.pool.lock().unwrap();
        state.offset = 0;
        state.allocations.clear();
    }

    /// Get current pool usage statistics
    pub fn stats(&self) -> (usize, usize) {
        let state = self.pool.lock().unwrap();
        (state.offset, self.pool_size_bytes)
    }
}

impl Drop for GpuStackAllocator {
    fn drop(&mut self) {
        // Buffer will be automatically freed when dropped
        let mut state = self.pool.lock().unwrap();
        state.buffer = None;
        state.offset = 0;
        state.allocations.clear();
    }
}

/// Staging Buffer Pool
///
/// Caches temporary memory (Input Buffer) required for Host -> Device copies.
/// This eliminates frequent cudaMalloc/cudaFree overhead.
pub struct StagingBufferPool {
    device: Arc<CudaDevice>,
    // Simple Free List: (capacity_bytes, buffer)
    // We use u8 to store generic bytes, cast to specific types when needed
    pool: Mutex<VecDeque<CudaSlice<u8>>>,
}

impl StagingBufferPool {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            pool: Mutex::new(VecDeque::new()),
        }
    }

    /// Acquire a buffer with at least `size_bytes` capacity
    pub fn acquire(&self, size_bytes: usize) -> Result<CudaSlice<u8>> {
        let mut pool = self.pool.lock().unwrap();

        // 1. Try to find a large enough idle buffer (Best Fit or First Fit)
        // For performance, we only use First Fit
        if let Some(idx) = pool.iter().position(|slice| slice.len() >= size_bytes) {
            // Found one! Remove and return
            return Ok(pool.remove(idx).unwrap());
        }

        // 2. Not found, allocate new (this is a slow operation, but will disappear after warm-up)
        // Allocate slightly larger (e.g., align to 1MB) to increase future reuse rate
        let alloc_size = size_bytes.max(1024 * 1024); // Minimum 1MB

        unsafe {
            self.device.alloc::<u8>(alloc_size)
        }.map_err(|e| MahoutError::MemoryAllocation(format!("Staging alloc failed: {:?}", e)))
    }

    /// Return buffer to pool
    pub fn release(&self, buffer: CudaSlice<u8>) {
        let mut pool = self.pool.lock().unwrap();
        // Simple strategy: only keep the most recent 8 buffers to avoid unbounded growth
        if pool.len() < 8 {
            pool.push_back(buffer);
        }
        // Otherwise, buffer will be dropped (cudaFree) when leaving scope
    }
}
