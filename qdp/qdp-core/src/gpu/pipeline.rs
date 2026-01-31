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

// Async Pipeline Infrastructure
//
// Provides generic double-buffered execution for large data processing.
// Separates the "streaming mechanics" from the "kernel logic".

// Allow unused_unsafe: CUDA FFI functions are unsafe in CUDA builds but safe stubs in no-CUDA builds.
// The compiler can't statically determine which path is taken.
#![allow(unused_unsafe)]

use crate::error::{MahoutError, Result};
#[cfg(target_os = "linux")]
use crate::gpu::cuda_ffi::{
    CUDA_ERROR_NOT_READY, CUDA_EVENT_DISABLE_TIMING, CUDA_MEMCPY_HOST_TO_DEVICE, CUDA_SUCCESS,
    cudaEventCreateWithFlags, cudaEventDestroy, cudaEventRecord, cudaEventSynchronize,
    cudaMemcpyAsync, cudaStreamQuery, cudaStreamWaitEvent,
};
#[cfg(target_os = "linux")]
use crate::gpu::memory::{PinnedHostBuffer, ensure_device_memory_available, map_allocation_error};
#[cfg(target_os = "linux")]
use crate::gpu::overlap_tracker::OverlapTracker;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, safe::CudaStream};
use std::ffi::c_void;
use std::sync::Arc;

/// Dual-stream context coordinating copy/compute with events.
/// Fix 3: events_compute_done lets the copy stream wait before reusing a device buffer.
/// Fix 4 (Token Ring): synchronize_copy_event lets the host wait only one slot's copy before reusing its pinned buffer.
#[cfg(target_os = "linux")]
pub struct PipelineContext {
    pub stream_compute: CudaStream,
    pub stream_copy: CudaStream,
    events_copy_done: Vec<*mut c_void>,
    events_compute_done: Vec<*mut c_void>,
}

#[cfg(target_os = "linux")]
fn validate_event_slot(events: &[*mut c_void], slot: usize) -> Result<()> {
    if slot >= events.len() {
        return Err(MahoutError::InvalidInput(format!(
            "Event slot {} out of range (max: {})",
            slot,
            events.len().saturating_sub(1)
        )));
    }
    Ok(())
}

/// N.6: Sync stream via cudaStreamQuery loop + yield instead of blocking cudaStreamSynchronize.
/// Ref: CUDA §2.3.2.4 Stream Synchronization (non-blocking approach).
#[cfg(target_os = "linux")]
fn sync_stream_via_query(stream: *mut c_void) -> Result<()> {
    loop {
        let ret = unsafe { cudaStreamQuery(stream) };
        if ret == CUDA_SUCCESS {
            break Ok(());
        }
        if ret != CUDA_ERROR_NOT_READY {
            return Err(MahoutError::Cuda(format!(
                "cudaStreamQuery failed: {}",
                ret
            )));
        }
        std::thread::yield_now();
    }
}

#[cfg(target_os = "linux")]
impl PipelineContext {
    pub fn new(device: &Arc<CudaDevice>, event_slots: usize) -> Result<Self> {
        let stream_compute = device
            .fork_default_stream()
            .map_err(|e| MahoutError::Cuda(format!("Failed to create compute stream: {:?}", e)))?;
        let stream_copy = device
            .fork_default_stream()
            .map_err(|e| MahoutError::Cuda(format!("Failed to create copy stream: {:?}", e)))?;

        let mut events_copy_done = Vec::with_capacity(event_slots);
        let mut events_compute_done = Vec::with_capacity(event_slots);
        for _ in 0..event_slots {
            let mut ev: *mut c_void = std::ptr::null_mut();
            unsafe {
                let ret = cudaEventCreateWithFlags(&mut ev, CUDA_EVENT_DISABLE_TIMING);
                if ret != 0 {
                    return Err(MahoutError::Cuda(format!(
                        "Failed to create CUDA copy event: {}",
                        ret
                    )));
                }
            }
            events_copy_done.push(ev);

            let mut ev_compute: *mut c_void = std::ptr::null_mut();
            unsafe {
                let ret = cudaEventCreateWithFlags(&mut ev_compute, CUDA_EVENT_DISABLE_TIMING);
                if ret != 0 {
                    return Err(MahoutError::Cuda(format!(
                        "Failed to create CUDA compute event: {}",
                        ret
                    )));
                }
            }
            events_compute_done.push(ev_compute);
        }

        Ok(Self {
            stream_compute,
            stream_copy,
            events_copy_done,
            events_compute_done,
        })
    }

    /// Async H2D copy on the copy stream.
    ///
    /// # Safety
    /// `src` must be valid for `len_elements` `f64` values and properly aligned.
    /// `dst` must point to device memory for `len_elements` `f64` values on the same device.
    /// Both pointers must remain valid until the copy completes on `stream_copy`.
    pub unsafe fn async_copy_to_device(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        len_elements: usize,
    ) -> Result<()> {
        crate::profile_scope!("GPU::H2D_Copy");
        unsafe {
            let ret = cudaMemcpyAsync(
                dst,
                src,
                len_elements * std::mem::size_of::<f64>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream_copy.stream as *mut c_void,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "Async H2D copy failed with CUDA error: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Record completion of the copy on the copy stream.
    ///
    /// # Safety
    /// `slot` must refer to a live event created by this context, and the context must
    /// remain alive until the event is no longer used by any stream.
    pub unsafe fn record_copy_done(&self, slot: usize) -> Result<()> {
        crate::profile_scope!("GPU::CopyEventRecord");
        validate_event_slot(&self.events_copy_done, slot)?;

        unsafe {
            let ret = cudaEventRecord(
                self.events_copy_done[slot],
                self.stream_copy.stream as *mut c_void,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventRecord failed: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Make compute stream wait for the copy completion event.
    ///
    /// # Safety
    /// `slot` must refer to a live event previously recorded on `stream_copy`, and the
    /// context and its streams must remain valid while waiting.
    pub unsafe fn wait_for_copy(&self, slot: usize) -> Result<()> {
        crate::profile_scope!("GPU::StreamWait");
        validate_event_slot(&self.events_copy_done, slot)?;

        unsafe {
            let ret = cudaStreamWaitEvent(
                self.stream_compute.stream as *mut c_void,
                self.events_copy_done[slot],
                0,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaStreamWaitEvent failed: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Record completion of the kernel on the compute stream (Fix 3: for device buffer reuse).
    ///
    /// Copy stream will wait on this event before reusing the same device buffer slot.
    /// Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    ///
    /// # Safety
    /// `slot` must refer to a live event created by this context; kernel for this slot
    /// must have been launched on `stream_compute` before calling.
    pub unsafe fn record_compute_done(&self, slot: usize) -> Result<()> {
        crate::profile_scope!("GPU::ComputeEventRecord");
        validate_event_slot(&self.events_compute_done, slot)?;

        unsafe {
            let ret = cudaEventRecord(
                self.events_compute_done[slot],
                self.stream_compute.stream as *mut c_void,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventRecord(compute_done) failed: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Make copy stream wait for the compute completion event (Fix 3: before reusing device buffer).
    ///
    /// Prevents overwriting device memory still in use by a kernel on the compute stream.
    /// Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
    ///
    /// # Safety
    /// `slot` must refer to a live event previously recorded on `stream_compute`.
    pub unsafe fn wait_for_compute(&self, slot: usize) -> Result<()> {
        crate::profile_scope!("GPU::CopyWaitCompute");
        validate_event_slot(&self.events_compute_done, slot)?;

        unsafe {
            let ret = cudaStreamWaitEvent(
                self.stream_copy.stream as *mut c_void,
                self.events_compute_done[slot],
                0,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaStreamWaitEvent(compute_done) failed: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Host waits for the copy-complete event on this slot (Fix 4: Token Ring).
    ///
    /// Before reusing the pinned buffer for `slot`, the host waits only for that slot's
    /// previous H2D copy to complete. This replaces full-stream sync and preserves overlap.
    /// Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    ///
    /// # Safety
    /// `slot` must refer to a live event previously recorded on `stream_copy`.
    pub unsafe fn synchronize_copy_event(&self, slot: usize) -> Result<()> {
        crate::profile_scope!("Pipeline::SyncCopyEvent");
        validate_event_slot(&self.events_copy_done, slot)?;
        unsafe {
            let ret = cudaEventSynchronize(self.events_copy_done[slot]);
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaEventSynchronize(copy_done[{}]) failed: {}",
                    slot, ret
                )));
            }
        }
        Ok(())
    }

    /// Make copy stream wait for the previous copy on this slot (event-driven pinned buffer reuse).
    ///
    /// Before reusing the pinned buffer for `slot`, the copy stream waits for the previous H2D
    /// copy that used this slot to complete. This avoids host sync and preserves copy-compute overlap.
    /// Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9
    ///
    /// # Safety
    /// `slot` must refer to a live event previously recorded on `stream_copy`.
    pub unsafe fn wait_for_previous_copy_on_copy_stream(&self, slot: usize) -> Result<()> {
        crate::profile_scope!("GPU::CopyWaitPreviousCopy");
        validate_event_slot(&self.events_copy_done, slot)?;

        unsafe {
            let ret = cudaStreamWaitEvent(
                self.stream_copy.stream as *mut c_void,
                self.events_copy_done[slot],
                0,
            );
            if ret != 0 {
                return Err(MahoutError::Cuda(format!(
                    "cudaStreamWaitEvent(copy_done on copy stream) failed: {}",
                    ret
                )));
            }
        }
        Ok(())
    }

    /// Sync copy stream (safe to reuse host buffer).
    /// N.6: Uses cudaStreamQuery in a short loop + yield instead of blocking cudaStreamSynchronize.
    ///
    /// # Safety
    /// The context and its copy stream must be valid and not destroyed while syncing.
    pub unsafe fn sync_copy_stream(&self) -> Result<()> {
        crate::profile_scope!("Pipeline::SyncCopy");
        sync_stream_via_query(self.stream_copy.stream as *mut c_void)
    }

    /// Sync compute stream via non-blocking query loop (N.6).
    /// Call after all chunks are enqueued; yields while stream is busy instead of blocking in driver.
    ///
    /// # Safety
    /// The context and its compute stream must be valid and not destroyed while syncing.
    pub unsafe fn sync_compute_stream_via_query(&self) -> Result<()> {
        crate::profile_scope!("Pipeline::SyncCompute");
        sync_stream_via_query(self.stream_compute.stream as *mut c_void)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::validate_event_slot;

    #[test]
    fn validate_event_slot_allows_in_range() {
        let events = vec![std::ptr::null_mut(); 2];
        assert!(validate_event_slot(&events, 0).is_ok());
        assert!(validate_event_slot(&events, 1).is_ok());
    }

    #[test]
    fn validate_event_slot_rejects_out_of_range() {
        let events = vec![std::ptr::null_mut(); 2];
        let err = validate_event_slot(&events, 2).unwrap_err();
        assert!(matches!(err, crate::error::MahoutError::InvalidInput(_)));
    }
}

#[cfg(target_os = "linux")]
impl Drop for PipelineContext {
    fn drop(&mut self) {
        unsafe {
            for ev in &mut self.events_copy_done {
                if !ev.is_null() {
                    let _ = cudaEventDestroy(*ev);
                }
            }
            for ev in &mut self.events_compute_done {
                if !ev.is_null() {
                    let _ = cudaEventDestroy(*ev);
                }
            }
        }
    }
}

/// Executes a task using dual-stream double-buffering pattern.
///
/// **Config flow (performance-first):** Chunk size and pinned pool size are resolved from
/// `PipelineConfig` (env + hardware defaults), validated (e.g. pinned ≤ 20% host), then passed
/// into the internal pipeline. This maximizes H2D overlap and reduces copy-stream stalls.
///
/// This function handles the generic pipeline mechanics:
/// - Dual stream creation and management
/// - Data chunking and async H2D copy
/// - Buffer lifetime management
/// - Stream synchronization
/// - Optional observability (pool metrics and overlap tracking)
///
/// The caller provides a `kernel_launcher` closure that handles the
/// specific kernel launch logic for each chunk.
///
/// # Arguments
/// * `device` - The CUDA device
/// * `host_data` - Full source data to process
/// * `kernel_launcher` - Closure that launches the specific kernel for each chunk
///
/// # Environment Variables
/// * `QDP_ENABLE_POOL_METRICS` - Enable pool utilization metrics (set to "1" or "true")
/// * `QDP_ENABLE_OVERLAP_TRACKING` - Enable H2D overlap tracking (set to "1" or "true")
///
/// # Example
/// ```rust,ignore
/// run_dual_stream_pipeline(device, host_data, |stream, input_ptr, offset, len| {
///     // Launch your specific kernel here
///     launch_my_kernel(input_ptr, offset, len, stream)?;
///     Ok(())
/// })?;
/// ```
#[cfg(target_os = "linux")]
pub fn run_dual_stream_pipeline<F>(
    device: &Arc<CudaDevice>,
    host_data: &[f64],
    kernel_launcher: F,
) -> Result<()>
where
    F: FnMut(&CudaStream, *const f64, usize, usize) -> Result<()>,
{
    crate::profile_scope!("GPU::AsyncPipeline");
    let mut pinned = PinnedHostBuffer::new(host_data.len())?;
    pinned.as_slice_mut()[..host_data.len()].copy_from_slice(host_data);
    let host_mem_gb = crate::gpu::pipeline_config::PipelineConfig::get_host_memory_gb();
    let config = crate::gpu::pipeline_config::PipelineConfig::get_or_build(device, host_mem_gb)?;
    let chunk_size_elements = config.chunk_size_elements()?;
    let pool_size = config.pinned_pool_size_resolved()?;
    run_dual_stream_pipeline_with_chunk_size_from_pinned(
        device,
        pinned.as_slice(),
        chunk_size_elements,
        pool_size,
        kernel_launcher,
    )
}

/// Executes a task using dual-stream double-buffering with aligned chunk boundaries.
///
/// `align_elements` must evenly divide the host data length and ensures chunks do not
/// split logical records (e.g., per-sample data in batch encoding).
#[cfg(target_os = "linux")]
pub fn run_dual_stream_pipeline_aligned<F>(
    device: &Arc<CudaDevice>,
    host_data: &[f64],
    align_elements: usize,
    kernel_launcher: F,
) -> Result<()>
where
    F: FnMut(&CudaStream, *const f64, usize, usize) -> Result<()>,
{
    crate::profile_scope!("GPU::AsyncPipelineAligned");

    if align_elements == 0 {
        return Err(MahoutError::InvalidInput(
            "Alignment must be greater than zero".to_string(),
        ));
    }
    if !host_data.len().is_multiple_of(align_elements) {
        return Err(MahoutError::InvalidInput(format!(
            "Host data length {} is not aligned to {} elements",
            host_data.len(),
            align_elements
        )));
    }
    let mut pinned = PinnedHostBuffer::new(host_data.len())?;
    pinned.as_slice_mut()[..host_data.len()].copy_from_slice(host_data);

    let host_mem_gb = crate::gpu::pipeline_config::PipelineConfig::get_host_memory_gb();
    let config = crate::gpu::pipeline_config::PipelineConfig::get_or_build(device, host_mem_gb)?;
    let base_chunk_elements = config.chunk_size_elements()?;
    let pool_size = config.pinned_pool_size_resolved()?;
    let chunk_size_elements = if align_elements >= base_chunk_elements {
        align_elements
    } else {
        base_chunk_elements - (base_chunk_elements % align_elements)
    };

    run_dual_stream_pipeline_with_chunk_size_from_pinned(
        device,
        pinned.as_slice(),
        chunk_size_elements,
        pool_size,
        kernel_launcher,
    )
}

/// Same as run_dual_stream_pipeline_aligned but source is already pinned: no copy to pinned_slots,
/// only async H2D from pinned_data. Use from batch pool worker after one copy (Vec → pinned).
#[cfg(target_os = "linux")]
pub fn run_dual_stream_pipeline_aligned_from_pinned<F>(
    device: &Arc<CudaDevice>,
    pinned_data: &[f64],
    align_elements: usize,
    kernel_launcher: F,
) -> Result<()>
where
    F: FnMut(&CudaStream, *const f64, usize, usize) -> Result<()>,
{
    crate::profile_scope!("GPU::AsyncPipelineFromPinned");

    if align_elements == 0 {
        return Err(MahoutError::InvalidInput(
            "Alignment must be greater than zero".to_string(),
        ));
    }
    if !pinned_data.len().is_multiple_of(align_elements) {
        return Err(MahoutError::InvalidInput(format!(
            "Pinned data length {} is not aligned to {} elements",
            pinned_data.len(),
            align_elements
        )));
    }

    let host_mem_gb = crate::gpu::pipeline_config::PipelineConfig::get_host_memory_gb();
    let config = crate::gpu::pipeline_config::PipelineConfig::get_or_build(device, host_mem_gb)?;
    let base_chunk_elements = config.chunk_size_elements()?;
    let pool_size = config.pinned_pool_size_resolved()?;
    let chunk_size_elements = if align_elements >= base_chunk_elements {
        align_elements
    } else {
        base_chunk_elements - (base_chunk_elements % align_elements)
    };

    run_dual_stream_pipeline_with_chunk_size_from_pinned(
        device,
        pinned_data,
        chunk_size_elements,
        pool_size,
        kernel_launcher,
    )
}

/// Single pipeline implementation: source is pinned, async H2D only. no copy to
/// pinned_slots, only async H2D from pinned_data ptr. Caller guarantees pinned_data is pinned.
#[cfg(target_os = "linux")]
fn run_dual_stream_pipeline_with_chunk_size_from_pinned<F>(
    device: &Arc<CudaDevice>,
    pinned_data: &[f64],
    chunk_size_elements: usize,
    pool_size: usize,
    mut kernel_launcher: F,
) -> Result<()>
where
    F: FnMut(&CudaStream, *const f64, usize, usize) -> Result<()>,
{
    if chunk_size_elements == 0 {
        return Err(MahoutError::InvalidInput(
            "Chunk size must be greater than zero".to_string(),
        ));
    }
    if pool_size == 0 {
        return Err(MahoutError::InvalidInput(
            "Pool size must be greater than zero".to_string(),
        ));
    }

    let enable_overlap_tracking = std::env::var("QDP_ENABLE_OVERLAP_TRACKING")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let ctx = PipelineContext::new(device, pool_size)?;
    let num_chunks = pinned_data.len().div_ceil(chunk_size_elements);
    if num_chunks == 0 {
        return Ok(());
    }

    fn chunk_offset_len(host_len: usize, chunk_size: usize, idx: usize) -> (usize, usize) {
        let offset = idx * chunk_size;
        let len = chunk_size.min(host_len.saturating_sub(offset));
        (offset, len)
    }

    let chunk_bytes = chunk_size_elements * std::mem::size_of::<f64>();
    ensure_device_memory_available(pool_size * chunk_bytes, "pipeline device buffer pool", None)?;

    let mut device_buffers: Vec<CudaSlice<f64>> = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        let buf = unsafe { device.alloc::<f64>(chunk_size_elements) }.map_err(|e| {
            map_allocation_error(chunk_bytes, "pipeline device buffer pool", None, e)
        })?;
        device_buffers.push(buf);
    }

    let overlap_tracker = if enable_overlap_tracking {
        Some(OverlapTracker::new(pool_size, true)?)
    } else {
        None
    };

    let pinned_ptr = pinned_data.as_ptr();
    let mut current_copy_idx: usize;
    let mut process_idx: usize;

    {
        crate::profile_scope!("GPU::Prologue");
        let prologue_chunks = pool_size.min(num_chunks);
        for (slot, buf) in device_buffers.iter().enumerate().take(prologue_chunks) {
            let (offset, len) = chunk_offset_len(pinned_data.len(), chunk_size_elements, slot);
            if len == 0 {
                break;
            }
            if let Some(ref tracker) = overlap_tracker
                && let Err(e) = tracker.record_copy_start(&ctx.stream_copy, slot)
            {
                log::warn!("Prologue chunk {}: record_copy_start failed: {}", slot, e);
            }
            unsafe {
                ctx.async_copy_to_device(
                    pinned_ptr.add(offset) as *const c_void,
                    *buf.device_ptr() as *mut c_void,
                    len,
                )?;
            }
            if let Some(ref tracker) = overlap_tracker
                && let Err(e) = tracker.record_copy_end(&ctx.stream_copy, slot)
            {
                log::warn!("Prologue chunk {}: record_copy_end failed: {}", slot, e);
            }
            unsafe {
                ctx.record_copy_done(slot)?;
            }
        }
        current_copy_idx = prologue_chunks;
        process_idx = 0;
    }

    while process_idx < num_chunks {
        let slot = process_idx % pool_size;
        let (offset, len) = chunk_offset_len(pinned_data.len(), chunk_size_elements, process_idx);
        if len == 0 {
            process_idx += 1;
            continue;
        }

        crate::profile_scope!("GPU::ChunkProcess");

        if current_copy_idx >= pool_size && process_idx > current_copy_idx - pool_size {
            let reuse_slot = (current_copy_idx - pool_size) % pool_size;
            unsafe {
                ctx.wait_for_compute(reuse_slot)?;
            }
        }

        if current_copy_idx < num_chunks
            && (current_copy_idx < pool_size || process_idx > current_copy_idx - pool_size)
        {
            let copy_slot = current_copy_idx % pool_size;
            if current_copy_idx >= pool_size {
                unsafe {
                    ctx.synchronize_copy_event(copy_slot)?;
                }
            }
            let (copy_offset, copy_len) =
                chunk_offset_len(pinned_data.len(), chunk_size_elements, current_copy_idx);
            if let Some(ref tracker) = overlap_tracker
                && let Err(e) = tracker.record_copy_start(&ctx.stream_copy, copy_slot)
            {
                log::warn!(
                    "Chunk {}: record_copy_start failed: {}",
                    current_copy_idx,
                    e
                );
            }
            unsafe {
                ctx.async_copy_to_device(
                    pinned_ptr.add(copy_offset) as *const c_void,
                    *device_buffers[copy_slot].device_ptr() as *mut c_void,
                    copy_len,
                )?;
            }
            if let Some(ref tracker) = overlap_tracker
                && let Err(e) = tracker.record_copy_end(&ctx.stream_copy, copy_slot)
            {
                log::warn!("Chunk {}: record_copy_end failed: {}", current_copy_idx, e);
            }
            unsafe {
                ctx.record_copy_done(copy_slot)?;
            }
            current_copy_idx += 1;
        }

        unsafe {
            ctx.wait_for_copy(slot)?;
        }

        let input_ptr = *device_buffers[slot].device_ptr() as *const f64;
        if let Some(ref tracker) = overlap_tracker
            && let Err(e) = tracker.record_compute_start(&ctx.stream_compute, slot)
        {
            log::warn!("Chunk {}: record_compute_start failed: {}", process_idx, e);
        }
        kernel_launcher(&ctx.stream_compute, input_ptr, offset, len)?;
        if let Some(ref tracker) = overlap_tracker
            && let Err(e) = tracker.record_compute_end(&ctx.stream_compute, slot)
        {
            log::warn!("Chunk {}: record_compute_end failed: {}", process_idx, e);
        }
        unsafe {
            ctx.record_compute_done(slot)?;
        }
        #[allow(clippy::manual_is_multiple_of)]
        if let Some(ref tracker) = overlap_tracker
            && (process_idx % 10 == 0 || process_idx == 0)
            && let Err(e) = tracker.log_overlap(process_idx)
            && log::log_enabled!(log::Level::Debug)
        {
            log::debug!("Overlap tracking failed for chunk {}: {}", process_idx, e);
        }
        process_idx += 1;
    }

    {
        crate::profile_scope!("GPU::StreamSync");
        unsafe {
            ctx.sync_copy_stream()?;
            ctx.sync_compute_stream_via_query()?;
        }
    }
    drop(device_buffers);
    Ok(())
}
