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

//! Request coalescing for encode() and encode_batch().
//!
//! Buffers multiple small encode / encode_batch requests and flushes them
//! as one large batch to reduce sync count and improve GPU utilization.
//! **Zero-Copy Ingress + One-Copy to Pinned**: queue stores only metadata
//! (`&[f64]`); flush writes once into a Pinned buffer (parallel memcpy) then
//! runs encode_batch pipeline. See docs/optimization/REQUEST_COALESCING_REFERENCE_AND_DESIGN.md.

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::error::{MahoutError, Result};
use crate::gpu::encodings::QuantumEncoder;
use crate::gpu::memory::GpuStateVector;

// Ref: https://doc.rust-lang.org/std/primitive.slice.html
// Ref: docs/optimization/REQUEST_COALESCING_REFERENCE_AND_DESIGN.md §3.2, §3.6

/// Wrapper for raw pointer used in parallel non-overlapping writes (rayon).
/// Safety: only used so that each thread writes to dst[off..off+len] with disjoint ranges.
#[cfg(target_os = "linux")]
struct SendPtr(*mut f64);
#[cfg(target_os = "linux")]
unsafe impl Send for SendPtr {}
#[cfg(target_os = "linux")]
unsafe impl Sync for SendPtr {}
#[cfg(target_os = "linux")]
impl SendPtr {
    #[inline]
    fn copy_into(&self, offset: usize, src: &[f64]) {
        debug_assert!(offset.checked_add(src.len()).is_some());
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.0.add(offset), src.len());
            dst.copy_from_slice(src);
        }
    }
}

/// One logical encode request: either a single sample (encode) or a batch (encode_batch).
/// Holds **references** only (Zero-Copy Ingress); no owned Vec.
#[derive(Clone, Copy, Debug)]
pub enum EncodeRequest<'a> {
    /// Single 1D encode: one sample, length = sample_size.
    Single(&'a [f64]),
    /// Batch encode: num_samples × sample_size elements.
    Batch {
        data: &'a [f64],
        num_samples: usize,
        sample_size: usize,
    },
}

impl<'a> EncodeRequest<'a> {
    /// Number of logical samples in this request.
    pub fn num_samples(&self) -> usize {
        match self {
            EncodeRequest::Single(_) => 1,
            EncodeRequest::Batch { num_samples, .. } => *num_samples,
        }
    }

    /// Sample size (elements per sample). Single uses data.len(); Batch uses sample_size.
    pub fn sample_size(&self) -> usize {
        match self {
            EncodeRequest::Single(d) => d.len(),
            EncodeRequest::Batch { sample_size, .. } => *sample_size,
        }
    }

    /// Total number of f64 elements.
    pub fn total_elements(&self) -> usize {
        match self {
            EncodeRequest::Single(d) => d.len(),
            EncodeRequest::Batch { data, .. } => data.len(),
        }
    }

    /// Reference to the underlying data slice (for copy into pinned buffer).
    pub fn as_slice(&self) -> &'a [f64] {
        match self {
            EncodeRequest::Single(d) => d,
            EncodeRequest::Batch { data, .. } => data,
        }
    }
}

/// Coalescer configuration: when to flush (max samples or max bytes).
/// **max_batch_bytes** should align with Pinned Pool single-block capacity (§3.11.6).
/// Ref: docs/optimization/REQUEST_COALESCING_REFERENCE_AND_DESIGN.md §3.3, §3.8
#[derive(Clone, Debug)]
pub struct CoalescerConfig {
    pub max_batch_samples: usize,
    pub max_batch_bytes: usize,
    /// Optional target batch size (e.g. 8 MB) for elastic scheduler (§3.8).
    pub target_batch_size: Option<usize>,
    /// Optional max delay from first request to flush (§3.8); 0.1 ms–1 ms recommended. Used by scheduler thread.
    pub max_delay: Option<std::time::Duration>,
}

impl Default for CoalescerConfig {
    fn default() -> Self {
        Self {
            max_batch_samples: 128,
            max_batch_bytes: 64 * 1024 * 1024,
            target_batch_size: None,
            max_delay: None,
        }
    }
}

/// Buffers encode requests (metadata only) and flushes into one Pinned buffer + one pipeline run.
///
/// **Zero-Copy**: `submit(EncodeRequest::Single(&slice))` stores only the slice reference.
/// **flush**: acquires a Pinned buffer from pool, copies all request slices in parallel (rayon)
/// into it, then calls `encoder.encode_batch_via_pipeline(device, pinned_slice, ...)` (§3.6.1). Results are returned
/// as one batch plus per-request boundaries `(start_sample, num_samples)`.
///
/// All requests in one coalescer must share the same `sample_size` and `num_qubits`.
#[derive(Clone)]
pub struct EncodeCoalescer<'a> {
    queue: Vec<EncodeRequest<'a>>,
    sample_size: usize,
    num_qubits: usize,
    config: CoalescerConfig,
    #[cfg(target_os = "linux")]
    pool: Option<Arc<crate::gpu::buffer_pool::PinnedBufferPool>>,
}

impl<'a> std::fmt::Debug for EncodeCoalescer<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodeCoalescer")
            .field("queue_len", &self.queue.len())
            .field("sample_size", &self.sample_size)
            .field("num_qubits", &self.num_qubits)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<'a> EncodeCoalescer<'a> {
    /// Create a coalescer with fixed sample_size and num_qubits.
    /// On Linux, allocates a Pinned buffer pool (single-block capacity = max_batch_bytes).
    pub fn new(sample_size: usize, num_qubits: usize, config: CoalescerConfig) -> Result<Self> {
        #[cfg(target_os = "linux")]
        let pool = {
            let elements_per_buffer = config.max_batch_bytes / std::mem::size_of::<f64>();
            if elements_per_buffer == 0 {
                return Err(MahoutError::InvalidInput(
                    "CoalescerConfig::max_batch_bytes too small (must fit at least one f64)"
                        .to_string(),
                ));
            }
            let pool_size = 2u32; // allow one in use, one free; avoid blocking next flush
            Some(crate::gpu::buffer_pool::PinnedBufferPool::new(
                pool_size as usize,
                elements_per_buffer,
            )?)
        };
        #[cfg(not(target_os = "linux"))]
        let _ = config.max_batch_bytes;

        Ok(Self {
            queue: Vec::new(),
            sample_size,
            num_qubits,
            config,
            #[cfg(target_os = "linux")]
            pool,
        })
    }

    /// Enqueue a request (metadata only; no copy). Returns error if sample_size does not match.
    /// Rejects single request whose size exceeds max_batch_bytes or pool single-block capacity (§3.11.7).
    pub fn submit(&mut self, request: EncodeRequest<'a>) -> Result<()> {
        let req_sample_size = request.sample_size();
        if req_sample_size != self.sample_size {
            return Err(MahoutError::InvalidInput(format!(
                "EncodeCoalescer: request sample_size {} does not match coalescer sample_size {}",
                req_sample_size, self.sample_size
            )));
        }
        let req_bytes = request.total_elements() * std::mem::size_of::<f64>();
        if req_bytes > self.config.max_batch_bytes {
            return Err(MahoutError::InvalidInput(format!(
                "EncodeCoalescer: single request size {} bytes exceeds max_batch_bytes {}; reject or split",
                req_bytes, self.config.max_batch_bytes
            )));
        }
        #[cfg(target_os = "linux")]
        if let Some(ref pool) = self.pool
            && request.total_elements() > pool.elements_per_buffer()
        {
            return Err(MahoutError::InvalidInput(format!(
                "EncodeCoalescer: single request elements {} exceeds pool block capacity {}",
                request.total_elements(),
                pool.elements_per_buffer()
            )));
        }
        let total_bytes_after = self.total_bytes() + req_bytes;
        if total_bytes_after > self.config.max_batch_bytes {
            return Err(MahoutError::InvalidInput(
                "EncodeCoalescer: adding this request would exceed max_batch_bytes; flush first"
                    .to_string(),
            ));
        }
        self.queue.push(request);
        Ok(())
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Number of requests currently queued.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Total logical samples in the queue.
    pub fn total_samples(&self) -> usize {
        self.queue.iter().map(EncodeRequest::num_samples).sum()
    }

    /// Total bytes (f64 elements * 8) in the queue.
    pub fn total_bytes(&self) -> usize {
        self.queue
            .iter()
            .map(EncodeRequest::total_elements)
            .sum::<usize>()
            * std::mem::size_of::<f64>()
    }

    /// True if adding one more request would exceed max_batch_samples or max_batch_bytes.
    pub fn would_exceed_limits(&self, request: &EncodeRequest<'a>) -> bool {
        let new_samples = self.total_samples() + request.num_samples();
        let new_bytes = self.total_bytes() + request.total_elements() * std::mem::size_of::<f64>();
        new_samples >= self.config.max_batch_samples || new_bytes >= self.config.max_batch_bytes
    }

    /// Merge queue into one contiguous Pinned buffer (parallel copy), run encode via pipeline once;
    /// return merged batch and per-request boundaries (start_sample, num_samples). Queue cleared after flush.
    ///
    /// **Linux**: Uses Pinned buffer pool; parallel memcpy (rayon) by request boundary; then
    /// `encoder.encode_batch_via_pipeline(device, pinned_slice, ...)` per §3.6.1 (run_dual_stream_pipeline).
    /// Pinned buffer is returned to pool on drop.
    /// **Non-Linux**: Returns error (pool and pipeline are Linux-only).
    #[cfg(target_os = "linux")]
    pub fn flush(
        &mut self,
        device: &Arc<CudaDevice>,
        encoder: &dyn QuantumEncoder,
    ) -> Result<(GpuStateVector, Vec<(usize, usize)>)> {
        if self.queue.is_empty() {
            return Err(MahoutError::InvalidInput(
                "EncodeCoalescer::flush called with empty queue".to_string(),
            ));
        }

        let total_elements: usize = self.queue.iter().map(EncodeRequest::total_elements).sum();
        let total_samples = self.total_samples();
        let pool = self.pool.as_ref().ok_or_else(|| {
            MahoutError::InvalidInput("Coalescer pool not initialized".to_string())
        })?;
        if total_elements > pool.elements_per_buffer() {
            return Err(MahoutError::InvalidInput(format!(
                "EncodeCoalescer::flush total elements {} exceeds pool single-block capacity {}; \
                 align max_batch_bytes with pool or split batch",
                total_elements,
                pool.elements_per_buffer()
            )));
        }

        let mut handle = pool.acquire();
        let pinned_slice = handle.as_slice_mut();
        let base_ptr = pinned_slice.as_mut_ptr();

        // Build (offset, len) per request for non-overlapping parallel copy.
        let mut offset = 0usize;
        let mut boundaries = Vec::with_capacity(self.queue.len());
        for req in &self.queue {
            let len = req.total_elements();
            boundaries.push((offset, len));
            offset += len;
        }

        // Parallel copy by request boundary (§3.11.1: non-overlapping, safe).
        let base = SendPtr(base_ptr);

        use rayon::prelude::*;
        let copy_ranges: Vec<(usize, usize)> = boundaries;
        self.queue
            .par_iter()
            .zip(copy_ranges.par_iter())
            .for_each(move |(req, (off, len))| {
                let src = req.as_slice();
                debug_assert!(src.len() == *len);
                base.copy_into(*off, src);
            });

        let batch_state = encoder.encode_batch_via_pipeline_from_pinned(
            device,
            &pinned_slice[..total_elements],
            total_samples,
            self.sample_size,
            self.num_qubits,
        )?;

        let sample_boundaries: Vec<(usize, usize)> = self
            .queue
            .iter()
            .scan(0usize, |start_sample, req| {
                let n = req.num_samples();
                let pair = (*start_sample, n);
                *start_sample += n;
                Some(pair)
            })
            .collect();

        self.queue.clear();
        drop(handle); // return buffer to pool (RAII)
        Ok((batch_state, sample_boundaries))
    }

    /// Flush on non-Linux: not supported (no Pinned pool).
    #[cfg(not(target_os = "linux"))]
    pub fn flush(
        &mut self,
        _device: &std::sync::Arc<CudaDevice>,
        _encoder: &dyn QuantumEncoder,
    ) -> Result<(GpuStateVector, Vec<(usize, usize)>)> {
        if self.queue.is_empty() {
            return Err(MahoutError::InvalidInput(
                "EncodeCoalescer::flush called with empty queue".to_string(),
            ));
        }
        Err(MahoutError::Cuda(
            "Request coalescing with Pinned buffer is only available on Linux".to_string(),
        ))
    }
}

/// Scope-based API: run a callback with a coalescer, then flush and return (batch, boundaries).
/// **Zero-Copy**: all `submit()` calls inside the closure use `&[f64]`; data is copied only once
/// into the Pinned buffer during flush. Caller must ensure slices outlive the closure.
///
/// Ref: docs/optimization/REQUEST_COALESCING_REFERENCE_AND_DESIGN.md §3.4 (Scheme C), §3.6
pub fn run_coalesced<'a, F>(
    device: &Arc<CudaDevice>,
    encoder: &dyn QuantumEncoder,
    sample_size: usize,
    num_qubits: usize,
    config: CoalescerConfig,
    callback: F,
) -> Result<(GpuStateVector, Vec<(usize, usize)>)>
where
    F: FnOnce(&mut EncodeCoalescer<'a>) -> Result<()>,
{
    let mut coalescer = EncodeCoalescer::new(sample_size, num_qubits, config)?;
    callback(&mut coalescer)?;
    coalescer.flush(device, encoder)
}

/// Batch API: one call with multiple slices (each slice = one sample of length `sample_size`).
/// Coalesces all slices into one batch, runs encode_batch once, returns merged batch and
/// per-request boundaries `(start_sample, num_samples)` — one `(i, 1)` per slice in order.
///
/// Reduces FFI and Python object overhead vs N separate encode() or encode_async() calls (§3.9, §3.11.2).
/// Caller must ensure total size ≤ `config.max_batch_bytes` (otherwise submit will error).
///
/// Ref: docs/optimization/REQUEST_COALESCING_REFERENCE_AND_DESIGN.md §3.9 (encode_list), §3.11.2
pub fn encode_list(
    device: &Arc<CudaDevice>,
    encoder: &dyn QuantumEncoder,
    sample_size: usize,
    num_qubits: usize,
    config: CoalescerConfig,
    slices: &[&[f64]],
) -> Result<(GpuStateVector, Vec<(usize, usize)>)> {
    if slices.is_empty() {
        return Err(MahoutError::InvalidInput(
            "encode_list: empty slice list".to_string(),
        ));
    }
    run_coalesced(
        device,
        encoder,
        sample_size,
        num_qubits,
        config,
        |batcher| {
            for s in slices {
                if s.len() != sample_size {
                    return Err(MahoutError::InvalidInput(format!(
                        "encode_list: slice len {} != sample_size {}",
                        s.len(),
                        sample_size
                    )));
                }
                batcher.submit(EncodeRequest::Single(s))?;
            }
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_num_samples_and_size() {
        let data = [0.0f64; 64];
        let single = EncodeRequest::Single(&data[..]);
        assert_eq!(single.num_samples(), 1);
        assert_eq!(single.sample_size(), 64);
        assert_eq!(single.total_elements(), 64);
        assert_eq!(single.as_slice().len(), 64);

        let batch_data = [0.0f64; 640];
        let batch = EncodeRequest::Batch {
            data: &batch_data[..],
            num_samples: 10,
            sample_size: 64,
        };
        assert_eq!(batch.num_samples(), 10);
        assert_eq!(batch.sample_size(), 64);
        assert_eq!(batch.total_elements(), 640);
    }

    #[test]
    fn coalescer_submit_validates_sample_size() {
        let config = CoalescerConfig::default();
        let mut c = EncodeCoalescer::new(64, 4, config).unwrap();
        let v64 = vec![0.0f64; 64];
        let v32 = vec![0.0f64; 32];
        assert!(c.submit(EncodeRequest::Single(&v64[..])).is_ok());
        assert!(c.submit(EncodeRequest::Single(&v32[..])).is_err());
    }

    #[test]
    fn encode_list_empty_errors() {
        let config = CoalescerConfig::default();
        let device = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let encoder = crate::gpu::encodings::AmplitudeEncoder;
        let slices: &[&[f64]] = &[];
        let res = encode_list(&device, &encoder, 64, 4, config, slices);
        assert!(matches!(res, Err(MahoutError::InvalidInput(_))));
    }

    #[test]
    fn encode_list_slice_len_mismatch_errors() {
        let config = CoalescerConfig::default();
        let device = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let encoder = crate::gpu::encodings::AmplitudeEncoder;
        let data = vec![0.0f64; 32];
        let slices = [data.as_slice()];
        let res = encode_list(&device, &encoder, 64, 4, config, &slices);
        assert!(matches!(res, Err(MahoutError::InvalidInput(_))));
    }

    #[test]
    fn coalescer_empty_flush_errors() {
        let config = CoalescerConfig::default();
        let mut c = EncodeCoalescer::new(64, 4, config).unwrap();
        let device = match cudarc::driver::CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU, skip
        };
        let encoder = crate::gpu::encodings::AmplitudeEncoder;
        let res = c.flush(&device, &encoder);
        assert!(
            matches!(
                res,
                Err(MahoutError::InvalidInput(_)) | Err(MahoutError::Cuda(_))
            ),
            "empty coalescer flush should return Err"
        );
    }
}
