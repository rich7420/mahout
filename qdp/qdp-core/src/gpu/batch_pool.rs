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

//! Rust-side batch pool: single GPU master thread + bounded queue of work descriptors.
//!
//! One dedicated master thread runs encode_batch_via_pipeline (one CUDA context, no multi-worker
//! contention). Bounded queue (env QDP_BATCH_QUEUE_CAPACITY, default 4) provides backpressure.
//! Ref: CUDA §2.3 async execution, §4.10 pipelines; Triton one scheduler per model; DALI prefetch.

use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use crossbeam_channel::bounded;
use cudarc::driver::CudaDevice;

use crate::error::{MahoutError, Result};
use crate::gpu::get_encoder;
use crate::gpu::memory::{GpuStateVector, PinnedHostBuffer, Precision, RegisteredHostBuffer};

/// N.7: Pin master thread to one CPU when QDP_MASTER_CPU_ID is set (Linux only, feature thread_affinity).
#[cfg(all(target_os = "linux", feature = "thread_affinity"))]
fn set_master_thread_affinity_if_requested() {
    let cpu_id = match std::env::var("QDP_MASTER_CPU_ID") {
        Ok(s) => match s.parse::<usize>() {
            Ok(n) => n,
            _ => return,
        },
        _ => return,
    };
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(cpu_id, &mut cpuset);
        let r = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
        if r != 0 {
            log::warn!(
                "QDP BatchPool: sched_setaffinity(cpu_id={}) failed (errno {})",
                cpu_id,
                std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
            );
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "thread_affinity")))]
fn set_master_thread_affinity_if_requested() {}

/// Input for one batch: either owned (Vec) or borrowed (ptr + len, zero-copy). Same API; no branch.
pub enum BatchJobData {
    Owned(Vec<f64>),
    Borrowed { ptr: *const f64, len: usize },
}

// Safe: Borrowed is only used when the caller blocks until the job completes (e.g. encode_batch blocks on get()).
unsafe impl Send for BatchJobData {}

/// One batch job: data (owned or borrowed) + channel to send result on completion.
pub struct BatchJob {
    pub data: BatchJobData,
    pub num_samples: usize,
    pub sample_size: usize,
    pub num_qubits: usize,
    pub encoding_method: String,
    pub response_tx: mpsc::Sender<Result<Arc<GpuStateVector>>>,
}

/// R.1 Recycled Pinning: cache last registered (ptr, len) to skip cudaHostRegister when Python reuses the same buffer.
/// Reduces "Registration Tax" (~300μs per batch) to <1μs when the same buffer is submitted consecutively.
fn worker_loop(
    job_rx: crossbeam_channel::Receiver<BatchJob>,
    device: Arc<CudaDevice>,
    precision: Precision,
) {
    set_master_thread_affinity_if_requested();
    #[cfg(target_os = "linux")]
    let mut cached_reg: Option<RegisteredHostBuffer> = None;
    while let Ok(job) = job_rx.recv() {
        let result = (|| {
            let encoder = get_encoder(&job.encoding_method)?;
            let state = match &job.data {
                BatchJobData::Owned(v) => {
                    #[cfg(target_os = "linux")]
                    {
                        cached_reg = None;
                    }
                    let mut pinned = PinnedHostBuffer::new(v.len())?;
                    pinned.as_slice_mut()[..v.len()].copy_from_slice(v);
                    encoder.encode_batch_via_pipeline_from_pinned(
                        &device,
                        pinned.as_slice(),
                        job.num_samples,
                        job.sample_size,
                        job.num_qubits,
                    )?
                }
                BatchJobData::Borrowed { ptr, len } => {
                    #[cfg(target_os = "linux")]
                    let pinned_slice: &[f64] = {
                        let reuse = cached_reg
                            .as_ref()
                            .map(|r| (r.ptr(), r.len()) == (*ptr, *len))
                            .unwrap_or(false);
                        if reuse {
                            cached_reg.as_ref().unwrap().as_slice()
                        } else {
                            cached_reg = None;
                            let reg = unsafe { RegisteredHostBuffer::new(*ptr, *len)? };
                            cached_reg = Some(reg);
                            cached_reg.as_ref().unwrap().as_slice()
                        }
                    };
                    #[cfg(not(target_os = "linux"))]
                    let pinned_slice: &[f64] = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                    encoder.encode_batch_via_pipeline_from_pinned(
                        &device,
                        pinned_slice,
                        job.num_samples,
                        job.sample_size,
                        job.num_qubits,
                    )?
                }
            };
            Ok(Arc::new(state.to_precision(&device, precision)?))
        })();
        let _ = job.response_tx.send(result);
    }
}

/// Pool: single GPU master thread + bounded job queue. Default path for encode_batch.
pub struct BatchPool {
    job_tx: crossbeam_channel::Sender<BatchJob>,
    _join: thread::JoinHandle<()>,
}

impl BatchPool {
    /// Queue capacity from env QDP_BATCH_QUEUE_CAPACITY (default 4). Clamped 2..=8. N.3: bounded queue for backpressure.
    pub fn queue_capacity_from_env() -> usize {
        std::env::var("QDP_BATCH_QUEUE_CAPACITY")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(|n: usize| n.clamp(2, 8))
            .unwrap_or(4)
    }

    /// Number of workers: always 1 (single GPU master). No env; kept for API compatibility.
    pub fn num_workers_from_env() -> usize {
        1
    }

    /// Create pool: one GPU master thread, bounded job channel (capacity from env, default 4).
    pub fn new(device: Arc<CudaDevice>, precision: Precision, _num_workers: usize) -> Self {
        let cap = Self::queue_capacity_from_env();
        let (job_tx, job_rx) = bounded(cap);
        let join = thread::Builder::new()
            .name("qdp-batch-pool-master".to_string())
            .spawn(move || worker_loop(job_rx, device, precision))
            .expect("spawn batch pool master");
        Self {
            job_tx,
            _join: join,
        }
    }

    /// Submit job; caller holds the receiver (BatchHandle). Blocks only until job is queued.
    pub fn submit(&self, job: BatchJob) -> Result<()> {
        self.job_tx
            .send(job)
            .map_err(|_| MahoutError::InvalidInput("BatchPool: channel closed".to_string()))
    }

    /// Submit and return handle. Caller can submit many then get() in order (encode_stream).
    /// Use BatchJobData::Borrowed for zero-copy when caller holds the buffer (e.g. &[f64]).
    pub fn submit_handle(
        &self,
        batch_input: BatchJobData,
        num_samples: usize,
        sample_size: usize,
        num_qubits: usize,
        encoding_method: String,
    ) -> Result<BatchHandle> {
        let (tx, rx) = mpsc::channel();
        self.submit(BatchJob {
            data: batch_input,
            num_samples,
            sample_size,
            num_qubits,
            encoding_method,
            response_tx: tx,
        })?;
        Ok(BatchHandle { rx })
    }
}

/// Handle for one submitted batch; call get() to block and receive DLPack.
pub struct BatchHandle {
    rx: mpsc::Receiver<Result<Arc<GpuStateVector>>>,
}

impl BatchHandle {
    /// Block until the batch result is ready; returns DLPack pointer.
    pub fn get(self) -> Result<*mut crate::dlpack::DLManagedTensor> {
        let batch = self.rx.recv().map_err(|_| {
            MahoutError::InvalidInput("BatchHandle: channel closed (pool dropped)".to_string())
        })??;
        Ok(batch.to_dlpack())
    }
}
