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

//! Elastic scheduler: dual-threshold (time + size) batching with a dedicated thread.
//! Ref: docs/optimization/REQUEST_COALESCING_REFERENCE_AND_DESIGN.md §3.8, §3.9, §3.11.4.

// Ref: https://docs.rs/crossbeam-channel/0.5/crossbeam_channel/fn.after.html (one-shot)
// Ref: https://docs.rs/crossbeam-channel/0.5/crossbeam_channel/fn.bounded.html (backpressure)

use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvError, after, bounded, select};
use cudarc::driver::CudaDevice;

use crate::dlpack::DLManagedTensor;
use crate::error::{MahoutError, Result};
use crate::gpu::encodings::QuantumEncoder;
use crate::gpu::memory::{GpuStateVector, Precision};

use super::coalescer::{CoalescerConfig, EncodeRequest, run_coalesced};

/// Channel message type: (batch state vector, (batch_start, batch_end) sample boundary).
pub type EncodeResult = Result<(Arc<GpuStateVector>, (usize, usize))>;
/// Sender for encode result (used by scheduler thread).
pub type EncodeResultSender = mpsc::Sender<EncodeResult>;
/// Receiver for encode result (used by EncodeHandle).
pub type EncodeResultReceiver = mpsc::Receiver<EncodeResult>;

/// One encode task: owned data + channel to send (batch_arc, boundary) on completion.
#[derive(Debug)]
pub struct EncodeTask {
    pub data: Vec<f64>,
    pub num_samples: usize,
    pub sample_size: usize,
    pub tx: EncodeResultSender,
}

impl EncodeTask {
    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    pub fn total_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>()
    }
}

/// Scheduler config: aligns with CoalescerConfig + channel capacity for backpressure (§3.11.2).
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub coalescer: CoalescerConfig,
    /// Bounded channel capacity; when full, send() blocks (backpressure).
    pub channel_capacity: usize,
    /// Max delay from first request to flush; 0.1 ms–1 ms recommended (§3.8).
    pub max_delay: Duration,
    /// Target batch size in bytes; flush when accumulated >= this (§3.8).
    pub target_batch_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            coalescer: CoalescerConfig::default(),
            channel_capacity: 1024,
            max_delay: Duration::from_micros(500), // 0.5 ms
            target_batch_size: 8 * 1024 * 1024,    // 8 MB
        }
    }
}

/// Flush buffered tasks: run coalescer, then send (Arc(batch), boundary) to each task's tx.
fn flush_buffer(
    device: &Arc<CudaDevice>,
    encoder: &dyn QuantumEncoder,
    sample_size: usize,
    num_qubits: usize,
    config: &CoalescerConfig,
    buffer: &mut Vec<EncodeTask>,
) -> Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }
    let tasks = std::mem::take(buffer);
    let run_result = run_coalesced(
        device,
        encoder,
        sample_size,
        num_qubits,
        config.clone(),
        |batcher| {
            for t in &tasks {
                if t.num_samples == 1 {
                    batcher.submit(EncodeRequest::Single(&t.data))?;
                } else {
                    batcher.submit(EncodeRequest::Batch {
                        data: &t.data,
                        num_samples: t.num_samples,
                        sample_size: t.sample_size,
                    })?;
                }
            }
            Ok(())
        },
    );
    match run_result {
        Ok((batch, boundaries)) => {
            let batch_arc = Arc::new(batch);
            for (task, (start, num)) in tasks.into_iter().zip(boundaries) {
                let _ = task.tx.send(Ok((Arc::clone(&batch_arc), (start, num))));
            }
        }
        Err(e) => {
            let msg = e.to_string();
            for task in tasks {
                let _ = task.tx.send(Err(MahoutError::InvalidInput(msg.clone())));
            }
        }
    }
    Ok(())
}

/// Scheduler thread loop: dual-threshold (size + time) with crossbeam select! + after(duration).
fn scheduler_loop(
    task_rx: Receiver<EncodeTask>,
    device: Arc<CudaDevice>,
    encoder: &dyn QuantumEncoder,
    sample_size: usize,
    num_qubits: usize,
    config: SchedulerConfig,
) {
    let mut buffer: Vec<EncodeTask> = Vec::new();
    let mut current_bytes: usize = 0;
    let mut deadline: Option<Instant> = None;
    let target = config.target_batch_size;
    let max_bytes = config.coalescer.max_batch_bytes;
    let max_delay = config.max_delay;

    loop {
        let now = Instant::now();
        let timeout_rx = deadline.map(|d| after(d.saturating_duration_since(now)));

        let received = if buffer.is_empty() {
            // Block until first task
            match task_rx.recv() {
                Ok(t) => Some(t),
                Err(RecvError) => break,
            }
        } else if let Some(ref timeout) = timeout_rx {
            select! {
                recv(task_rx) -> msg => msg.ok(),
                recv(timeout) -> _ => {
                    deadline = None;
                    if let Err(e) = flush_buffer(
                        &device,
                        encoder,
                        sample_size,
                        num_qubits,
                        &config.coalescer,
                        &mut buffer,
                    ) {
                        log::warn!("Scheduler flush_buffer error: {}", e);
                    }
                    current_bytes = 0;
                    continue;
                }
            }
        } else {
            match task_rx.recv() {
                Ok(t) => Some(t),
                Err(RecvError) => break,
            }
        };

        let Some(task) = received else { break };

        let task_bytes = task.total_bytes();
        let task_samples = task.num_samples;
        let task_ss = task.sample_size;

        // Reject single task exceeding max_batch_bytes (§3.11.7 optional: avoid futile flush)
        if task_bytes > max_bytes {
            let _ = task.tx.send(Err(MahoutError::InvalidInput(format!(
                "Scheduler: task size {} bytes exceeds max_batch_bytes {}; reject",
                task_bytes, max_bytes
            ))));
            continue;
        }

        // Pass-through: single task >= target_batch_size and buffer empty (§3.8)
        if buffer.is_empty() && task_bytes >= target {
            let run_result = run_coalesced(
                &device,
                encoder,
                task_ss,
                num_qubits,
                config.coalescer.clone(),
                |batcher| {
                    if task_samples == 1 {
                        batcher.submit(EncodeRequest::Single(&task.data))?;
                    } else {
                        batcher.submit(EncodeRequest::Batch {
                            data: &task.data,
                            num_samples: task.num_samples,
                            sample_size: task.sample_size,
                        })?;
                    }
                    Ok(())
                },
            );
            match run_result {
                Ok((batch, boundaries)) => {
                    let (start, num) = boundaries.into_iter().next().unwrap_or((0, 1));
                    let _ = task.tx.send(Ok((Arc::new(batch), (start, num))));
                }
                Err(e) => {
                    let _ = task.tx.send(Err(e));
                }
            }
            continue;
        }

        // Would exceed max_batch_bytes: flush buffer first, then push task
        if current_bytes + task_bytes > max_bytes && !buffer.is_empty() {
            if let Err(e) = flush_buffer(
                &device,
                encoder,
                sample_size,
                num_qubits,
                &config.coalescer,
                &mut buffer,
            ) {
                log::warn!("Scheduler flush_buffer error: {}", e);
            }
            current_bytes = 0;
            deadline = None;
        }

        if task_ss != sample_size {
            let _ = task.tx.send(Err(MahoutError::InvalidInput(format!(
                "Scheduler: task sample_size {} != scheduler sample_size {}",
                task_ss, sample_size
            ))));
            continue;
        }

        current_bytes += task_bytes;
        if deadline.is_none() {
            deadline = Some(now + max_delay);
        }
        buffer.push(task);

        if current_bytes >= target {
            if let Err(e) = flush_buffer(
                &device,
                encoder,
                sample_size,
                num_qubits,
                &config.coalescer,
                &mut buffer,
            ) {
                log::warn!("Scheduler flush_buffer error: {}", e);
            }
            current_bytes = 0;
            deadline = None;
        }
    }

    // Drain remaining
    if !buffer.is_empty() {
        let _ = flush_buffer(
            &device,
            encoder,
            sample_size,
            num_qubits,
            &config.coalescer,
            &mut buffer,
        );
    }
}

/// Handle to submit encode tasks; scheduler runs in a background thread.
pub struct EncodeScheduler {
    task_tx: crossbeam_channel::Sender<EncodeTask>,
    _join: Option<thread::JoinHandle<()>>,
}

impl EncodeScheduler {
    /// Start the scheduler thread. Encoder is owned by the thread (e.g. from get_encoder()).
    pub fn new(
        device: Arc<CudaDevice>,
        encoder: Box<dyn QuantumEncoder + Send>,
        sample_size: usize,
        num_qubits: usize,
        config: SchedulerConfig,
    ) -> Self {
        let (task_tx, task_rx) = bounded(config.channel_capacity);
        let join = thread::Builder::new()
            .name("qdp-coalesce-scheduler".into())
            .spawn(move || {
                scheduler_loop(task_rx, device, &*encoder, sample_size, num_qubits, config);
            })
            .expect("spawn scheduler thread");
        Self {
            task_tx,
            _join: Some(join),
        }
    }

    /// Submit a task; blocks until the scheduler accepts it (backpressure if channel full).
    pub fn submit(&self, task: EncodeTask) -> Result<()> {
        self.task_tx
            .send(task)
            .map_err(|_| MahoutError::InvalidInput("EncodeScheduler: channel closed".to_string()))
    }

    /// Non-blocking submit; returns Err if channel is full (§3.11.2).
    pub fn try_submit(&self, task: EncodeTask) -> Result<()> {
        self.task_tx.try_send(task).map_err(|e| {
            MahoutError::InvalidInput(format!("EncodeScheduler: try_send failed ({:?})", e))
        })
    }
}

/// Handle for deferred result of encode_async (§3.9).
/// Call `get()` to block and receive the encoded tensor (DLPack pointer).
pub struct EncodeHandle {
    rx: EncodeResultReceiver,
    device: Arc<CudaDevice>,
    precision: Precision,
    _num_qubits: usize,
}

impl EncodeHandle {
    /// Create a handle that will receive the result from the scheduler.
    pub fn new(
        rx: EncodeResultReceiver,
        device: Arc<CudaDevice>,
        precision: Precision,
        num_qubits: usize,
    ) -> Self {
        Self {
            rx,
            device,
            precision,
            _num_qubits: num_qubits,
        }
    }

    /// Block until the result is ready; returns DLPack pointer for the encoded state.
    /// Preserves order: result corresponds to the single request submitted with encode_async.
    pub fn get(self) -> Result<*mut DLManagedTensor> {
        let msg = self.rx.recv().map_err(|_| {
            MahoutError::InvalidInput(
                "EncodeHandle: channel closed (scheduler dropped)".to_string(),
            )
        })?;
        let (batch, (start, num)) = msg?;
        let sub = GpuStateVector::copy_sample_range(&self.device, &batch, start, num)?;
        let state = sub.to_precision(&self.device, self.precision)?;
        Ok(state.to_dlpack())
    }
}
