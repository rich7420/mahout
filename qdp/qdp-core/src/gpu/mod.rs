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

#[cfg(target_os = "linux")]
pub mod batch_pool;
#[cfg(target_os = "linux")]
pub mod buffer_pool;
pub mod coalescer;
pub mod encodings;
pub mod memory;
#[cfg(target_os = "linux")]
pub mod overlap_tracker;
pub mod pipeline;
#[cfg(target_os = "linux")]
pub mod pipeline_config;
#[cfg(target_os = "linux")]
pub mod pool_metrics;
#[cfg(target_os = "linux")]
pub mod scheduler;

#[cfg(target_os = "linux")]
pub(crate) mod cuda_ffi;

#[cfg(target_os = "linux")]
pub use buffer_pool::{PinnedBufferHandle, PinnedBufferPool};
pub use coalescer::{CoalescerConfig, EncodeCoalescer, EncodeRequest, encode_list, run_coalesced};
pub use encodings::{AmplitudeEncoder, AngleEncoder, BasisEncoder, QuantumEncoder, get_encoder};
pub use memory::GpuStateVector;
pub use pipeline::run_dual_stream_pipeline;

#[cfg(target_os = "linux")]
pub use batch_pool::{BatchHandle, BatchJob, BatchJobData, BatchPool};
#[cfg(target_os = "linux")]
pub use overlap_tracker::OverlapTracker;
#[cfg(target_os = "linux")]
pub use pipeline::PipelineContext;
#[cfg(target_os = "linux")]
pub use pool_metrics::{PoolMetrics, PoolUtilizationReport};
#[cfg(target_os = "linux")]
pub use scheduler::{
    EncodeHandle, EncodeResult, EncodeResultReceiver, EncodeResultSender, EncodeScheduler,
    EncodeTask, SchedulerConfig,
};
#[cfg(target_os = "linux")]
/// Map key (num_qubits, batch_size, encoding_id) -> EncodeScheduler. Reduces type complexity.
pub type SchedulerMap = std::collections::HashMap<(usize, usize, String), EncodeScheduler>;
