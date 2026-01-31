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

//! Pipeline config: chunk size and pinned pool size with env override and hardware defaults.
//!
//! **Performance-first (PR3 Phase 3):** Design and defaults maximize pipeline throughput, H2D overlap,
//! and reduce copy-stream stalls. PCIe bandwidth dominates H2D time (~5–16 GB/s vs hundreds GB/s on device);
//! small chunks amplify TLP/header overhead; large chunks hurt overlap. We choose chunk "large enough
//! to saturate bandwidth, small enough to overlap" per PCIe/GPU. Pool size 2–4 allows copy stream to
//! transfer chunk N+1 while compute runs chunk N; pinned total is capped at 20% host memory.
//!
//! **Refs:** [NVIDIA: Optimize Data Transfers](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/),
//! [NVIDIA: Overlap Data Transfers](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/),
//! [CUDA Async Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html),
//! [CUDA Memory Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations).
//!
//! **Env:** `QDP_CHUNK_SIZE_MB` (1–256), `QDP_PINNED_POOL_SIZE` (1–16), `QDP_PCIE_GEN` (3/4/5 or gen3/gen4/gen5).
//! Pinned total ≤ 20% host memory. Defaults from PCIe/GPU when unset (PCIe from env only). Chunk size ≥ 1 MB
//! keeps transfers above the ~64 KB threshold recommended for efficient pinned async H2D.

// Ref: https://doc.rust-lang.org/std/sync/struct.OnceLock.html
use std::sync::{Mutex, OnceLock};

use cudarc::driver::CudaDevice;
use std::sync::Arc;

use crate::error::{MahoutError, Result};
use crate::gpu::cuda_ffi::{
    CUDA_SUCCESS, cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
    cudaDeviceGetAttribute,
};

/// Per-process cache: (device_ordinal, config). Avoids rebuilding config on every pipeline call.
/// Benchmark runs 200 encode() calls per run, each building config → 200× /proc/meminfo + 400× cudaDeviceGetAttribute.
static CONFIG_CACHE: OnceLock<Mutex<Option<(i32, PipelineConfig)>>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PCIeGeneration {
    Gen3,
    Gen4,
    Gen5,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputeCapability {
    Ampere,
    Ada,
    Hopper,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default)]
pub struct PipelineConfig {
    pub chunk_size_mb: Option<usize>,
    pub pinned_pool_size: Option<usize>,
    pub enable_async_alloc: bool,
}

impl PipelineConfig {
    /// From env: QDP_CHUNK_SIZE_MB, QDP_PINNED_POOL_SIZE, QDP_PCIE_GEN. Parse failure => leave None.
    pub fn from_env() -> Self {
        let chunk_size_mb = std::env::var("QDP_CHUNK_SIZE_MB")
            .ok()
            .and_then(|s| s.parse().ok());
        let pinned_pool_size = std::env::var("QDP_PINNED_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse().ok());
        Self {
            chunk_size_mb,
            pinned_pool_size,
            enable_async_alloc: false,
        }
    }

    /// Read host memory in GB from /proc/meminfo (MemTotal). Failure => None.
    /// Ref: https://www.kernel.org/doc/html/latest/filesystems/proc.html#meminfo
    pub fn get_host_memory_gb() -> Option<usize> {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in s.lines() {
            if line.starts_with("MemTotal:") {
                let rest = line.trim_start_matches("MemTotal:").trim();
                let kb = rest
                    .split_ascii_whitespace()
                    .next()?
                    .parse::<usize>()
                    .ok()?;
                return Some(kb / (1024 * 1024)); // KB -> GB
            }
        }
        None
    }

    /// PCIe from env only (QDP_PCIE_GEN: 3/4/5 or gen3/gen4/gen5). Unset or bad => Unknown.
    pub fn detect_pcie_generation() -> PCIeGeneration {
        let s = match std::env::var("QDP_PCIE_GEN") {
            Ok(v) => v,
            Err(_) => return PCIeGeneration::Unknown,
        };
        let s = s.trim();
        if s.eq_ignore_ascii_case("3") || s.eq_ignore_ascii_case("gen3") {
            PCIeGeneration::Gen3
        } else if s.eq_ignore_ascii_case("4") || s.eq_ignore_ascii_case("gen4") {
            PCIeGeneration::Gen4
        } else if s.eq_ignore_ascii_case("5") || s.eq_ignore_ascii_case("gen5") {
            PCIeGeneration::Gen5
        } else {
            PCIeGeneration::Unknown
        }
    }

    /// GPU compute capability via cudaDeviceGetAttribute (Major=75, Minor=76). Failure => Unknown.
    /// Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    pub fn detect_compute_capability(device: &Arc<CudaDevice>) -> ComputeCapability {
        let dev = device.ordinal() as i32;
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            if cudaDeviceGetAttribute(&mut major, cudaDevAttrComputeCapabilityMajor, dev)
                != CUDA_SUCCESS
            {
                return ComputeCapability::Unknown;
            }
            if cudaDeviceGetAttribute(&mut minor, cudaDevAttrComputeCapabilityMinor, dev)
                != CUDA_SUCCESS
            {
                return ComputeCapability::Unknown;
            }
        }
        match (major, minor) {
            (9, _) => ComputeCapability::Hopper,
            (8, 9) => ComputeCapability::Ada,
            (8, _) => ComputeCapability::Ampere,
            _ => ComputeCapability::Unknown,
        }
    }

    /// Fill chunk/pool from hardware defaults when unset. Chunk from PCIe/GPU table; pool default 2 (legacy parity), ≤20% host, clamp 1..=16.
    /// Default pool=2 matches legacy PINNED_POOL_SIZE and avoids regression vs baseline (see PR3 comparison docs).
    /// Ref: CUDA Best Practices – overlap requires ≥2 buffers; use QDP_PINNED_POOL_SIZE=4 for tuning when needed.
    pub fn with_hardware_defaults(
        mut self,
        device: &Arc<CudaDevice>,
        host_mem_gb: Option<usize>,
    ) -> Result<Self> {
        let pcie = Self::detect_pcie_generation();
        let gpu = Self::detect_compute_capability(device);
        if self.chunk_size_mb.is_none() {
            self.chunk_size_mb = Some(chunk_default_mb(pcie, gpu));
        }
        if self.pinned_pool_size.is_none() {
            let chunk_mb = self.chunk_size_mb.unwrap();
            let host_mb = host_mem_gb.unwrap_or(16).saturating_mul(1024);
            let max_by_mem = if chunk_mb > 0 {
                ((0.2 * host_mb as f64) / chunk_mb as f64).floor() as usize
            } else {
                16
            };
            // Default 2 for legacy parity and best observed throughput; cap by 20% host memory.
            let pool = (2.min(max_by_mem)).clamp(1, 16);
            self.pinned_pool_size = Some(pool);
        }
        Ok(self)
    }

    /// Validate: chunk 1..=256, pool 1..=16, pinned total ≤ 20% host (host unknown => 16 GB).
    /// Ref: CUDA Best Practices – pinned memory cap (~20% host) for system stability.
    pub fn validate(&self) -> Result<()> {
        let chunk = self.chunk_size_mb.ok_or_else(|| {
            MahoutError::InvalidInput(
                "chunk_size_mb not set; call with_hardware_defaults first".to_string(),
            )
        })?;
        let pool = self.pinned_pool_size.ok_or_else(|| {
            MahoutError::InvalidInput(
                "pinned_pool_size not set; call with_hardware_defaults first".to_string(),
            )
        })?;
        if !(1..=256).contains(&chunk) {
            return Err(MahoutError::InvalidInput(format!(
                "chunk_size_mb must be 1..=256, got {}",
                chunk
            )));
        }
        if !(1..=16).contains(&pool) {
            return Err(MahoutError::InvalidInput(format!(
                "pinned_pool_size must be 1..=16, got {}",
                pool
            )));
        }
        let host_gb = Self::get_host_memory_gb().unwrap_or(16);
        let host_mb = host_gb * 1024;
        let pinned_mb = pool * chunk;
        if pinned_mb as f64 > 0.2 * host_mb as f64 {
            return Err(MahoutError::InvalidInput(format!(
                "pinned memory {} MB exceeds 20% of host {} MB",
                pinned_mb, host_mb
            )));
        }
        Ok(())
    }

    /// Chunk size in elements (f64). Requires chunk_size_mb set.
    pub fn chunk_size_elements(&self) -> Result<usize> {
        let mb = self
            .chunk_size_mb
            .ok_or_else(|| MahoutError::InvalidInput("chunk_size_mb not set".to_string()))?;
        Ok(mb * 1024 * 1024 / std::mem::size_of::<f64>())
    }

    /// Resolved pinned pool size. Requires pinned_pool_size set.
    pub fn pinned_pool_size_resolved(&self) -> Result<usize> {
        self.pinned_pool_size
            .ok_or_else(|| MahoutError::InvalidInput("pinned_pool_size not set".to_string()))
    }

    /// Per-device cached config: build once per (device ordinal), reuse on subsequent calls.
    /// Avoids 200× get_host_memory_gb + 400× cudaDeviceGetAttribute per benchmark run.
    /// Ref: https://doc.rust-lang.org/std/sync/struct.OnceLock.html
    pub fn get_or_build(device: &Arc<CudaDevice>, host_mem_gb: Option<usize>) -> Result<Self> {
        let ordinal = device.ordinal() as i32;
        let cache = CONFIG_CACHE.get_or_init(|| Mutex::new(None));
        let mut guard = cache.lock().map_err(|_| {
            MahoutError::InvalidInput("PipelineConfig cache lock poisoned".to_string())
        })?;
        if let Some(ref pair) = *guard
            && pair.0 == ordinal
        {
            return Ok(pair.1.clone());
        }
        let config = Self::from_env().with_hardware_defaults(device, host_mem_gb)?;
        config.validate()?;
        let to_store = config.clone();
        *guard = Some((ordinal, to_store.clone()));
        Ok(to_store)
    }
}

/// Default chunk size in MB. Uses 8 MB for legacy parity (matches original hardcoded CHUNK_SIZE).
/// Set QDP_CHUNK_SIZE_MB=16 for Gen5 tuning if needed.
/// Ref: PR3 comparison docs; NVIDIA overlap best practices.
fn chunk_default_mb(pcie: PCIeGeneration, gpu: ComputeCapability) -> usize {
    match (pcie, gpu) {
        (PCIeGeneration::Gen5, _) => 8, // legacy parity; use QDP_CHUNK_SIZE_MB=16 for Gen5 tuning
        (PCIeGeneration::Gen4, ComputeCapability::Hopper) => 8, // was 12; 8 for parity, tune via env
        (PCIeGeneration::Gen4, _) => 8,
        (PCIeGeneration::Gen3, _) => 8, // was 4; 8 for parity, tune via env if needed
        (PCIeGeneration::Unknown, _) => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize env-dependent tests so parallel runs don't cross-talk.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_unset() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        unsafe {
            std::env::remove_var("QDP_CHUNK_SIZE_MB");
            std::env::remove_var("QDP_PINNED_POOL_SIZE");
            std::env::remove_var("QDP_PCIE_GEN");
        }
        let c = PipelineConfig::from_env();
        assert!(c.chunk_size_mb.is_none());
        assert!(c.pinned_pool_size.is_none());
    }

    #[test]
    fn from_env_parses() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("QDP_CHUNK_SIZE_MB", "16");
            std::env::set_var("QDP_PINNED_POOL_SIZE", "4");
        }
        let c = PipelineConfig::from_env();
        assert_eq!(c.chunk_size_mb, Some(16));
        assert_eq!(c.pinned_pool_size, Some(4));
        unsafe {
            std::env::remove_var("QDP_CHUNK_SIZE_MB");
            std::env::remove_var("QDP_PINNED_POOL_SIZE");
        }
    }

    #[test]
    fn detect_pcie_env() {
        let _guard = ENV_TEST_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("QDP_PCIE_GEN", "gen4");
        }
        assert_eq!(
            PipelineConfig::detect_pcie_generation(),
            PCIeGeneration::Gen4
        );
        unsafe {
            std::env::set_var("QDP_PCIE_GEN", "5");
        }
        assert_eq!(
            PipelineConfig::detect_pcie_generation(),
            PCIeGeneration::Gen5
        );
        unsafe {
            std::env::remove_var("QDP_PCIE_GEN");
        }
    }

    #[test]
    fn validate_bounds() {
        let c = PipelineConfig {
            chunk_size_mb: Some(8),
            pinned_pool_size: Some(2),
            ..Default::default()
        };
        assert!(c.validate().is_ok());
        let c_bad_chunk = PipelineConfig {
            chunk_size_mb: Some(0),
            ..c.clone()
        };
        assert!(c_bad_chunk.validate().is_err());
        let c_bad_chunk_high = PipelineConfig {
            chunk_size_mb: Some(257),
            ..c.clone()
        };
        assert!(c_bad_chunk_high.validate().is_err());
        let c_bad_pool = PipelineConfig {
            pinned_pool_size: Some(0),
            ..c.clone()
        };
        assert!(c_bad_pool.validate().is_err());
        let c_bad_pool_high = PipelineConfig {
            pinned_pool_size: Some(17),
            ..c.clone()
        };
        assert!(c_bad_pool_high.validate().is_err());
    }

    #[test]
    fn chunk_size_elements_and_pool_resolved() {
        let c = PipelineConfig {
            chunk_size_mb: Some(8),
            pinned_pool_size: Some(2),
            ..Default::default()
        };
        assert_eq!(c.chunk_size_elements().unwrap(), 8 * 1024 * 1024 / 8);
        assert_eq!(c.pinned_pool_size_resolved().unwrap(), 2);
    }
}
