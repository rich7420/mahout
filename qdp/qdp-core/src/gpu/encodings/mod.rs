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

// Quantum encoding strategies (Strategy Pattern)

use std::sync::Arc;
use arrow::array::Float64Array;
use cudarc::driver::CudaDevice;
use crate::error::Result;
use crate::gpu::memory::GpuStateVector;
use crate::gpu::pool::StagingBufferPool;
use crate::types::QuantumFloat;

/// Quantum encoding strategy interface (Generic version)
/// Implemented by: AmplitudeEncoder, AngleEncoder, BasisEncoder
///
/// This is the new generic trait that supports both f32 and f64.
pub trait QuantumEncoder<T: QuantumFloat>: Send + Sync {
    /// Encode classical data to quantum state on GPU
    fn encode(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        data: &[T],
        num_qubits: usize,
    ) -> Result<GpuStateVector<T>>;

    /// Encode from chunked Arrow arrays
    ///
    /// Takes Float64Arrays (standard Parquet read) and converts to T if necessary.
    fn encode_chunked(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        chunks: &[Float64Array],
        num_qubits: usize,
    ) -> Result<GpuStateVector<T>> {
        // 1. Flatten Arrow arrays to Vec<f64>
        let raw_data = crate::io::arrow_to_vec_chunked(chunks);

        // 2. Convert to Vec<T> (f64 -> f32 conversion happens here if needed)
        // This is a CPU copy, but necessary when reading standard Parquet into f32 engine.
        // Since T is QuantumFloat (f32 or f64), we use cast.
        let data: Vec<T> = raw_data.iter()
            .map(|&x| num_traits::cast::cast(x).expect("Cast failed"))
            .collect();

        // 3. Delegate to standard encode (which handles GPU transfer & kernel)
        self.encode(device, pool, &data, num_qubits)
    }

    /// Strategy name
    fn name(&self) -> &'static str;

    /// Strategy description
    fn description(&self) -> &'static str;
}

/// Legacy trait for backward compatibility (returns f64)
/// This will be deprecated in favor of the generic QuantumEncoder<T>
pub trait QuantumEncoderLegacy: Send + Sync {
    /// Encode classical data to quantum state on GPU
    fn encode(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<GpuStateVector<f64>>;

    /// Encode from chunked Arrow arrays
    ///
    /// Default implementation flattens chunks. (TODO: Encoders can override for true zero-copy.)
    fn encode_chunked(
        &self,
        device: &Arc<CudaDevice>,
        pool: &Arc<StagingBufferPool>,
        chunks: &[Float64Array],
        num_qubits: usize,
    ) -> Result<GpuStateVector<f64>> {
        // Default: flatten and use regular encode
        let data = crate::io::arrow_to_vec_chunked(chunks);
        self.encode(device, pool, &data, num_qubits)
    }

    /// Strategy name
    fn name(&self) -> &'static str;

    /// Strategy description
    fn description(&self) -> &'static str;
}

// Encoding implementations
pub mod amplitude;
pub mod angle;
pub mod basis;

pub use amplitude::AmplitudeEncoder;
pub use angle::AngleEncoder;
pub use basis::BasisEncoder;

/// Create encoder by name: "amplitude", "angle", or "basis" (Legacy, returns f64 encoder)
///
/// Note: This is kept for backward compatibility. New code should use the generic
/// QuantumEncoder<T> trait directly.
pub fn get_encoder(name: &str) -> Result<Box<dyn QuantumEncoderLegacy>> {
    match name.to_lowercase().as_str() {
        "amplitude" => Ok(Box::new(AmplitudeEncoder)),
        "angle" => Ok(Box::new(AngleEncoder)),
        "basis" => Ok(Box::new(BasisEncoder)),
        _ => Err(crate::error::MahoutError::InvalidInput(
            format!("Unknown encoder: {}. Available: amplitude, angle, basis", name)
        )),
    }
}

/// Generic factory for T
pub fn get_encoder_generic<T: QuantumFloat>(name: &str) -> Result<Box<dyn QuantumEncoder<T>>> {
    match name.to_lowercase().as_str() {
        "amplitude" => Ok(Box::new(AmplitudeEncoder)),
        // Angle/Basis not implemented for generic yet, fallback or error
        "angle" => Err(crate::error::MahoutError::InvalidInput(
            "Angle encoding not supported for generic API yet".into()
        )),
        "basis" => Err(crate::error::MahoutError::InvalidInput(
            "Basis encoding not supported for generic API yet".into()
        )),
        _ => Err(crate::error::MahoutError::InvalidInput(
            format!("Unknown encoder: {}. Available: amplitude", name)
        )),
    }
}
