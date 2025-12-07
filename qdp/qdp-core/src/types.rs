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

//! Quantum computing type system
//!
//! Defines traits and types for quantum floating-point operations,
//! supporting both f32 (single precision) and f64 (double precision).

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use qdp_kernels::{CuFloatComplex, CuDoubleComplex};
use std::ffi::c_void;

/// Abstract quantum floating-point type
///
/// This trait defines the requirements for floating-point types
/// used in quantum state encoding. Currently supports f32 and f64.
pub trait QuantumFloat:
    Copy + Send + Sync + 'static +
    DeviceRepr + ValidAsZeroBits +
    num_traits::Float + std::fmt::Debug
{
    /// Associated complex number type
    type Complex: DeviceRepr + ValidAsZeroBits + Copy + Send + Sync + 'static;

    /// Human-readable type name for debugging
    const TYPE_NAME: &'static str;

    /// DLPack Type Code (bits)
    const DLPACK_BITS: u8;

    /// Size in bytes
    fn size_bytes() -> usize {
        std::mem::size_of::<Self>()
    }

    // === Kernel Dispatch ===
    // Let concrete types decide which C function to call
    unsafe fn launch_batch_encode(
        input: *const Self,
        output: *mut c_void,
        norms: *const Self,
        batch_size: usize,
        input_dim: usize,
        state_dim: usize,
        stream: *mut c_void,
    ) -> i32;
}

impl QuantumFloat for f32 {
    type Complex = CuFloatComplex;
    const TYPE_NAME: &'static str = "f32";
    const DLPACK_BITS: u8 = 64; // Complex64 (2 * 32)

    unsafe fn launch_batch_encode(
        input: *const Self,
        output: *mut c_void,
        norms: *const Self,
        batch_size: usize,
        input_dim: usize,
        state_dim: usize,
        stream: *mut c_void,
    ) -> i32 {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                qdp_kernels::launch_batch_encode_f32(input, output, norms, batch_size, input_dim, state_dim, stream)
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (input, output, norms, batch_size, input_dim, state_dim, stream);
            999 // Error: CUDA unavailable
        }
    }
}

impl QuantumFloat for f64 {
    type Complex = CuDoubleComplex;
    const TYPE_NAME: &'static str = "f64";
    const DLPACK_BITS: u8 = 128; // Complex128 (2 * 64)

    unsafe fn launch_batch_encode(
        input: *const Self,
        output: *mut c_void,
        norms: *const Self,
        batch_size: usize,
        input_dim: usize,
        state_dim: usize,
        stream: *mut c_void,
    ) -> i32 {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                qdp_kernels::launch_batch_encode_f64(input, output, norms, batch_size, input_dim, state_dim, stream)
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (input, output, norms, batch_size, input_dim, state_dim, stream);
            999 // Error: CUDA unavailable
        }
    }
}

/// Complex number type corresponding to QuantumFloat
///
/// Maps f32 -> cuFloatComplex, f64 -> cuDoubleComplex
pub trait QuantumComplex: Copy + Send + Sync + 'static + DeviceRepr + ValidAsZeroBits {
    type Real: QuantumFloat;
    const TYPE_NAME: &'static str;
}

// Note: Actual complex types are defined in qdp-kernels
// This trait is for future extensibility
