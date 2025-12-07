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

// DLPack protocol for zero-copy GPU memory sharing with PyTorch

use std::os::raw::{c_int, c_void};
use std::sync::Arc;
use crate::gpu::memory::GpuStateVector;
use crate::types::QuantumFloat;

// DLPack C structures (matching dlpack/dlpack.h)

#[repr(C)]
#[allow(non_camel_case_types)]
pub enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
    // Other types omitted
}

#[repr(C)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: c_int,
}

#[repr(C)]
pub struct DLDataType {
    pub code: u8,  // kDLInt=0, kDLUInt=1, kDLFloat=2, kDLBfloat=4, kDLComplex=5
    pub bits: u8,
    pub lanes: u16,
}

// DLPack data type codes (PyTorch 2.2+)
#[allow(dead_code)]
pub const DL_INT: u8 = 0;
#[allow(dead_code)]
pub const DL_UINT: u8 = 1;
#[allow(dead_code)]
pub const DL_FLOAT: u8 = 2;
#[allow(dead_code)]
pub const DL_BFLOAT: u8 = 4;
pub const DL_COMPLEX: u8 = 5;

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: c_int,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

// Deleter: frees memory when PyTorch is done

/// Called by PyTorch to free tensor memory (f32 version)
///
/// # Safety
/// Frees shape, strides, GPU buffer, and managed tensor.
/// Caller must ensure the pointer is valid and points to a properly initialized DLManagedTensor.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn dlpack_deleter_f32(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }

    let tensor = &(*managed).dl_tensor;

    // 1. Free shape array (Box<[i64]>)
    if !tensor.shape.is_null() {
        let len = if tensor.ndim > 0 { tensor.ndim as usize } else { 1 };
        let slice_ptr: *mut [i64] = std::ptr::slice_from_raw_parts_mut(tensor.shape, len);
        let _ = Box::from_raw(slice_ptr);
    }

    // 2. Free strides array
    if !tensor.strides.is_null() {
        let len = if tensor.ndim > 0 { tensor.ndim as usize } else { 1 };
        let slice_ptr: *mut [i64] = std::ptr::slice_from_raw_parts_mut(tensor.strides, len);
        let _ = Box::from_raw(slice_ptr);
    }

    // 3. Free GPU buffer (Arc reference count)
    let ctx = (*managed).manager_ctx;
    if !ctx.is_null() {
        let _ = Arc::from_raw(ctx as *const crate::gpu::memory::GpuBufferRaw<f32>);
    }

    // 4. Free DLManagedTensor
    let _ = Box::from_raw(managed);
}

/// Called by PyTorch to free tensor memory (f64 version)
///
/// # Safety
/// Frees shape, strides, GPU buffer, and managed tensor.
/// Caller must ensure the pointer is valid and points to a properly initialized DLManagedTensor.
#[allow(unsafe_op_in_unsafe_fn)]
unsafe extern "C" fn dlpack_deleter_f64(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }

    let tensor = &(*managed).dl_tensor;

    // 1. Free shape array (Box<[i64]>)
    if !tensor.shape.is_null() {
        let len = if tensor.ndim > 0 { tensor.ndim as usize } else { 1 };
        let slice_ptr: *mut [i64] = std::ptr::slice_from_raw_parts_mut(tensor.shape, len);
        let _ = Box::from_raw(slice_ptr);
    }

    // 2. Free strides array
    if !tensor.strides.is_null() {
        let len = if tensor.ndim > 0 { tensor.ndim as usize } else { 1 };
        let slice_ptr: *mut [i64] = std::ptr::slice_from_raw_parts_mut(tensor.strides, len);
        let _ = Box::from_raw(slice_ptr);
    }

    // 3. Free GPU buffer (Arc reference count)
    let ctx = (*managed).manager_ctx;
    if !ctx.is_null() {
        let _ = Arc::from_raw(ctx as *const crate::gpu::memory::GpuBufferRaw<f64>);
    }

    // 4. Free DLManagedTensor
    let _ = Box::from_raw(managed);
}

impl<T: QuantumFloat> GpuStateVector<T> {
    /// Convert to DLPack format for PyTorch
    ///
    /// Returns raw pointer for torch.from_dlpack() (zero-copy, GPU memory).
    ///
    /// # Safety
    /// Freed by DLPack deleter when PyTorch releases tensor.
    /// Do not free manually.
    pub fn to_dlpack(&self) -> *mut DLManagedTensor {
        // 1. Setup Shape
        let shape = vec![self.size_elements as i64];
        let strides = vec![1i64];

        // Transfer ownership to DLPack deleter
        let shape_ptr = Box::into_raw(shape.into_boxed_slice()) as *mut i64;
        let strides_ptr = Box::into_raw(strides.into_boxed_slice()) as *mut i64;

        // 2. Setup Context
        let ctx = Arc::into_raw(self.buffer.clone()) as *mut c_void;

        // 3. Select Deleter based on type
        let deleter = if T::TYPE_NAME == "f32" {
            Some(dlpack_deleter_f32 as unsafe extern "C" fn(*mut DLManagedTensor))
        } else {
            Some(dlpack_deleter_f64 as unsafe extern "C" fn(*mut DLManagedTensor))
        };

        // 4. Construct Tensor
        let tensor = DLTensor {
            data: self.ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::kDLCUDA,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: DL_COMPLEX,  // Complex
                bits: T::DLPACK_BITS, // 64 for f32 (complex64), 128 for f64 (complex128)
                lanes: 1,
            },
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        };

        let managed = DLManagedTensor {
            dl_tensor: tensor,
            manager_ctx: ctx,
            deleter,
        };

        Box::into_raw(Box::new(managed))
    }
}
