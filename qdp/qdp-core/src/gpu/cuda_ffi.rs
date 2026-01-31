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

//! Centralized CUDA Runtime API FFI declarations.

use std::ffi::c_void;

pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_DEVICE: u32 = 2;
pub(crate) const CUDA_EVENT_DISABLE_TIMING: u32 = 0x02;
pub(crate) const CUDA_EVENT_DEFAULT: u32 = 0x00;

// CUDA error codes
pub(crate) const CUDA_SUCCESS: i32 = 0;
// Note: CUDA_ERROR_NOT_READY may be used in future optimizations for non-blocking event checks
// Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0028
#[allow(dead_code)]
pub(crate) const CUDA_ERROR_NOT_READY: i32 = 34;

// cudaHostRegister flags: pin existing host memory for async DMA (zero-copy ingress).
// Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
#[allow(non_upper_case_globals)]
pub(crate) const cudaHostRegisterDefault: u32 = 0;

unsafe extern "C" {
    pub(crate) fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: u32) -> i32;
    pub(crate) fn cudaFreeHost(ptr: *mut c_void) -> i32;

    /// Pin existing host memory for async H2D (no copy to PinnedHostBuffer).
    /// Call cudaHostUnregister(ptr) when done. Ref: Part M/N, CuPy #3450.
    pub(crate) fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
    pub(crate) fn cudaHostUnregister(ptr: *mut c_void) -> i32;

    pub(crate) fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;

    pub(crate) fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: u32) -> i32;

    pub(crate) fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: u32,
        stream: *mut c_void,
    ) -> i32;

    pub(crate) fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    pub(crate) fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    pub(crate) fn cudaEventDestroy(event: *mut c_void) -> i32;
    pub(crate) fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
    #[allow(dead_code)]
    pub(crate) fn cudaStreamSynchronize(stream: *mut c_void) -> i32;

    /// Non-blocking stream query (N.6).
    ///
    /// Returns CUDA_SUCCESS if all operations in the stream have completed,
    /// CUDA_ERROR_NOT_READY if not. Ref: CUDA §2.3.2.4 Stream Synchronization.
    pub(crate) fn cudaStreamQuery(stream: *mut c_void) -> i32;

    pub(crate) fn cudaMemsetAsync(
        devPtr: *mut c_void,
        value: i32,
        count: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Non-blocking event query
    ///
    /// Returns CUDA_SUCCESS if the event has completed, CUDA_ERROR_NOT_READY if not.
    /// Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    ///
    /// Note: May be used in future optimizations for non-blocking event checks to reduce
    /// synchronization overhead in pipeline operations.
    #[allow(dead_code)]
    pub(crate) fn cudaEventQuery(event: *mut c_void) -> i32;

    /// Blocking event synchronization
    ///
    /// Waits until the completion of all work currently captured in the event.
    /// Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    pub(crate) fn cudaEventSynchronize(event: *mut c_void) -> i32;

    /// Calculate elapsed time between two events (in milliseconds)
    ///
    /// Both events must have been created with CUDA_EVENT_DEFAULT flag.
    /// Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    pub(crate) fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;

    /// Query device attribute. Returns CUDA_SUCCESS (0) on success.
    /// Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    pub(crate) fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

// Device attributes for compute capability (cudaDeviceAttr; keep CUDA naming).
// Ref: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html (cudaDeviceAttr 75, 76)
#[allow(non_upper_case_globals)]
pub(crate) const cudaDevAttrComputeCapabilityMajor: i32 = 75;
#[allow(non_upper_case_globals)]
pub(crate) const cudaDevAttrComputeCapabilityMinor: i32 = 76;
