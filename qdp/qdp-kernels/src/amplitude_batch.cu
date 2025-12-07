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

#include <cuda_runtime.h>
#include <cuComplex.h>

// Define generic complex type
template<typename T> struct CuComplexType;
template<> struct CuComplexType<float> { using type = cuFloatComplex; };
template<> struct CuComplexType<double> { using type = cuDoubleComplex; };

// Helper function: create complex number
__device__ __forceinline__ cuFloatComplex make_complex(float x, float y) {
    return make_cuFloatComplex(x, y);
}

__device__ __forceinline__ cuDoubleComplex make_complex(double x, double y) {
    return make_cuDoubleComplex(x, y);
}

/**
 * Batch Amplitude Encoding Kernel
 *
 * Assumptions:
 * 1. Grid.y represents Batch Index (which vector)
 * 2. Grid.x / Block.x handles elements within that vector
 * 3. Each vector independently computes Norm and writes
 */
template<typename T>
__global__ void batch_amplitude_encode_kernel(
    const T* __restrict__ input_flat,      // [Batch * Input_Dim]
    typename CuComplexType<T>::type* output_flat, // [Batch * State_Dim]
    size_t input_dim,
    size_t state_dim,
    const T* __restrict__ precomputed_norms // Precomputed norms
) {
    // Which vector we're currently processing
    size_t batch_idx = blockIdx.y;

    // Calculate the vector's offset in global memory
    size_t input_offset = batch_idx * input_dim;
    size_t output_offset = batch_idx * state_dim;

    // Get the vector's Norm
    T norm = precomputed_norms ? precomputed_norms[batch_idx] : (T)1.0;
    T inv_norm = (norm > (T)1e-9) ? (T)1.0 / norm : (T)0.0;

    // Grid-Stride Loop to process elements
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < state_dim; i += gridDim.x * blockDim.x) {
        T val = (T)0.0;

        // Only has value within input_dim range, otherwise pad with 0
        if (i < input_dim) {
            val = input_flat[input_offset + i];
        }

        // Write normalized complex (real=val*inv_norm, imag=0)
        output_flat[output_offset + i] = make_complex(val * inv_norm, (T)0.0);
    }
}

// Explicit template instantiation
extern "C" {

// Float32 entry point
int launch_batch_encode_f32(
    const float* input,
    void* output,
    const float* norms,
    size_t batch_size,
    size_t input_dim,
    size_t state_dim,
    cudaStream_t stream
) {
    dim3 block(256);
    // Grid Y = Batch Size (each Block Y processes one vector)
    // Grid X = dynamically adjusted based on vector length
    size_t needed_blocks_x = (state_dim + 255) / 256;
    size_t max_blocks_x = 128; // Limit Grid X size
    dim3 grid((needed_blocks_x < max_blocks_x) ? needed_blocks_x : max_blocks_x, batch_size);

    batch_amplitude_encode_kernel<float><<<grid, block, 0, stream>>>(
        input, (cuFloatComplex*)output, input_dim, state_dim, norms
    );
    return (int)cudaGetLastError();
}

// Float64 entry point
int launch_batch_encode_f64(
    const double* input,
    void* output,
    const double* norms,
    size_t batch_size,
    size_t input_dim,
    size_t state_dim,
    cudaStream_t stream
) {
    dim3 block(256);
    size_t needed_blocks_x = (state_dim + 255) / 256;
    size_t max_blocks_x = 128;
    dim3 grid((needed_blocks_x < max_blocks_x) ? needed_blocks_x : max_blocks_x, batch_size);

    batch_amplitude_encode_kernel<double><<<grid, block, 0, stream>>>(
        input, (cuDoubleComplex*)output, input_dim, state_dim, norms
    );
    return (int)cudaGetLastError();
}

}
