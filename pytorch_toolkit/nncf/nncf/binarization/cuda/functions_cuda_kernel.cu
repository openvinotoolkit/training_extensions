#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <THC/THC.h>

extern THCState *state;

const int CUDA_WARP_SIZE = 32;
const int CUDA_GRID_SIZE = 56; // #SM*2
const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

namespace {

__device__ bool last_block(int* counter) {
    __threadfence();

    int last = 0;
    if (threadIdx.x == 0) {
        last = atomicAdd(counter, 1);
    }

    return __syncthreads_or(last == gridDim.x - 1);
}

// support only warp size = 32
template <typename scalar_t>
__device__ void sum_warp(volatile scalar_t* sharr) {
    int tidx = threadIdx.x & 31;
    if (tidx < 16) {
        sharr[tidx] += sharr[tidx + 16];
        sharr[tidx] += sharr[tidx + 8];
        sharr[tidx] += sharr[tidx + 4];
        sharr[tidx] += sharr[tidx + 2];
        sharr[tidx] += sharr[tidx + 1];
    }
}

template <typename scalar_t>
__device__ void sumReduce(
        scalar_t* __restrict__ sh_grad,
        scalar_t sum,
        const int tidx,
        const int bidx,
        scalar_t* __restrict__ dev_tmp,
        int* __restrict__ dev_last_block_counter,
        scalar_t* __restrict__ grad) {
    sh_grad[tidx] = sum;

    __syncthreads();
    sum_warp(sh_grad + (tidx & ~(CUDA_WARP_SIZE - 1)));

    __syncthreads();
    if (tidx < CUDA_WARP_SIZE) {
        sh_grad[tidx] = tidx * CUDA_WARP_SIZE < CUDA_NUM_THREADS ? sh_grad[tidx * CUDA_WARP_SIZE] : 0;
        sum_warp(sh_grad);
        if (tidx == 0) {
            dev_tmp[bidx] = sh_grad[0];
        }
    }

    if (last_block(dev_last_block_counter)) {
        sh_grad[tidx] = tidx < gridDim.x ? dev_tmp[tidx] : 0;

        __syncthreads();
        sum_warp(sh_grad + (tidx & ~(CUDA_WARP_SIZE - 1)));

        __syncthreads();
        if (tidx < CUDA_WARP_SIZE) {
            sh_grad[tidx] = tidx * CUDA_WARP_SIZE < CUDA_NUM_THREADS ? sh_grad[tidx * CUDA_WARP_SIZE] : 0;
            sum_warp(sh_grad);
            if (tidx == 0) {
                grad[0] = sh_grad[0];
            }
        }
    }
}


template <typename scalar_t>
__global__ void wb_cuda_scale_calc_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ scale_output,
        scalar_t* __restrict__ dev_tmp,
        int* __restrict__ dev_last_block_counter,
        const int64_t total_elements_count) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int gtidx = bidx * CUDA_NUM_THREADS + tidx;
    const int grid_size = CUDA_NUM_THREADS * gridDim.x;

    scalar_t sum = 0;
    for (int i = gtidx; i < total_elements_count; i += grid_size) {
        sum += abs(*(input + i));
    }

    sum /= total_elements_count;

    __shared__ scalar_t sh_mem[CUDA_NUM_THREADS];
    sumReduce<scalar_t>(sh_mem, sum, tidx, bidx, dev_tmp, dev_last_block_counter, scale_output);
}

template <typename scalar_t>
__global__ void wb_cuda_binarize_kernel(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ scale,
        const int64_t scale_count,
        const int64_t elements_per_scale,
        const int64_t size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64_t scale_idx = static_cast<int64_t>(idx / elements_per_scale) % scale_count;
        scalar_t scale_element = *(scale + scale_idx);
        *(output + idx) = (*(input + idx) > 0) ? scale_element : -scale_element;
    }
}


template <typename scalar_t>
__global__ void ab_cuda_forward_kernel(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ scale,
        const scalar_t* __restrict__ thresholds,
        const int64_t threshold_count,
        const int64_t contiguous_elements_per_threshold,
        const int64_t size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int64_t threshold_idx = static_cast<int64_t>(idx / contiguous_elements_per_threshold) % threshold_count;
        scalar_t threshold_element = (*(thresholds + threshold_idx)) * (*scale);
        *(output + idx) = (*(input + idx) > threshold_element) ? (*scale) : 0;
    }
}


template <typename scalar_t>
__global__ void ab_cuda_grad_input_kernel(
        scalar_t* __restrict__ grad_input,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ scale,
        const int64_t size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const scalar_t input_element = *(input + idx);
        *(grad_input + idx) = (input_element > 0 && input_element < *scale) ? *(grad_output + idx) : 0;
    }
}

template <typename scalar_t>
__global__ void ab_cuda_grad_scale_kernel(
        scalar_t* __restrict__ grad_scale,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ scale,
        scalar_t* __restrict__ dev_tmp,
        int* __restrict__ dev_last_block_counter,
        const int64_t total_elements_count) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int gtidx = bidx * CUDA_NUM_THREADS + tidx;
    const int grid_size = CUDA_NUM_THREADS * gridDim.x;

    scalar_t sum = 0;
    for (int i = gtidx; i < total_elements_count; i += grid_size) {
        scalar_t err_element = (*(output + i) - *(input + i)) / *scale;
        scalar_t grad_element = *(grad_output + i);
        sum += (*(input + i) < *scale) ? err_element * grad_element : grad_element;
    }

    __shared__ scalar_t sh_mem[CUDA_NUM_THREADS];
    sumReduce<scalar_t>(sh_mem, sum, tidx, bidx, dev_tmp, dev_last_block_counter, grad_scale);
}

template <typename scalar_t>
__global__ void ab_cuda_grad_thresholds_kernel(
        scalar_t* __restrict__ grad_thresholds,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ scale,
        scalar_t* __restrict__ dev_tmp,
        int* __restrict__ dev_last_block_counter,
        int64_t total_elements_per_threshold,
        int64_t contiguous_elements_per_threshold,
        int64_t threshold_count,
        int64_t channel_offset) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int gtidx = bidx * CUDA_NUM_THREADS + tidx;
    const int grid_size = CUDA_NUM_THREADS * gridDim.x;

    scalar_t sum = 0;
    for (int i = gtidx; i < total_elements_per_threshold; i += grid_size) {
        // i is the global thread index - need to calculate the input array index
        // that belongs to a specific scale index from i. Will do this by treating i
        // as the index in a non-existing array where input values belonging to a single
        // scale have a contiguous block layout, but will recalculate actual index into the
        // input/output array based on the fact that the values belonging to a single scale
        // in reality have interleaved block layout, with a spacing between the blocks
        // equal to channel_offset
        int actual_idx = (i / contiguous_elements_per_threshold) * channel_offset + (i % contiguous_elements_per_threshold);
        scalar_t input_element = *(input + actual_idx);
        if (input_element < *scale && input_element > 0)
        {
            sum += -*(grad_output + actual_idx);
        }
    }

    __shared__ scalar_t sh_mem[CUDA_NUM_THREADS];
    sumReduce<scalar_t>(sh_mem, sum, tidx, bidx, dev_tmp, dev_last_block_counter, grad_thresholds);
}

}

at::Tensor wb_cuda_forward(
        at::Tensor input,
        bool per_channel) {
    const auto quantized_elements_count = input.numel();

    int64_t elements_per_scale = 0;
    int64_t scale_count = per_channel ? input.size(0) : 1;
    int64_t input_elements_count = input.numel();

    auto scale = at::zeros({scale_count}, input.options());
    elements_per_scale = input_elements_count / input.size(0);

    auto grid_size = std::min(GET_BLOCKS(elements_per_scale), CUDA_GRID_SIZE);
    auto dev_tmp = at::empty({grid_size}, input.options());
    auto dev_last_block_counter = at::zeros({1},  at::device(input.options().device()).dtype(at::kInt));


    auto output = at::empty_like(input);

    for (int ch_idx = 0; ch_idx < scale_count; ch_idx++)
    {
        AT_DISPATCH_FLOATING_TYPES(input.type(), "wb_cuda_forward_scale", ([&] {
          wb_cuda_scale_calc_kernel<scalar_t><<<grid_size, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
              input.data<scalar_t>() + ch_idx * elements_per_scale,
              scale.data<scalar_t>() + ch_idx,
              dev_tmp.data<scalar_t>(),
              dev_last_block_counter.data<int>(),
              elements_per_scale);
        }));
        dev_tmp.fill_(0.0);
        dev_last_block_counter.fill_(0);
    }

    AT_DISPATCH_FLOATING_TYPES(input.type(), "wb_cuda_forward_binarize", ([&] {
      wb_cuda_binarize_kernel<scalar_t><<<GET_BLOCKS(input_elements_count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          output.data<scalar_t>(),
          input.data<scalar_t>(),
          scale.data<scalar_t>(),
          scale_count,
          elements_per_scale,
          input_elements_count
          );
    }));

    return output;
}


at::Tensor ab_cuda_forward(
        at::Tensor input,
        at::Tensor scale,
        at::Tensor thresholds) {
    const auto quantized_elements_count = input.numel();

    int64_t input_elements_count = input.numel();
    int64_t threshold_count = thresholds.numel();
    AT_CHECK(input.size(1) == threshold_count, "Threshold count is not equal to activations channel count");
    int64_t contiguous_elements_per_threshold = input_elements_count / input.size(0) / input.size(1);

    auto output = at::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "ab_cuda_forward", ([&] {
      ab_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(input_elements_count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          output.data<scalar_t>(),
          input.data<scalar_t>(),
          scale.data<scalar_t>(),
          thresholds.data<scalar_t>(),
          threshold_count,
          contiguous_elements_per_threshold,
          input_elements_count
          );
    }));

    return output;
}


std::vector<at::Tensor> ab_cuda_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        at::Tensor output) {
    int64_t input_elements_count = input.numel();
    int64_t threshold_count = input.size(1);
    int64_t channel_offset = input.numel() / input.size(0);

    std::vector<int64_t> threshold_shape(input.dim());
    for (int64_t dim_idx = 0; dim_idx < input.dim(); dim_idx++)
    {
        if (dim_idx != 1)
        {
            threshold_shape[dim_idx] = 1;
        }
        else
        {
            threshold_shape[dim_idx] = input.size(dim_idx);
        }
    }

    auto grad_input = at::empty_like(input);
    auto grad_scale = at::empty_like(scale);
    auto grad_thresholds = at::empty(threshold_shape, input.options());

    int64_t total_elements_per_threshold = input.numel() / threshold_count;
    int64_t contiguous_elements_per_threshold = input_elements_count / input.size(0) / input.size(1);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "ab_cuda_backward", ([&] {
      ab_cuda_grad_input_kernel<scalar_t><<<GET_BLOCKS(input_elements_count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          grad_input.data<scalar_t>(),
          grad_output.data<scalar_t>(),
          input.data<scalar_t>(),
          scale.data<scalar_t>(),
          input_elements_count
          );
    }));

    auto grid_size = std::min(GET_BLOCKS(input_elements_count), CUDA_GRID_SIZE);
    auto dev_tmp = at::empty({grid_size}, grad_output.options());
    auto dev_last_block_counter = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));


    AT_DISPATCH_FLOATING_TYPES(input.type(), "ab_cuda_backward", ([&] {
          ab_cuda_grad_scale_kernel<scalar_t><<<grid_size, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
              grad_scale.data<scalar_t>(),
              grad_output.data<scalar_t>(),
              output.data<scalar_t>(),
              input.data<scalar_t>(),
              scale.data<scalar_t>(),
              dev_tmp.data<scalar_t>(),
              dev_last_block_counter.data<int>(),
              input_elements_count);
        }));

    grid_size = std::min(GET_BLOCKS(total_elements_per_threshold), CUDA_GRID_SIZE);
    dev_tmp = at::empty({grid_size}, grad_output.options());
    dev_last_block_counter = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));

    // Same concept as for per activation channel quantization
    for (int64_t ch_idx = 0; ch_idx < threshold_count; ch_idx++)
    {
        auto init_element_offset = contiguous_elements_per_threshold * ch_idx;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "ab_cuda_backward", ([&] {
          ab_cuda_grad_thresholds_kernel<scalar_t><<<grid_size, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
              grad_thresholds.data<scalar_t>() + ch_idx,
              grad_output.data<scalar_t>() + init_element_offset,
              input.data<scalar_t>() + init_element_offset,
              scale.data<scalar_t>(),
              dev_tmp.data<scalar_t>(),
              dev_last_block_counter.data<int>(),
              total_elements_per_threshold,
              contiguous_elements_per_threshold,
              threshold_count,
              channel_offset);
        }));
        dev_tmp.fill_(0.0);
        dev_last_block_counter.fill_(0);
    }
    return {grad_input, grad_scale, grad_thresholds};
}

