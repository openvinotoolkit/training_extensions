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

template <typename scalar_t>
__device__ void fakeQuantize(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ scale,
        const scalar_t level_low,
        const scalar_t level_high
        ) {
    scalar_t s = level_high / (*scale);
    (*output) = round(min(max((*input) * s, level_low), level_high)) / s;
}

template <typename scalar_t>
__global__ void q_cuda_forward_kernel(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ scale,
        const scalar_t level_low,
        const scalar_t level_high,
        const int64_t size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        fakeQuantize<scalar_t>((output + idx), (input + idx), scale, level_low, level_high);
    }
}

template <typename scalar_t>
__device__ void calcGrad(
        scalar_t* __restrict__ val_grad_input,
        scalar_t* __restrict__ val_grad_scale,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ output,
        const scalar_t scale_low,
        const scalar_t scale_high,
        const scalar_t rscale,
        const scalar_t val_low_grad) {
    *val_grad_scale = 0;
    *val_grad_input = 0;

    if ((*input) < scale_low) {
        (*val_grad_scale) = val_low_grad * (*grad_output);
    } else if ((*input) > scale_high) {
        (*val_grad_scale) = (*grad_output);
    } else {
        (*val_grad_scale) = (*grad_output) * (((*output) - (*input)) * rscale);
        (*val_grad_input) = (*grad_output);
    }
}  

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
__global__ void q_cuda_backward_kernel(
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_scale,
        scalar_t* __restrict__ dev_tmp,
        int* __restrict__ dev_last_block_counter,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ scale,
        const scalar_t level_low,
        const scalar_t level_high,
        const int64_t size) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int gtidx = bidx * CUDA_NUM_THREADS + tidx;
    const int grid_size = CUDA_NUM_THREADS * gridDim.x;

    scalar_t sum = 0;
    scalar_t output, val_grad_scale;
    scalar_t alpha = level_low / level_high;
    scalar_t scale_low = (*scale) * alpha;
    scalar_t scale_high = (*scale);
    scalar_t rscale = 1 / (*scale);
    for (int i = gtidx; i < size; i += grid_size) {
        fakeQuantize<scalar_t>(&output, (input + i), scale, level_low, level_high);
        calcGrad<scalar_t>((grad_input + i), &val_grad_scale, (grad_output + i),
                 (input + i), &output, scale_low, scale_high, rscale, alpha);
        sum += val_grad_scale;
    }

    __shared__ scalar_t sh_grad_scale[CUDA_NUM_THREADS];
    sh_grad_scale[tidx] = sum;

    __syncthreads();
    sum_warp(sh_grad_scale + (tidx & ~(CUDA_WARP_SIZE - 1)));

    __syncthreads();
    if (tidx < CUDA_WARP_SIZE) {
        sh_grad_scale[tidx] = tidx * CUDA_WARP_SIZE < CUDA_NUM_THREADS ? sh_grad_scale[tidx * CUDA_WARP_SIZE] : 0;
        sum_warp(sh_grad_scale);
        if (tidx == 0) {
            dev_tmp[bidx] = sh_grad_scale[0];
        }
    }

    if (last_block(dev_last_block_counter)) {
        sh_grad_scale[tidx] = tidx < gridDim.x ? dev_tmp[tidx] : 0;

        __syncthreads();
        sum_warp(sh_grad_scale + (tidx & ~(CUDA_WARP_SIZE - 1)));

        __syncthreads();
        if (tidx < CUDA_WARP_SIZE) {
            sh_grad_scale[tidx] = tidx * CUDA_WARP_SIZE < CUDA_NUM_THREADS ? sh_grad_scale[tidx * CUDA_WARP_SIZE] : 0;
            sum_warp(sh_grad_scale);
            if (tidx == 0) {
                grad_scale[0] = sh_grad_scale[0];
            }
        }
    }
}
}

at::Tensor qs_cuda_forward(
        at::Tensor input,
        at::Tensor scale,
        int level_low,
        int level_high) {
    const auto size = input.numel();
    auto output = at::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "qs_cuda_forward", ([&] {
      q_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          output.data<scalar_t>(),
          input.data<scalar_t>(),
          scale.data<scalar_t>(),
          level_low,
          level_high,
          size);
    }));

    return output;
}

std::vector<at::Tensor> qs_cuda_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        int level_low,
        int level_high) {
    const auto size = input.numel();
    auto grad_input = at::empty_like(grad_output);
    auto grad_scale = at::empty({1}, grad_output.options());

    auto grid_size = std::min(GET_BLOCKS(size), CUDA_GRID_SIZE);
    auto dev_tmp = at::empty({grid_size}, grad_output.options());
    auto dev_last_block_counter = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "qs_cuda_backward", ([&] {
      q_cuda_backward_kernel<scalar_t><<<grid_size, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          grad_input.data<scalar_t>(),
          grad_scale.data<scalar_t>(),
          dev_tmp.data<scalar_t>(),
          dev_last_block_counter.data<int>(),
          grad_output.data<scalar_t>(),
          input.data<scalar_t>(),
          scale.data<scalar_t>(),
          level_low,
          level_high,
          size);
    }));

    return {grad_input, grad_scale};
}
