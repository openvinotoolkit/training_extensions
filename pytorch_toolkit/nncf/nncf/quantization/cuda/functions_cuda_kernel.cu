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

enum class ScaleType
{
    SINGLE_SCALE,
    PER_WEIGHT_CHANNEL,
    PER_ACTIVATION_CHANNEL
};

ScaleType get_scale_type(const at::Tensor& input, const at::Tensor& input_low, const at::Tensor& input_range)
{
    AT_CHECK(input_low.dim() == input_range.dim(), "input_low and input_range have different dimensionality");
    int64_t scale_dim = input_range.dim();
    for (int i = 0; i < scale_dim; i++)
    {
        AT_CHECK(input_low.size(i) == input_range.size(i), "input_low and input_range have different dimension sizes");
    }

    int64_t scale_count = input_range.numel();

    if (scale_dim > 0)
    {
        // For (NxCxHxW) input/output tensors, it is assumed that input_range is
        // either (1) for single-scale quantization, or (Nx1x1x1) for
        // per-channel scale weights quantization, or (1xCx1x1) for per-channel
        // activation quantization
        if (input_range.size(0) > 1)
        {
            AT_CHECK(input_range.size(0) == input.size(0), "Scale count and weights input channel count is different");
            AT_CHECK(input_range.size(0) == scale_count, "Scale shape is not flat");
            return ScaleType::PER_WEIGHT_CHANNEL;
        }
        else if (scale_dim >= 2 and input_range.size(1) > 1)
        {
            AT_CHECK(input_range.size(1) == input.size(1), "Scale count and activations channel count is different");
            AT_CHECK(input_range.size(1) == scale_count, "Scale shape is not flat");
            return  ScaleType::PER_ACTIVATION_CHANNEL;
        }
    }

    return ScaleType::SINGLE_SCALE;
}

namespace {

template <typename scalar_t>
__device__ void fakeQuantize(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels
        ) {
    scalar_t s = (levels - 1) / (*input_range);
    (*output) = round((min(max((*input), (*input_low)), (*input_low) + (*input_range)) - (*input_low)) * s) / s + (*input_low);
}

template <typename scalar_t>
__global__ void q_cuda_forward_kernel(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const int64_t size,
        const int64_t contiguous_elements_per_scale,
        const int64_t scale_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // "Scales" are derived from input_low/input_range
        int64_t scale_idx = static_cast<int64_t>(idx / contiguous_elements_per_scale) % scale_count;
        fakeQuantize<scalar_t>((output + idx), (input + idx), input_low + scale_idx, input_range + scale_idx, levels);
    }
}

template <typename scalar_t>
__device__ void calcGrad(
        scalar_t* __restrict__ val_grad_input,
        scalar_t* __restrict__ val_grad_input_low,
        scalar_t* __restrict__ val_grad_input_range,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ output,
        const scalar_t range_low,
        const scalar_t range_high,
        const scalar_t reverted_range,
        const scalar_t val_low_grad) {
    *val_grad_input_range = 0;
    *val_grad_input_low = 0;
    *val_grad_input = 0;

    if ((*input) < range_low) {
        (*val_grad_input_range) = val_low_grad * (*grad_output);
        (*val_grad_input_low) = (*grad_output);
    } else if ((*input) > range_high) {
        (*val_grad_input_range) = (*grad_output);
        (*val_grad_input_low) = (*grad_output);
    } else {
        (*val_grad_input_range) = (*grad_output) * (((*output) - (*input)) * reverted_range);
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
__device__ void sumGrad(
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
__global__ void q_scale_per_activation_channel_cuda_backward_kernel(
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_input_low,
        scalar_t* __restrict__ grad_input_range,
        scalar_t* __restrict__ dev_tmp_range,
        scalar_t* __restrict__ dev_tmp_low,
        int* __restrict__ dev_last_block_counter_range,
        int* __restrict__ dev_last_block_counter_low,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const scalar_t level_low,
        const scalar_t level_high,
        const int64_t total_elements_per_scale,
        const int64_t contiguous_elements_per_scale,
        const int64_t scale_count,
        const int64_t channel_offset) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int gtidx = bidx * CUDA_NUM_THREADS + tidx;
    const int grid_size = CUDA_NUM_THREADS * gridDim.x;

    scalar_t sum_range = 0, sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low) + (*input_range) * alpha;
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);

    for (int i = gtidx; i < total_elements_per_scale; i += grid_size) {
        // i is the global thread index - need to calculate the input array index
        // that belongs to a specific scale index from i. Will do this by treating i
        // as the index in a non-existing array where input values belonging to a single
        // scale have a contiguous block layout, but will recalculate actual index into the
        // input/output array based on the fact that the values belonging to a single scale
        // in reality have interleaved block layout, with a spacing between the blocks
        // equal to channel_offset
        int actual_idx = (i / contiguous_elements_per_scale) * channel_offset + (i % contiguous_elements_per_scale);
        fakeQuantize<scalar_t>(&output, (input + actual_idx), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + actual_idx), &val_grad_input_low, &val_grad_input_range, (grad_output + actual_idx),
                 (input + actual_idx), &output, range_low, range_high, reverted_range, alpha);
        sum_range += val_grad_input_range;
        sum_low += val_grad_input_low;
    }

    __shared__ scalar_t sh_grad_range[CUDA_NUM_THREADS];
    __shared__ scalar_t sh_grad_low[CUDA_NUM_THREADS];
    sumGrad<scalar_t>(sh_grad_range, sum_range, tidx, bidx, dev_tmp_range, dev_last_block_counter_range, grad_input_range);
    sumGrad<scalar_t>(sh_grad_low, sum_low, tidx, bidx, dev_tmp_low, dev_last_block_counter_low, grad_input_low);
}


template <typename scalar_t>
__global__ void q_single_scale_cuda_backward_kernel(
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_input_low,
        scalar_t* __restrict__ grad_input_range,
        scalar_t* __restrict__ dev_tmp_range,
        scalar_t* __restrict__ dev_tmp_low,
        int* __restrict__ dev_last_block_counter_range,
        int* __restrict__ dev_last_block_counter_low,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const scalar_t level_low,
        const scalar_t level_high,
        const int64_t size) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int gtidx = bidx * CUDA_NUM_THREADS + tidx;
    const int grid_size = CUDA_NUM_THREADS * gridDim.x;

    scalar_t sum_range = 0, sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low);
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);
    for (int i = gtidx; i < size; i += grid_size) {
        fakeQuantize<scalar_t>(&output, (input + i), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + i), &val_grad_input_low, &val_grad_input_range, (grad_output + i),
                 (input + i), &output, range_low, range_high, reverted_range, alpha);
        sum_range += val_grad_input_range;
        sum_low += val_grad_input_low;
    }

    __shared__ scalar_t sh_grad_range[CUDA_NUM_THREADS];
    __shared__ scalar_t sh_grad_low[CUDA_NUM_THREADS];
    sumGrad<scalar_t>(sh_grad_range, sum_range, tidx, bidx, dev_tmp_range, dev_last_block_counter_range, grad_input_range);
    sumGrad<scalar_t>(sh_grad_low, sum_low, tidx, bidx, dev_tmp_low, dev_last_block_counter_low, grad_input_low);
}

}

at::Tensor q_cuda_forward(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels) {
    const auto quantized_elements_count = input.numel();

    ScaleType scale_type = get_scale_type(input, input_low, input_range);

    int64_t contiguous_elements_per_scale = 0;
    int64_t scale_count = input_range.numel();
    switch (scale_type)
    {
        case ScaleType::PER_ACTIVATION_CHANNEL:
            // Scale count should be equal to 1-st input tensor dimension
            contiguous_elements_per_scale = quantized_elements_count / (input.size(0) * scale_count);
            break;
        case ScaleType::PER_WEIGHT_CHANNEL:
            // Scale count should be equal to 0-th input tensor dimension
            contiguous_elements_per_scale = quantized_elements_count / scale_count;
            break;
        default:
            contiguous_elements_per_scale = quantized_elements_count;
            break;
    }

    auto output = at::empty_like(input);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "q_cuda_forward", ([&] {
      q_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(quantized_elements_count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          output.data<scalar_t>(),
          input.data<scalar_t>(),
          input_low.data<scalar_t>(),
          input_range.data<scalar_t>(),
          levels,
          quantized_elements_count,
          contiguous_elements_per_scale,
          scale_count);
    }));

    return output;
}


std::vector<at::Tensor> q_scale_per_activation_channel_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {

    const auto scale_count = input_range.size(1);
    const auto total_elements_per_scale = input.numel() / scale_count;
    const auto contiguous_elements_per_scale = input.numel() / (scale_count * input.size(0));
    const auto channel_offset = input.numel() / input.size(0);

    auto grad_input = at::empty_like(grad_output);

    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    auto grid_size = std::min(GET_BLOCKS(total_elements_per_scale), CUDA_GRID_SIZE);
    auto dev_tmp_range = at::empty({grid_size}, grad_output.options());
    auto dev_tmp_low = at::empty({grid_size}, grad_output.options());
    auto dev_last_block_counter_range = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));

    // Reusing the single scale backward kernel for now, since in this case the
    // elements that correspond to a single scale value are laid out in memory
    // as contiguous blocks.
    for (int64_t scale_idx = 0; scale_idx < scale_count; scale_idx++)
    {
        auto init_element_offset = contiguous_elements_per_scale * scale_idx;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "q_scale_per_activation_channel_cuda_backward", ([&] {
          q_scale_per_activation_channel_cuda_backward_kernel<scalar_t><<<grid_size, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
              grad_input.data<scalar_t>() + init_element_offset,
              grad_input_low.data<scalar_t>() + scale_idx,
              grad_input_range.data<scalar_t>() + scale_idx,
              dev_tmp_range.data<scalar_t>(),
              dev_tmp_low.data<scalar_t>(),
              dev_last_block_counter_range.data<int>(),
              dev_last_block_counter_low.data<int>(),
              grad_output.data<scalar_t>() + init_element_offset,
              input.data<scalar_t>() + init_element_offset,
              input_low.data<scalar_t>() + scale_idx,
              input_range.data<scalar_t>() + scale_idx,
              levels,
              level_low,
              level_high,
              total_elements_per_scale,
              contiguous_elements_per_scale,
              scale_count,
              channel_offset);
        }));
        dev_tmp_range.fill_(0.0);
        dev_tmp_low.fill_(0.0);
        dev_last_block_counter_low.fill_(0);
        dev_last_block_counter_range.fill_(0);
    }
    return {grad_input, grad_input_low, grad_input_range};
}

std::vector<at::Tensor> q_scale_per_weight_channel_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {

    const auto scale_count = input_range.size(0);
    const auto elements_per_scale = input.numel() / scale_count;

    auto grad_input = at::empty_like(grad_output);

    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    auto grid_size = std::min(GET_BLOCKS(elements_per_scale), CUDA_GRID_SIZE);
    auto dev_tmp_range = at::empty({grid_size}, grad_output.options());
    auto dev_tmp_low = at::empty({grid_size}, grad_output.options());
    auto dev_last_block_counter_range = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));

    // Reusing the single scale backward kernel for now, since in this case the
    // elements that correspond to a single scale value are laid out in memory
    // as contiguous blocks.
    for (int64_t scale_idx = 0; scale_idx < scale_count; scale_idx++)
    {
        auto init_element_offset = elements_per_scale * scale_idx;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "q_single_scale_cuda_backward", ([&] {
          q_single_scale_cuda_backward_kernel<scalar_t><<<grid_size, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
              grad_input.data<scalar_t>() + init_element_offset,
              grad_input_low.data<scalar_t>() + scale_idx,
              grad_input_range.data<scalar_t>() + scale_idx,
              dev_tmp_range.data<scalar_t>(),
              dev_tmp_low.data<scalar_t>(),
              dev_last_block_counter_range.data<int>(),
              dev_last_block_counter_low.data<int>(),
              grad_output.data<scalar_t>() + init_element_offset,
              input.data<scalar_t>() + init_element_offset,
              input_low.data<scalar_t>() + scale_idx,
              input_range.data<scalar_t>() + scale_idx,
              levels,
              level_low,
              level_high,
              elements_per_scale);
        }));
        dev_tmp_range.fill_(0.0);
        dev_tmp_low.fill_(0.0);
        dev_last_block_counter_low.fill_(0);
        dev_last_block_counter_range.fill_(0);
    }
    return {grad_input, grad_input_low, grad_input_range};
}


std::vector<at::Tensor> q_single_scale_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {

    const auto size = input.numel();
    auto grad_input = at::empty_like(grad_output);

    auto grad_input_range = at::empty({1}, grad_output.options());
    auto grad_input_low = at::empty({1}, grad_output.options());

    auto grid_size = std::min(GET_BLOCKS(size), CUDA_GRID_SIZE);
    auto dev_tmp_range = at::empty({grid_size}, grad_output.options());
    auto dev_tmp_low = at::empty({grid_size}, grad_output.options());
    auto dev_last_block_counter_range = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "q_single_scale_cuda_backward", ([&] {
      q_single_scale_cuda_backward_kernel<scalar_t><<<grid_size, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
          grad_input.data<scalar_t>(),
          grad_input_low.data<scalar_t>(),
          grad_input_range.data<scalar_t>(),
          dev_tmp_range.data<scalar_t>(),
          dev_tmp_low.data<scalar_t>(),
          dev_last_block_counter_range.data<int>(),
          dev_last_block_counter_low.data<int>(),
          grad_output.data<scalar_t>(),
          input.data<scalar_t>(),
          input_low.data<scalar_t>(),
          input_range.data<scalar_t>(),
          levels,
          level_low,
          level_high,
          size);
    }));

    return {grad_input, grad_input_low, grad_input_range};
}


std::vector<at::Tensor> q_cuda_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {

    ScaleType scale_type = get_scale_type(input, input_low, input_range);

    switch (scale_type)
    {
        case ScaleType::PER_ACTIVATION_CHANNEL:
            return q_scale_per_activation_channel_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
        case ScaleType::PER_WEIGHT_CHANNEL:
            return q_scale_per_weight_channel_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
        case ScaleType::SINGLE_SCALE:
        default:
            return q_single_scale_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
    };
}
