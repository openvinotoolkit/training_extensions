"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

#include <ATen/ATen.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <assert.h>
#include <cstdio>

extern THCState *state;

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ float deformable_im2col_bilinear(const float *bottom_data,
                                            const int data_width,
                                            const int height, const int width,
                                            float h, float w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high;
  int w_high;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (float)h_low;
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (float)w_low;
  } else {
    w_high = w_low + 1;
  }

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = bottom_data[h_low * data_width + w_low];
  float v2 = bottom_data[h_low * data_width + w_high];
  float v3 = bottom_data[h_high * data_width + w_low];
  float v4 = bottom_data[h_high * data_width + w_high];
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ float get_gradient_weight(float argmax_h, float argmax_w,
                                     const int h, const int w, const int height,
                                     const int width) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  argmax_h = max(argmax_h, (float)0.0f);
  argmax_w = max(argmax_w, (float)0.0f);

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (float)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (float)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  float weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  } else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}


__device__ float get_coordinate_weight(float argmax_h, float argmax_w,
                                       const int height, const int width,
                                       const float *im_data,
                                       const int data_width, const int bp_dir) {

  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  if (argmax_h < 0) {
    argmax_h = 0;
  }
  if (argmax_w < 0) {
    argmax_w = 0;
  }

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (float)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (float)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  float weight = 0;

  if (bp_dir == 0) {
    weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}


__global__ void deformable_im2col_gpu_kernel(
    const int n, const float *data_im, const float *data_offset,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, float *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int c_im = (index / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    float *data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
    const float *data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
    const float *data_offset_ptr = data_offset + deformable_group_index * 2 *
                                   kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          const float map_h = i * dilation_h + offset_h;
          const float map_w = j * dilation_w + offset_w;
          const int cur_height = height - h_in;
          const int cur_width = width - w_in;
          val = deformable_im2col_bilinear(data_im_ptr, width, cur_height,
                                           cur_width, map_h, map_w);
        }
        *data_col_ptr = val;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}


void deformable_im2col(const float *data_im,
                       const float *data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, float *data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.

  auto stream = THCState_getCurrentStream(state);
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  int channel_per_deformable_group = channels / deformable_group;
  // Launch
  deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0,  stream>>>(
      num_kernels, data_im, data_offset, height, width, ksize_h, ksize_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, height_col, width_col, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}


__global__ void deformable_col2im_gpu_kernel(
    const int n, const float *data_col, const float *data_offset,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, float *grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col) % kernel_w;
    const int i = (index / width_col / height_col / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const float *data_offset_ptr = data_offset + deformable_group_index * 2 *
                                   kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const float cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos = (c * height + cur_h + dy) * width + cur_w + dx;
          float weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy,
                                             cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}


void deformable_col2im(const float *data_col,
                       const float *data_offset, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int deformable_group, float *grad_im) {

  auto stream = THCState_getCurrentStream(state);
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * ksize_h * ksize_w * height_col * width_col;
  int channel_per_deformable_group = channels / deformable_group;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, data_col, data_offset, channels, height, width, ksize_h,
      ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, height_col, width_col, grad_im);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}


__global__ void deformable_col2im_coord_gpu_kernel(
    const int n, const float *data_col, const float *data_im,
    const float *data_offset, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col,
    const int width_col, float *grad_offset) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = index / width_col / height_col;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const float *data_col_ptr = data_col + deformable_group_index *
        channel_per_deformable_group * width_col * height_col;
    const float *data_im_ptr =
        data_im + deformable_group_index * channel_per_deformable_group /
                      kernel_h / kernel_w * height * width;
    const float *data_offset_ptr = data_offset + deformable_group_index * 2 *
        kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
      const int col_pos = ((col_c * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col) % kernel_w;
      int i = (col_pos / width_col / height_col / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      float inv_h = h_in + i * dilation_h + offset_h;
      float inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h < 0 || inv_w < 0 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -1;
      }
      const float weight = get_coordinate_weight(
          inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
          width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}


void deformable_col2im_coord(const float *data_col,
                             const float *data_im, const float *data_offset,
                             const int channels, const int height, const int width,
                             const int ksize_h,  const int ksize_w, const int pad_h,
                             const int pad_w, const int stride_h, const int stride_w,
                             const int dilation_h, const int dilation_w,
                             const int deformable_group, float *grad_offset) {

  auto stream = THCState_getCurrentStream(state);
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group;
  int channel_per_deformable_group = channels * ksize_h * ksize_w / deformable_group;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, data_col, data_im, data_offset, channels, height, width,
      ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
      dilation_w, channel_per_deformable_group, height_col, width_col,
      grad_offset);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
    // TODO(BZ) panic
  }
}


/*=========== General functions ===========*/


void shape_check(const at::Tensor &input, const at::Tensor &offset,
                 const at::Tensor* gradOutput, const at::Tensor &weight,
                 const int kH, const int kW, const int dH, const int dW,
                 const int padH, const int padW,
                 const int dilationH, const int dilationW, const int deformable_group) {

  THArgCheck(weight.ndimension() == 4, 5,
             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
             "but got: %s",
             weight.ndimension());

  THArgCheck(weight.is_contiguous(), 5,
             "weight tensor has to be contiguous");

  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d",
             kH, kW);

  THArgCheck(weight.size(2) == kH && weight.size(3) == kW, 9,
             "kernel size should be consistent with weight, ",
             "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
             kW, weight.size(2), weight.size(3));

  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH,
             dW);

  THArgCheck(
      dilationW > 0 && dilationH > 0, 14,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THArgCheck(ndim == 3 || ndim == 4, 2,
             "3D or 4D input tensor expected but got: %s", ndim);

  long nInputPlane  = weight.size(1);
  long inputHeight  = input.size(dimh);
  long inputWidth   = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth  + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  THArgCheck(nInputPlane % deformable_group == 0, 2,
             "input channels must divide deformable group size");

  if (outputWidth < 1 || outputHeight < 1)
    THError(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  THArgCheck(input.size(1) == nInputPlane, 2,
             "invalid number of input planes, expected: %d, but got: %d",
             nInputPlane, input.size(1));

  THArgCheck((inputHeight >= kH && inputWidth >= kW), 2,
             "input image is smaller than kernel");

  THArgCheck(
      (offset.size(2) == outputHeight && offset.size(3) == outputWidth), 3,
      "invalid spatial size of offset, expected height: %d width: %d, but got height: %d width: %d", outputHeight, outputWidth,
      offset.size(2), offset.size(3));

  THArgCheck((offset.size(1) == deformable_group * 2 * kH * kW), 3,
             "invalid number of channels of offset");

  if (gradOutput != nullptr) {
    THArgCheck(gradOutput->size(dimf) == nOutputPlane, 4,
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, gradOutput->size(dimf));

    THArgCheck((gradOutput->size(dimh) == outputHeight && gradOutput->size(dimw) == outputWidth),
               4, "invalid size of gradOutput, expected height: %d width: %d, but got height: %d width: %d", outputHeight, outputWidth,
               gradOutput->size(dimh), gradOutput->size(dimw));
  }
}

int deform_conv_forward_cuda(at::Tensor &input, at::Tensor &weight,
                             at::Tensor &offset, at::Tensor &output,
                             at::Tensor &columns, at::Tensor &ones,
                             const int kW, const int kH, const int dW, const int dH,
                             const int padW, const int padH,
                             const int dilationH, const int dilationW,
                             const int deformable_group) {

  assert(input.get_device() == weight.get_device() &&
         input.get_device() == offset.get_device() &&
         input.get_device() == output.get_device() &&
         input.get_device() == columns.get_device() &&
         input.get_device() == ones.get_device());

  shape_check(input, offset, nullptr, weight, kH, kW, dH, dW, padH, padW, dilationH, dilationW, deformable_group);

  input.contiguous();
  offset.contiguous();
  weight.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;

    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    offset.resize_({1, offset.size(0), offset.size(1), offset.size(2)});
  }

  long batchSize   = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth  = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset.size(0) == batchSize), 3, "invalid batch size of offset");

  output.resize_({batchSize, nOutputPlane, outputHeight, outputWidth});
  columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  if (ones.ndimension() != 2 || ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones.resize_({outputHeight, outputWidth});
    ones.fill_(1);
  }

  at::Tensor input_n = at::empty(0, input.options());
  at::Tensor offset_n = at::empty(0, offset.options());
  at::Tensor output_n = at::empty(0, output.options());

  for (int elt = 0; elt < batchSize; elt++) {
    input_n = input.select(0, elt);
    offset_n = offset.select(0, elt);
    output_n = output.select(0, elt);

    output_n.fill_(0);

    deformable_im2col(
        input_n.data<float>(), offset_n.data<float>(),
        nInputPlane, inputHeight, inputWidth,
        kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group, columns.data<float>());

    long m = nOutputPlane;
    long n = columns.size(1);
    long k = nInputPlane * kH * kW;

    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                     columns.data<float>(), n,
                     weight.data<float>(), k, 1.0f,
                     output_n.data<float>(), n);
  }

  //THCudaTensor_free(state, input_n);
  //THCudaTensor_free(state, offset_n);
  //THCudaTensor_free(state, output_n);

  if (batch == 0) {
    input.resize_({nInputPlane, inputHeight, inputWidth});
    offset.resize_({offset.size(1), offset.size(2), offset.size(3)});
    output.resize_({nOutputPlane, outputHeight, outputWidth});
  }

  //THCudaTensor_free(state, input);
  //THCudaTensor_free(state, offset);
  //THCudaTensor_free(state, weight);

  return 1;
}

int deform_conv_backward_input_cuda(at::Tensor &input, at::Tensor &offset,
                                    at::Tensor &gradOutput, at::Tensor &gradInput,
                                    at::Tensor &gradOffset, at::Tensor &weight,
                                    at::Tensor &columns,
                                    const int kW, const int kH,
                                    const int dW, const int dH,
                                    const int padW, const int padH,
                                    const int dilationH, const int dilationW,
                                    const int deformable_group) {

  assert(input.get_device() == weight.get_device() &&
         input.get_device() == offset.get_device() &&
         input.get_device() == gradOutput.get_device() &&
         input.get_device() == gradInput.get_device() &&
         input.get_device() == gradOffset.get_device() &&
         input.get_device() == columns.get_device());

  shape_check(input, offset, &gradOutput, weight, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW, deformable_group);

  input.contiguous();
  offset.contiguous();
  weight.contiguous();
  gradOutput.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    offset.resize_({1, offset.size(0), offset.size(1), offset.size(2)});
    gradOutput.resize_({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize   = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth  = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset.size(0) == batchSize), 3, "invalid batch size of offset");

  gradInput.resize_({batchSize, nInputPlane, inputHeight, inputWidth});
  columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  at::Tensor input_n = at::empty(0, input.options());
  at::Tensor offset_n = at::empty(0, offset.options());
  at::Tensor gradInput_n = at::empty(0, gradInput.options());
  at::Tensor gradOffset_n = at::empty(0, gradOffset.options());
  at::Tensor gradOutput_n = at::empty(0, gradOutput.options());

  for (int elt = 0; elt < batchSize; elt++) {
    input_n = input.select(0, elt);
    offset_n = offset.select(0, elt);
    gradInput_n = gradInput.select(0, elt);
    gradOffset_n = gradOffset.select(0, elt);
    gradOutput_n = gradOutput.select(0, elt);

    long m = nInputPlane * kW * kH;
    long n = columns.size(1);
    long k = nOutputPlane;

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                     gradOutput_n.data<float>(), n,
                     weight.data<float>(), m, 0.0f,
                     columns.data<float>(), n);

    deformable_col2im_coord(
        columns.data<float>(), input_n.data<float>(), offset_n.data<float>(),
        nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
        dilationH, dilationW, deformable_group,
        gradOffset_n.data<float>());

    deformable_col2im(
        columns.data<float>(), offset_n.data<float>(),
        nInputPlane, inputHeight, inputWidth,
        kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        deformable_group, gradInput_n.data<float>());
  }

  //THCudaTensor_free(state, gradInput_n);
  //THCudaTensor_free(state, gradOffset_n);
  //THCudaTensor_free(state, input_n);
  //THCudaTensor_free(state, offset_n);
  //THCudaTensor_free(state, gradOutput_n);

  if (batch == 0) {
    gradOutput.resize_({nOutputPlane, outputHeight, outputWidth});
    input.resize_({nInputPlane, inputHeight, inputWidth});
    gradInput.resize_({nInputPlane, inputHeight, inputWidth});
    offset.resize_({offset.size(1), offset.size(2), offset.size(3)});
    gradOffset.resize_({offset.size(1), offset.size(2), offset.size(3)});
  }

  //THCudaTensor_free(state, input);
  //THCudaTensor_free(state, offset);
  //THCudaTensor_free(state, gradOutput);
  //THCudaTensor_free(state, weight);

  return 1;
}

int deform_conv_backward_parameters_cuda(at::Tensor &input, at::Tensor &offset,
                                         at::Tensor &gradOutput, at::Tensor &gradWeight,
                                         at::Tensor &columns, at::Tensor &ones,
                                         const int kW, const int kH,
                                         const int dW, const int dH,
                                         const int padW, const int padH,
                                         const int dilationH, const int dilationW,
                                         const int deformable_group, const float scale) {

  assert(input.get_device() == offset.get_device() &&
         input.get_device() == gradOutput.get_device() &&
         input.get_device() == gradWeight.get_device() &&
         input.get_device() == columns.get_device() &&
         input.get_device() == ones.get_device());

  shape_check(input, offset, &gradOutput, gradWeight, kH, kW, dH, dW,
              padH, padW, dilationH, dilationW, deformable_group);

  input.contiguous();
  offset.contiguous();
  gradOutput.contiguous();

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
    gradOutput.resize_({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize   = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth  = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((offset.size(0) == batchSize), 3, "invalid batch size of offset");

  columns.resize_({nInputPlane * kW * kH, outputHeight * outputWidth});

  at::Tensor input_n = at::empty(0, input.options());
  at::Tensor offset_n = at::empty(0, offset.options());
  at::Tensor gradOutput_n = at::empty(0, gradOutput.options());

  for (int elt = 0; elt < batchSize; elt++) {
    input_n = input.select(0, elt);
    offset_n = offset.select(0, elt);
    gradOutput_n = gradOutput.select(0, elt);

    deformable_im2col(input_n.data<float>(), offset_n.data<float>(),
                      nInputPlane, inputHeight, inputWidth,
                      kH, kW, padH, padW, dH, dW, dilationH, dilationW,
                      deformable_group, columns.data<float>());

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = columns.size(1);

    THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
                     columns.data<float>(), k,
                     gradOutput_n.data<float>(), k, 1.0f,
                     gradWeight.data<float>(), n);
  }

  //THCudaTensor_free(state, input_n);
  //THCudaTensor_free(state, offset_n);
  //THCudaTensor_free(state, gradOutput_n);

  if (batch == 0) {
    gradOutput.resize_({nOutputPlane, outputHeight, outputWidth});
    input.resize_({nInputPlane, inputHeight, inputWidth});
  }

  //THCudaTensor_free(state, input);
  //THCudaTensor_free(state, offset);
  //THCudaTensor_free(state, gradOutput);

  return 1;
}
