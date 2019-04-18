// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/torch.h>

#include "roi_align.h"

at::Tensor roi_align_forward(const at::Tensor& input,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int sampling_ratio) {
  if (input.type().is_cuda()) {
    return roi_align_forward_gpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
  }
  return roi_align_forward_cpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
};

at::Tensor roi_align_backward(const at::Tensor& grad,
                              const at::Tensor& rois,
                              const float spatial_scale,
                              const int pooled_height,
                              const int pooled_width,
                              const int batch_size,
                              const int channels,
                              const int height,
                              const int width,
                              const int sampling_ratio) {
  if (grad.type().is_cuda()) {
    return roi_align_backward_gpu(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size,
                                  channels, height, width, sampling_ratio);
  }
  AT_ERROR("Not implemented on the CPU");
};
