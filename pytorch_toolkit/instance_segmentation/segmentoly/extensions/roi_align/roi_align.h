// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include <torch/torch.h>


at::Tensor roi_align_forward_cpu(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);

at::Tensor roi_align_forward_gpu(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);

at::Tensor roi_align_backward_gpu(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio);

at::Tensor roi_align_forward(const at::Tensor& input,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int sampling_ratio);

at::Tensor roi_align_backward(const at::Tensor& grad,
                              const at::Tensor& rois,
                              const float spatial_scale,
                              const int pooled_height,
                              const int pooled_width,
                              const int batch_size,
                              const int channels,
                              const int height,
                              const int width,
                              const int sampling_ratio);
