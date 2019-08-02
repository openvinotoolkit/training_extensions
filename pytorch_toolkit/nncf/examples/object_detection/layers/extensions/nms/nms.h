// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include <torch/torch.h>

at::Tensor nms_gpu(const at::Tensor& boxes, float nms_overlap_thresh, int64_t top_k);
at::Tensor nms_cpu(const at::Tensor& boxes, float nms_overlap_thresh, int64_t top_k);
at::Tensor nms(const at::Tensor& boxes, const float nms_overlap_thresh, int64_t top_k);
