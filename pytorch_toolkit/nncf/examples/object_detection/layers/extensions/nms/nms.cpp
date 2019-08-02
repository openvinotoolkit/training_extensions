// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/torch.h>

#include "nms.h"

at::Tensor nms(const at::Tensor& boxes, const float nms_overlap_thresh, const int64_t top_k) {
  if (boxes.type().is_cuda()) {
    return nms_gpu(boxes, nms_overlap_thresh, top_k);
  }
  return nms_cpu(boxes, nms_overlap_thresh, top_k);
}
