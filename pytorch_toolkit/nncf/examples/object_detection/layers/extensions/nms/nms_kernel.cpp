// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"


template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& boxes,
                          const float threshold, const int64_t top_k) {
  AT_ASSERTM(!boxes.type().is_cuda(), "dets must be a CPU tensor");

  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = boxes.select(1, 0).contiguous();
  auto y1_t = boxes.select(1, 1).contiguous();
  auto x2_t = boxes.select(1, 2).contiguous();
  auto y2_t = boxes.select(1, 3).contiguous();
  auto scores = boxes.select(1, 4).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = std::min(boxes.size(0), top_k);

  at::Tensor suppressed_t = at::zeros({ndets}, boxes.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    if (suppressed[_i] == 1)
      continue;
    auto i = order[_i];
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      if (suppressed[_j] == 1)
        continue;
      auto j = order[_j];
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = abs(w * h);
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[_j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor nms_cpu(const at::Tensor& boxes, const float threshold, int64_t top_k) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(boxes.type(), "nms_cpu", [&] {
    result = nms_cpu_kernel<scalar_t>(boxes, threshold, top_k);
  });
  return result;
}
