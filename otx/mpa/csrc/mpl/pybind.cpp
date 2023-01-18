// Copyright (c) 2018, Sergei Belousov
// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <torch/extension.h>

void compute_weights(int size,
                     const torch::Tensor losses,
                     const torch::Tensor indices,
                     torch::Tensor weights,
                     float ratio,
                     float p);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_weights", &compute_weights, "compute_weights",
        py::arg("size"), py::arg("losses"), py::arg("indices"),
        py::arg("weights"), py::arg("ratio"), py::arg("p"));
}
