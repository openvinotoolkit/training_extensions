// Copyright (c) 2018, Sergei Belousov
// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The original repo: https://github.com/bes-dev/mpl.pytorch

#include <limits>
#include <cmath>

#include <torch/extension.h>

void compute_weights(int size,
                     const torch::Tensor losses,
                     const torch::Tensor indices,
                     torch::Tensor weights,
                     float ratio,
                     float p) {
    const float* losses_data = losses.data_ptr<float>();
    const int64_t* indices_data = indices.data_ptr<int64_t>();
    float* weights_data = weights.data_ptr<float>();

    // find a first nonzero element
    int pos = 0;
    while(losses_data[pos] < std::numeric_limits<float>::epsilon()) {
        ++pos;
    }

    // Algorithm #1
    int n = size - pos;
    int m = int(ratio * n);
    if (n <= 0 || m <= 0) {
        return;
    }

    float q = p / (p - 1.0);
    int c = m - n;
    float a[2] = {0.0, 0.0};
    int i = pos;
    float eta = 0.0;
    for(; i < size && eta < std::numeric_limits<float>::epsilon(); ++i) {
        float loss_q = pow(losses_data[i] / losses_data[size - 1], q);

        a[0] = a[1];
        a[1] += loss_q;

        c += 1;
        eta = float(c) * loss_q - a[1];
    }

    // compute alpha
    float alpha;
    if (eta < std::numeric_limits<float>::epsilon()) {
        c += 1;
        a[0] = a[1];
    }
    alpha = pow(a[0] / c, 1.0 / q) * losses_data[size - 1];

    // compute weights
    float tau = 1.0 / (pow(n, 1.0 / q) * pow(m, 1.0 / p));
    for (int k = i; k < size; ++k) {
        weights_data[indices_data[k]] = tau;
    }
    if (alpha > -std::numeric_limits<float>::epsilon()) {
        for(int k = pos; k < i; ++k) {
            weights_data[indices_data[k]] = tau * pow(losses_data[k] / alpha, q - 1);
        }
    }
}
