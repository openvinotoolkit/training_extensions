// Copyright (c) 2018, Sergei Belousov
// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
void compute_weights(int size,
                     const torch::Tensor losses,
                     const torch::Tensor indices,
                     torch::Tensor weights,
                     float ratio,
                     float p);
