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

#pragma once

#include <torch/torch.h>

int deform_conv_forward_cuda(
    at::Tensor &input, at::Tensor &weight,
    at::Tensor &offset, at::Tensor &output,
    at::Tensor &columns, at::Tensor &ones,
    const int kW, const int kH, const int dW, const int dH,
    const int padW, const int padH,
    const int dilationH, const int dilationW,
    const int deformable_group);

int deform_conv_backward_input_cuda(
    at::Tensor &input, at::Tensor &offset,
    at::Tensor &gradOutput, at::Tensor &gradInput,
    at::Tensor &gradOffset, at::Tensor &weight,
    at::Tensor &columns,
    const int kW, const int kH,
    const int dW, const int dH,
    const int padW, const int padH,
    const int dilationH, const int dilationW,
    const int deformable_group);

int deform_conv_backward_parameters_cuda(
    at::Tensor &input, at::Tensor &offset,
    at::Tensor &gradOutput, at::Tensor &gradWeight,
    at::Tensor &columns, at::Tensor &ones,
    const int kW, const int kH,
    const int dW, const int dH,
    const int padW, const int padH,
    const int dilationH, const int dilationW,
    const int deformable_group, const float scale);
