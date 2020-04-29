"""
 Copyright (c) 2020 Intel Corporation
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
import torch


def l1_filter_norm(weight_tensor):
    """
    Calculates L1 for weight_tensor for the first dimension.
    """
    return torch.norm(weight_tensor.view(weight_tensor.shape[0], -1), p=1, dim=1)


def l2_filter_norm(weight_tensor):
    """
    Calculates L2 for weight_tensor for the first dimension.
    """
    return torch.norm(weight_tensor.view(weight_tensor.shape[0], -1), p=2, dim=1)


def tensor_l2_normalizer(weight_tensor):
    norm = torch.sqrt(torch.sum(torch.abs(weight_tensor) ** 2))
    return weight_tensor / norm


def geometric_median_filter_norm(weight_tensor):
    """
    Compute geometric median norm for filters.
    :param weight_tensor: tensor with weights
    :return: metric value for every weight from weights_tensor
    """
    filters_count = weight_tensor.size(0)
    weight_vec = weight_tensor.view(filters_count, -1)
    similar_matrix = torch.zeros((filters_count, filters_count))
    pdist_fn = torch.nn.PairwiseDistance(p=2)
    for i in range(filters_count):
        for j in range(i, filters_count):
            similar_matrix[i, j] = pdist_fn(weight_vec[None, i], weight_vec[None, j])[0].item()
            similar_matrix[j, i] = similar_matrix[i, j]
    similar_sum = similar_matrix.sum(axis=0)
    return similar_sum


FILTER_IMPORTANCE_FUNCTIONS = {
    'L2': l2_filter_norm,
    'L1': l1_filter_norm,
    'geometric_median': geometric_median_filter_norm
}


def calculate_binary_mask(importance, threshold):
    return (importance >= threshold).float()
