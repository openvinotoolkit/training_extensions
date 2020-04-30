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
import pytest
import torch

from nncf.pruning.filter_pruning.functions import FILTER_IMPORTANCE_FUNCTIONS, calculate_binary_mask


@pytest.mark.parametrize(("norm_name", "input_tensor", "reference"),
                         [('L1', torch.arange(120.0).view(2, 3, 4, 5), torch.tensor([1770.0, 5370.0])),
                          ('L2', torch.arange(120.0).view(2, 3, 4, 5), torch.tensor([264.9716966, 706.12321871])),
                          ('geometric_median', torch.arange(120.0).view(3, 2, 4, 5),
                           torch.tensor([758.94663844, 505.96442563, 758.94663844])),
                         ])
def test_norms(norm_name, input_tensor, reference):
    """
    Test correctness of all norms calculations.
    """
    norm_fn = FILTER_IMPORTANCE_FUNCTIONS.get(norm_name)
    result = norm_fn(input_tensor)
    assert torch.allclose(result, reference)


@pytest.mark.parametrize(("importance", "threshold", "reference"),
                         [(torch.arange(20.), 10.0, torch.tensor([0.0]*10 + [1.0]*10))]
                         )
def test_calculate_binary_mask(importance, threshold, reference):
    """
    Test correctness of binary mask calculation.
    """
    mask = calculate_binary_mask(importance, threshold)
    assert torch.allclose(mask, reference)
    assert isinstance(mask, torch.FloatTensor)
