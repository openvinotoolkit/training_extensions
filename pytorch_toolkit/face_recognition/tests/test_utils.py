"""
 Copyright (c) 2018 Intel Corporation
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

import unittest

import torch
from utils.utils import get_model_parameters_number


class UtilsTests(unittest.TestCase):
    """Tests for utils"""
    def test_parameters_counter(self):
        """Checks output of get_model_parameters_number"""
        class ParamsHolder(torch.nn.Module):
            """Dummy parameters holder"""
            def __init__(self, n_params):
                super(ParamsHolder, self).__init__()
                self.p1 = torch.nn.Parameter(torch.Tensor(n_params // 2))
                self.p2 = torch.nn.Parameter(torch.Tensor(n_params // 2))
                self.dummy = -1

        params_num = 1000
        module = ParamsHolder(params_num)
        estimated_params = get_model_parameters_number(module, as_string=False)
        self.assertEqual(estimated_params, params_num)

if __name__ == '__main__':
    unittest.main()
