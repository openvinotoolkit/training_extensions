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

import torch.nn as nn


class BaseOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    @property
    def operand(self):
        return self.op

    def forward(self, *inputs):
        return self.op(*inputs)


class UpdateInputs(BaseOp):
    def __call__(self, _, inputs):
        return super().__call__(*inputs)


class UpdateParameter(BaseOp):
    def __init__(self, param_name, op):
        super().__init__(op)
        self._param_name = param_name

    def __call__(self, module, _):
        if not hasattr(module, self._param_name):
            raise TypeError('{} should have {} attribute'.format(type(module), self._param_name))

        value = getattr(module, self._param_name)
        result = super().__call__(value)
        setattr(module, self._param_name, result)


class UpdateWeight(UpdateParameter):
    def __init__(self, op):
        super().__init__("weight", op)
