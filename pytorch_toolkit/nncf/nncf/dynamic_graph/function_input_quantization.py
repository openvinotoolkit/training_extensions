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

class FunctionQuantizationInfo:
    """A FunctionQuantizationInfo('foo', [0, 2, 3]) will specify that 0-th, 2-nd and 3-rd arguments
    of the function torch.nn.functional.foo will be considered for quantization."""
    def __init__(self, name: str, positions_of_args_to_quantize: list):
        self.name = name
        self.positions_of_args_to_quantize = positions_of_args_to_quantize


# Specification for function inputs to be quantized
# E.g.: x = torch.nn.functional.linear(input, weight, bias)

FUNCTIONS_TO_QUANTIZE = [
    FunctionQuantizationInfo('linear', [0, 1])
]
