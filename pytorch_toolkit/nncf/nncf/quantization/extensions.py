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

import os
import pathlib
import os.path
from nncf.definitions import get_install_type
from torch.utils.cpp_extension import load


if "VIRTUAL_ENV" in os.environ:
    build_dir = os.path.join(os.environ["VIRTUAL_ENV"], "torch_extensions")
    pathlib.Path(build_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = build_dir

ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu")
QuantizedFunctionsCPU = load(
    'quantized_functions_cpu', [
        os.path.join(ext_dir, 'functions_cpu.cpp')
    ],
    verbose=False
)
if get_install_type() == 'GPU':
    ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
    QuantizedFunctionsCUDA = load(
        'quantized_functions_cuda', [
            os.path.join(ext_dir, 'functions_cuda.cpp'),
            os.path.join(ext_dir, 'functions_cuda_kernel.cu')
        ],
        verbose=False
    )
else:
    QuantizedFunctionsCUDA = None
