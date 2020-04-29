"""
 Copyright (c) 2019-2020 Intel Corporation
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

from .version import __version__

# Required for correct COMPRESSION_ALGORITHMS registry functioning
from .binarization import algo as binarization_algo
from .quantization import algo as quantization_algo
from .sparsity.const import algo as const_sparsity_algo
from .sparsity.magnitude import algo as magnitude_sparsity_algo
from .sparsity.rb import algo as rb_sparsity_algo
from .pruning.filter_pruning import algo as filter_pruning_algo

# Functions most commonly used in integrating NNCF into training pipelines are
# listed below for importing convenience

from .model_creation import create_compressed_model
from .checkpoint_loading import load_state
from .config import Config
from .nncf_logger import disable_logging
from .nncf_logger import set_log_level

# NNCF relies on tracing PyTorch operations. Each code that uses NNCF
# should be executed with PyTorch operators wrapped via a call to "patch_torch_operators",
# so this call is moved to package __init__ to ensure this.
from .dynamic_graph.patch_pytorch import patch_torch_operators
patch_torch_operators()
