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

import os.path as osp
from typing import Callable, Any

from torch.nn import Module

from nncf import QuantizedNetwork
from nncf.algo_selector import create_compression_algorithm, NoCompressionAlgorithm
from nncf.config import Config
from nncf.dynamic_graph.context import reset_context
from nncf.dynamic_graph.graph_builder import GraphBuilder, create_input_infos
from nncf.utils import create_dummy_forward_fn, get_all_modules

from .utils import is_main_process


def create_compressed_model(model: Module, config: Config, dummy_forward_fn: Callable[[Module], Any] = None):
    """dummy_forward_fn will be used instead of a *forward* function call to build
    the graph - useful when the original training pipeline has special formats of
    data loader output or has additional *forward* arguments other than input tensors.
    Otherwise, the *forward* call of the model will be made with a single Tensor with
    a shape and type specified in config."""

    if dummy_forward_fn is None:
        input_info_list = create_input_infos(config)
        graph_builder = GraphBuilder(custom_forward_fn=
                                     create_dummy_forward_fn(input_info_list))
    else:
        graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)

    if is_main_process():
        print(*get_all_modules(model).keys(), sep="\n")
        reset_context('create_model')
        graph = graph_builder.build_graph(model, 'create_model')
        graph.dump_graph(osp.join(config.log_dir, "original_graph.dot"))

    compression_algo = create_compression_algorithm(model, config, dummy_forward_fn)

    compressed_model = compression_algo.model
    if is_main_process() and not isinstance(compression_algo, NoCompressionAlgorithm):
        context_name = 'create_compressed_graph'
        if isinstance(compressed_model, QuantizedNetwork):
            context_name = compressed_model.get_context_name()
        graph = graph_builder.build_graph(compression_algo.model, context_name)
        graph.dump_graph(osp.join(config.log_dir, "compressed_graph.dot"))

    return compression_algo, compressed_model
