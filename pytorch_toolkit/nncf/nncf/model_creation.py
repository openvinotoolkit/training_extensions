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

from os import path as osp
from typing import Callable, Any, Tuple

from torch.nn import Module

from nncf.algo_selector import create_compression_algorithm_builders
from nncf.compression_method_api import CompressionAlgorithmController
from nncf.config import Config
from nncf.debug import is_debug, set_debug_log_dir
from nncf.dynamic_graph.graph_builder import create_input_infos, GraphBuilder, create_dummy_forward_fn
from nncf.nncf_network import NNCFNetwork
from nncf.utils import is_main_process


def create_compressed_model(model: Module, config: Config, dummy_forward_fn: Callable[[Module], Any] = None,
                            dump_graphs=True) -> Tuple[CompressionAlgorithmController, NNCFNetwork]:
    """
    The main function used to produce a model ready for compression fine-tuning from an original PyTorch
    model and a configuration object.
    dummy_forward_fn
    :param model: The original model. Should have its parameters already loaded from a checkpoint or another
    source.
    :param config: A configuration object used to determine the exact compression modifications to be applied
    to the model
    :param dummy_forward_fn: will be used instead of a *forward* function call to build
    the internal graph representation via tracing. Specifying this is useful when the original training pipeline
    has special formats of data loader output or has additional *forward* arguments other than input tensors.
    Otherwise, the *forward* call of the model during graph tracing will be made with mock tensors according
    to the shape specified in the config object.
    :param dump_graphs: Whether or not should also dump the internal graph representation of the
    original and compressed models in the .dot format into the log directory.
    :return: A controller for the compression algorithm (or algorithms, in which case the controller
    is an instance of CompositeCompressionController) and the model ready for compression wrapped
    as an object of NNCFNetwork."""

    if dump_graphs:
        if dummy_forward_fn is None:
            input_info_list = create_input_infos(config)
            graph_builder = GraphBuilder(custom_forward_fn=
                                         create_dummy_forward_fn(input_info_list,
                                                                 with_input_tracing=True))
        else:
            graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)

        if is_main_process():
            graph = graph_builder.build_graph(model)
            graph.dump_graph(osp.join(config.log_dir, "original_graph.dot"), extended=True)

    if is_debug():
        set_debug_log_dir(config.log_dir)

    input_info_list = create_input_infos(config)
    scopes_without_shape_matching = config.get('scopes_without_shape_matching', [])
    ignored_scopes = config.get('ignored_scopes')
    target_scopes = config.get('target_scopes')

    compressed_model = NNCFNetwork(model, input_infos=input_info_list,
                                   dummy_forward_fn=dummy_forward_fn,
                                   ignored_scopes=ignored_scopes,
                                   target_scopes=target_scopes,
                                   scopes_without_shape_matching=scopes_without_shape_matching)

    compression_algo_builder_list = create_compression_algorithm_builders(config)

    for builder in compression_algo_builder_list:
        compressed_model = builder.apply_to(compressed_model)
    compression_ctrl = compressed_model.commit_compression_changes()

    if dump_graphs and is_main_process() and compression_algo_builder_list:
        if dummy_forward_fn is None:
            compressed_graph_builder = GraphBuilder(custom_forward_fn=
                                                    create_dummy_forward_fn(input_info_list,
                                                                            with_input_tracing=False))
        else:
            compressed_graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)

        graph = compressed_graph_builder.build_graph(compressed_model, compressed_model.get_tracing_context())
        graph.dump_graph(osp.join(config.log_dir, "compressed_graph.dot"), extended=True)

    return compression_ctrl, compressed_model
