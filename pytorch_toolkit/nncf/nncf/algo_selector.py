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

import logging
from copy import copy
from typing import List

from .compression_method_api import CompressionAlgorithm
from .composite_compression import CompositeCompressionAlgorithm
from .registry import Registry
from nncf.dynamic_graph.graph_builder import ModelInputInfo, create_mock_tensor, create_input_infos

logger = logging.getLogger(__name__)

COMPRESSION_ALGORITHMS = Registry('compression algorithm')


@COMPRESSION_ALGORITHMS.register('NoCompressionAlgorithm')
class NoCompressionAlgorithm(CompressionAlgorithm):
    pass


def get_compression_algorithm(config):
    algorithm_key = config.get('algorithm', 'NoCompressionAlgorithm')
    logger.info("Creating compression algorithm: {}".format(algorithm_key))
    return COMPRESSION_ALGORITHMS.get(algorithm_key)


def remove_key(d, key):
    sd = copy(d)
    del sd[key]
    return sd


def create_dummy_forward_fn(input_infos: List[ModelInputInfo]):
    def default_dummy_forward_fn(model):
        device = next(model.parameters()).device
        tensor_list = [create_mock_tensor(info, device) for info in input_infos]
        return model(*tuple(tensor_list))

    return default_dummy_forward_fn


def create_compression_algorithm(model, config, dummy_forward_fn=None):
    compression_config = config.get('compression', {})

    input_info_list = create_input_infos(config)

    if isinstance(compression_config, dict):
        return get_compression_algorithm(compression_config)(model, compression_config,
                                                             input_infos=input_info_list,
                                                             dummy_forward_fn=dummy_forward_fn)
    if isinstance(compression_config, list) and len(compression_config) == 1:
        return get_compression_algorithm(compression_config[0])(model, compression_config[0],
                                                                input_infos=input_info_list,
                                                                dummy_forward_fn=dummy_forward_fn)

    logger.info("Creating composite compression algorithm:")
    composite_compression_algorithm = CompositeCompressionAlgorithm(model, compression_config,
                                                                    input_infos=input_info_list,
                                                                    dummy_forward_fn=dummy_forward_fn)

    for algo_config in compression_config:
        compression_algorithm = get_compression_algorithm(algo_config)(
            composite_compression_algorithm.model, algo_config,
            input_infos=input_info_list,
            dummy_forward_fn=dummy_forward_fn)
        composite_compression_algorithm.add(compression_algorithm)

    from nncf.utils import check_for_quantization_before_sparsity
    check_for_quantization_before_sparsity(composite_compression_algorithm.child_algos)
    return composite_compression_algorithm
