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

from .compression_method_api import CompressionAlgorithm
from .composite_compression import CompositeCompressionAlgorithm
from .registry import Registry

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


def create_compression_algorithm(model, config):
    compression_config = config.get('compression', {})
    input_size = config.get("input_sample_size", (1, 3, 224, 224))
    if isinstance(compression_config, dict):
        return get_compression_algorithm(compression_config)(model, compression_config, input_size)

    logger.info("Creating composite compression algorithm:")
    composite_compression_algorithm = CompositeCompressionAlgorithm(model, compression_config, input_size)

    for algo_config in compression_config:
        compression_algorithm = get_compression_algorithm(algo_config)(
            composite_compression_algorithm.model, algo_config, input_size
        )
        composite_compression_algorithm.add(compression_algorithm)

    from nncf.utils import check_for_quantization_before_sparsity
    check_for_quantization_before_sparsity(composite_compression_algorithm.child_algos)
    return composite_compression_algorithm
