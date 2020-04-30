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

from copy import copy
from typing import List

from nncf.hw_config import HWConfigType
from .compression_method_api import CompressionAlgorithmBuilder, CompressionAlgorithmController
from .registry import Registry

from nncf.nncf_logger import logger as nncf_logger

COMPRESSION_ALGORITHMS = Registry('compression algorithm')


@COMPRESSION_ALGORITHMS.register('NoCompressionAlgorithmBuilder')
class NoCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    pass


class NoCompressionAlgorithmController(CompressionAlgorithmController):
    pass


def get_compression_algorithm(config):
    algorithm_key = config.get('algorithm', 'NoCompressionAlgorithmBuilder')
    nncf_logger.info("Creating compression algorithm: {}".format(algorithm_key))
    return COMPRESSION_ALGORITHMS.get(algorithm_key)


def remove_key(d, key):
    sd = copy(d)
    del sd[key]
    return sd


def create_compression_algorithm_builders(config) -> List[CompressionAlgorithmBuilder]:
    compression_config = config.get('compression', {})

    hw_config_type = None
    hw_config_type_str = config.get("hw_config_type")
    if hw_config_type_str is not None:
        hw_config_type = HWConfigType.from_str(config.get("hw_config_type"))
    if isinstance(compression_config, dict):
        compression_config["hw_config_type"] = hw_config_type
        return [get_compression_algorithm(compression_config)(compression_config), ]
    retval = []
    for algo_config in compression_config:
        algo_config["hw_config_type"] = hw_config_type
        retval.append(get_compression_algorithm(algo_config)(algo_config))
    return retval
