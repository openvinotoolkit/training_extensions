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

from nncf.algo_selector import create_compression_algorithm
from nncf.dynamic_graph.utils import dump_graph, build_graph
from nncf.utils import get_all_modules
from .utils import is_main_process


def create_compressed_model(model, config):
    input_args = (next(model.parameters()).new_empty(config['input_sample_size']),)
    if is_main_process():
        print(*get_all_modules(model).keys(), sep="\n")
        ctx = build_graph(model, 'create_model', input_args=input_args, reset_context=True)
        dump_graph(ctx, osp.join(config.log_dir, "original_graph.dot"))

    compression_algo = create_compression_algorithm(model, config)

    if is_main_process():
        if hasattr(compression_algo.model, "build_graph"):
            ctx = compression_algo.model.build_graph()
        else:
            ctx = build_graph(compression_algo.model, "create_model_compressed",
                              input_args=input_args, reset_context=True)
        dump_graph(ctx, osp.join(config.log_dir, "compressed_graph.dot"))

    model = compression_algo.model
    return compression_algo, model
