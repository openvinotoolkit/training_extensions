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
from typing import Callable, Any, List
import logging
from copy import deepcopy

import torch

from nncf.dynamic_graph import context, get_context
from nncf.dynamic_graph.graph import NNCFGraph

logger = logging.getLogger(__name__)


class ModelInputInfo:
    MOCK_TYPE_ONES = "ones"
    MOCK_TYPE_ZEROS = "zeros"
    MOCK_TYPE_RANDOM = "random"
    MOCK_TYPES = [MOCK_TYPE_ONES, MOCK_TYPE_ZEROS, MOCK_TYPE_RANDOM]

    def __init__(self, shape: tuple, type_str: str = "float", mock=None):
        self.shape = shape
        self.type = self._string_to_torch_type(type_str)
        if mock is None:
            self.mock = self.MOCK_TYPE_ONES
        else:
            self.mock = mock
            if self.mock not in self.MOCK_TYPES:
                raise RuntimeError("Unknown input mocking type: {}".format(mock))

    def _string_to_torch_type(self, string):
        if string == "long":
            return torch.long
        return torch.float32


def create_input_infos(config) -> List[ModelInputInfo]:
    input_infos = config.get("input_info", [])
    if isinstance(input_infos, dict):
        return [ModelInputInfo(input_infos.get("sample_size"),
                               input_infos.get("type"),
                               input_infos.get("mock")), ]
    if isinstance(input_infos, list):
        if not input_infos:
            return [ModelInputInfo((1, 3, 224, 224))]
        return [ModelInputInfo(info_dict.get("sample_size"),
                               info_dict.get("type"),
                               info_dict.get("mock")) for info_dict in input_infos]
    raise RuntimeError("Invalid input_infos specified in config - should be either dict or list of dicts")


def create_mock_tensor(input_info: ModelInputInfo, device: str):
    args = {"size": input_info.shape, "dtype": input_info.type, "device": device}
    if input_info.mock == ModelInputInfo.MOCK_TYPE_ZEROS:
        return torch.zeros(**args)
    if input_info.mock == ModelInputInfo.MOCK_TYPE_ONES:
        return torch.ones(**args)
    if input_info.mock == ModelInputInfo.MOCK_TYPE_RANDOM:
        return torch.rand(**args)
    raise RuntimeError


class GraphBuilder:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def build_graph(self, model: torch.nn.Module, context_name: str) -> NNCFGraph:
        logger.info("Building graph with context: {}".format(context_name))
        sd = deepcopy(model.state_dict())

        ctx = get_context(context_name)
        with context(context_name):
            self.custom_forward_fn(model)
        model.load_state_dict(sd)

        if isinstance(model, PostGraphBuildActing):
            model.post_build_graph_actions()
        return ctx.graph


class PostGraphBuildActing:
    def post_build_graph_actions(self):
        pass
