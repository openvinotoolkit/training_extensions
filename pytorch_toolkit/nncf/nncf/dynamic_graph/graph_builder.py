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
from typing import Callable, Any, List, Optional
from copy import deepcopy

import torch


class ModelInputInfo:
    FILLER_TYPE_ONES = "ones"
    FILLER_TYPE_ZEROS = "zeros"
    FILLER_TYPE_RANDOM = "random"
    FILLER_TYPES = [FILLER_TYPE_ONES, FILLER_TYPE_ZEROS, FILLER_TYPE_RANDOM]

    def __init__(self, shape: tuple, type_str: str = "float", keyword=None, filler=None):
        self.shape = shape
        self.type = self._string_to_torch_type(type_str)
        self.keyword = keyword
        if filler is None:
            self.filler = self.FILLER_TYPE_ONES
        else:
            self.filler = filler
            if self.filler not in self.FILLER_TYPES:
                raise RuntimeError("Unknown input filler type: {}".format(filler))

    def _string_to_torch_type(self, string):
        if string == "long":
            return torch.long
        return torch.float32


def create_input_infos(config) -> List[ModelInputInfo]:
    input_infos = config.get("input_info", [])
    if isinstance(input_infos, dict):
        return [ModelInputInfo(input_infos.get("sample_size"),
                               input_infos.get("type"),
                               input_infos.get("keyword"),
                               input_infos.get("filler")), ]
    if isinstance(input_infos, list):
        if not input_infos:
            return [ModelInputInfo((1, 3, 224, 224))]
        return [ModelInputInfo(info_dict.get("sample_size"),
                               info_dict.get("type"),
                               info_dict.get("keyword"),
                               info_dict.get("filler")) for info_dict in input_infos]
    raise RuntimeError("Invalid input_infos specified in config - should be either dict or list of dicts")


def create_mock_tensor(input_info: ModelInputInfo, device: str):
    args = {"size": input_info.shape, "dtype": input_info.type, "device": device}
    if input_info.filler == ModelInputInfo.FILLER_TYPE_ZEROS:
        return torch.zeros(**args)
    if input_info.filler == ModelInputInfo.FILLER_TYPE_ONES:
        return torch.ones(**args)
    if input_info.filler == ModelInputInfo.FILLER_TYPE_RANDOM:
        return torch.rand(**args)
    raise RuntimeError


class GraphBuilder:
    def __init__(self, custom_forward_fn: Callable[[torch.nn.Module], Any]):
        self.custom_forward_fn = custom_forward_fn

    def build_graph(self, model: torch.nn.Module, context_to_use: Optional['TracingContext'] = None) -> 'NNCFGraph':
        sd = deepcopy(model.state_dict())

        from nncf.dynamic_graph.context import TracingContext
        if context_to_use is None:
            context_to_use = TracingContext()

        with context_to_use as _ctx:
            self.custom_forward_fn(model)
        model.load_state_dict(sd)

        if isinstance(model, PostGraphBuildActing):
            model.post_build_graph_actions()
        return context_to_use.graph


class PostGraphBuildActing:
    def post_build_graph_actions(self):
        pass


def create_dummy_forward_fn(input_infos: List[ModelInputInfo], with_input_tracing=False):
    from nncf.dynamic_graph.patch_pytorch import nncf_model_input

    def default_dummy_forward_fn(model):
        device = next(model.parameters()).device
        args_list = [create_mock_tensor(info, device) for info in input_infos if info.keyword is None]
        kwargs = {info.keyword: create_mock_tensor(info, device) for info in input_infos if info.keyword is not None}

        if with_input_tracing:
            for idx, tensor in enumerate(args_list):
                args_list[idx] = nncf_model_input(tensor)
            for key, tensor in kwargs.items():
                kwargs[key] = nncf_model_input(tensor)

        args = tuple(args_list)
        return model(*args, **kwargs)

    return default_dummy_forward_fn
