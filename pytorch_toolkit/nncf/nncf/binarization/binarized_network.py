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
from collections import OrderedDict
from texttable import Texttable
from typing import List

import torch
from torch import nn

from ..dynamic_graph.patch_pytorch import ignore_scope
from ..dynamic_graph.transform_graph import in_scope_list, replace_modules_by_nncf_modules
from ..layers import NNCF_MODULES, NNCFConv2d
from ..operations import UpdateWeight, UpdateInputs
from ..utils import get_all_modules_by_type
from .layers import ActivationBinarizationScaleThreshold
from nncf.dynamic_graph.graph_builder import ModelInputInfo

logger = logging.getLogger(__name__)


@ignore_scope
class BinarizedNetwork(nn.Module):
    def __init__(self, module, binarize_module_creator_fn, input_infos: List[ModelInputInfo],
                 ignored_scopes=None, target_scopes=None):
        super().__init__()
        self.input_infos = input_infos
        self.ignored_scopes = ignored_scopes
        self.target_scopes = target_scopes
        self.module = module
        self.binarize_module_creator_fn = binarize_module_creator_fn
        self.binarized_modules = OrderedDict()
        self._latest_inputs_outputs = None

        device = next(module.parameters()).device

        self._key_to_name = {}
        # all modules should be replaced prior to graph building
        self._replace_binarized_modules_by_nncf_modules(device)
        self._register_binarization_operations(device)
        self._compute_and_display_flops_binarization_rate()

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return out

    def _replace_binarized_modules_by_nncf_modules(self, device):
        self.module, _ = replace_modules_by_nncf_modules(self.module,
                                                         ignored_scopes=self.ignored_scopes,
                                                         target_scopes=self.target_scopes,
                                                         logger=logger)
        self.module = self.module.to(device)

    def _register_binarization_operation(self, module_name, module, device):
        if isinstance(module, torch.nn.modules.Conv2d):
            logger.info("Adding Weight binarizer in scope: {}".format(module_name))
            op_weights = UpdateWeight(
                self.binarize_module_creator_fn()
            ).to(device)
            module.register_pre_forward_operation(op_weights)

            logger.info("Adding Activation binarizer in scope: {}".format(module_name))
            op_inputs = UpdateInputs(ActivationBinarizationScaleThreshold(module.weight.shape))
            module.register_pre_forward_operation(op_inputs)

    def _register_binarization_operations(self, device):
        modules = get_all_modules_by_type(self.module, NNCF_MODULES)

        for name, module in modules.items():
            if in_scope_list(name, self.ignored_scopes):
                logger.info("Ignored adding Weight binarizer in scope: {}".format(name))
                continue

            if self.target_scopes is None or in_scope_list(name, self.target_scopes):
                self.binarized_modules[name] = module
                self._register_binarization_operation(name, module, device)

    def get_context_name(self):
        return None

    def _compute_and_display_flops_binarization_rate(self):
        net = self.module
        device = next(net.parameters()).device
        weight_list = {}
        state_dict = net.state_dict()
        for n, v in state_dict.items():
            weight_list[n] = v.clone()

        ops_dict = OrderedDict()

        def get_hook(name):
            def compute_flops_hook(self, input_, output):
                name_type = str(type(self).__name__)
                if isinstance(self, (nn.Conv2d, nn.ConvTranspose2d)):
                    ks = self.weight.data.shape
                    ops_count = ks[0] * ks[1] * ks[2] * ks[3] * output.shape[3] * output.shape[2]
                elif isinstance(self, nn.Linear):
                    ops_count = input_[0].shape[1] * output.shape[1]
                else:
                    return
                ops_dict[name] = (name_type, ops_count, isinstance(self, NNCFConv2d))

            return compute_flops_hook

        hook_list = [m.register_forward_hook(get_hook(n)) for n, m in net.named_modules()]

        input_shape = tuple([1] + list(self.input_infos[0].shape)[1:])
        var = torch.randn(*input_shape, device=device)

        net.eval()
        net(var)
        net.train()

        for h in hook_list:
            h.remove()

        # restore all parameters that can be corrupted due forward pass
        for n, v in state_dict.items():
            # print(n,v.shape,weight_list[n].shape, len(v.shape))
            state_dict[n].data.copy_(weight_list[n].data)

        ops_bin = 0
        ops_total = 0

        for layer_name, (layer_type, ops, is_binarized) in ops_dict.items():
            ops_total += ops
            if is_binarized:
                ops_bin += ops

        table = Texttable()
        header = ["Layer name", "Layer type", "Binarized", "MAC count", "MAC share"]
        table_data = [header]

        for layer_name, (layer_type, ops, is_binarized) in ops_dict.items():
            drow = {h: 0 for h in header}
            drow["Layer name"] = layer_name
            drow["Layer type"] = layer_type
            drow["Binarized"] = 'Y' if is_binarized else 'N'
            drow["MAC count"] = "{:.3f}G".format(ops*1e-9)
            drow["MAC share"] = "{:2.1f}%".format(ops / ops_total * 100)
            row = [drow[h] for h in header]
            table_data.append(row)

        table.add_rows(table_data)
        print(table.draw())
        print("Total binarized MAC share: {:.1f}%".format(ops_bin / ops_total * 100))
