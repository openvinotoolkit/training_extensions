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
from collections import OrderedDict
from typing import List

import torch
from texttable import Texttable
from torch import nn

from nncf.nncf_network import NNCFNetwork
from nncf.config import Config
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext
from nncf.layers import NNCFConv2d
from nncf.nncf_network import InsertionCommand, InsertionPoint, InsertionType, OperationPriority
from nncf.module_operations import UpdateWeight, UpdateInputs
from nncf.binarization.layers import BINARIZATION_MODULES, BinarizationMode, WeightBinarizer, ActivationBinarizer, \
    ActivationBinarizationScaleThreshold
from nncf.binarization.schedulers import BINARIZATION_SCHEDULERS
from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.compression_method_api import CompressionAlgorithmBuilder, CompressionAlgorithmController

from nncf.nncf_logger import logger as nncf_logger


@COMPRESSION_ALGORITHMS.register('binarization')
class BinarizationBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.mode = self.config.get('mode', BinarizationMode.XNOR)

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        insertion_commands = self._binarize_weights_and_module_inputs(target_model)
        for command in insertion_commands:
            target_model.register_insertion_command(command)

        target_model.register_algorithm(self)
        return target_model

    def __create_binarize_module(self):
        return BINARIZATION_MODULES.get(self.mode)()

    def _binarize_weights_and_module_inputs(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        device = next(target_model.parameters()).device
        modules = target_model.get_nncf_modules()

        insertion_commands = []
        for scope, module in modules.items():
            scope_str = str(scope)

            if not self._should_consider_scope(scope_str):
                nncf_logger.info("Ignored adding binarizers in scope: {}".format(scope_str))
                continue

            if isinstance(module, torch.nn.modules.Conv2d):
                nncf_logger.info("Adding Weight binarizer in scope: {}".format(scope_str))
                op_weights = UpdateWeight(
                    self.__create_binarize_module()
                ).to(device)

                nncf_logger.info("Adding Activation binarizer in scope: {}".format(scope_str))
                op_inputs = UpdateInputs(ActivationBinarizationScaleThreshold(module.weight.shape)).to(device)

                insertion_commands.append(InsertionCommand(
                    InsertionPoint(
                        InputAgnosticOperationExecutionContext("", scope, 0),
                        InsertionType.NNCF_MODULE_PRE_OP), op_weights, OperationPriority.QUANTIZATION_PRIORITY))

                insertion_commands.append(InsertionCommand(
                    InsertionPoint(
                        InputAgnosticOperationExecutionContext("", scope, 0),
                        InsertionType.NNCF_MODULE_PRE_OP), op_inputs, OperationPriority.QUANTIZATION_PRIORITY))
        return insertion_commands

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return BinarizationController(target_model, self.config)


class BinarizationController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork, params: Config):
        super().__init__(target_model)

        self.is_distributed = False
        scheduler_cls = BINARIZATION_SCHEDULERS.get("staged")
        self._scheduler = scheduler_cls(self, params)
        self._compute_and_display_flops_binarization_rate()

    def distributed(self):
        self.is_distributed = True

    def enable_activation_binarization(self):
        if self._model is not None:
            for _, m in self._model.named_modules():
                if isinstance(m, ActivationBinarizer):
                    m.enable()

    def enable_weight_binarization(self):
        if self._model is not None:
            for _, m in self._model.named_modules():
                if isinstance(m, WeightBinarizer):
                    m.enable()

    def _compute_and_display_flops_binarization_rate(self):
        net = self._model
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

        net.do_dummy_forward(force_eval=True)

        for h in hook_list:
            h.remove()

        # restore all parameters that can be corrupted due forward pass
        for n, v in state_dict.items():
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
        nncf_logger.info(table.draw())
        nncf_logger.info("Total binarized MAC share: {:.1f}%".format(ops_bin / ops_total * 100))
