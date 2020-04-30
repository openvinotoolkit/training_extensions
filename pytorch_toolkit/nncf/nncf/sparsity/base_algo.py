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

from collections import namedtuple
from typing import List


from texttable import Texttable

from nncf.nncf_network import NNCFNetwork
from nncf.compression_method_api import CompressionAlgorithmBuilder, CompressionAlgorithmController
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext
from nncf.layer_utils import COMPRESSION_MODULES
from nncf.nncf_network import InsertionCommand, InsertionPoint, InsertionType, OperationPriority
from nncf.module_operations import UpdateWeight
from nncf.nncf_logger import logger as nncf_logger

SparseModuleInfo = namedtuple('SparseModuleInfo', ['module_name', 'module', 'operand'])


class BaseSparsityAlgoBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)
        self._sparsified_module_info = []

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        insertion_commands = self._sparsify_weights(target_model)
        for command in insertion_commands:
            target_model.register_insertion_command(command)
        target_model.register_algorithm(self)
        return target_model

    def _sparsify_weights(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        device = next(target_model.parameters()).device
        sparsified_modules = target_model.get_nncf_modules()
        insertion_commands = []
        for module_scope, module in sparsified_modules.items():
            scope_str = str(module_scope)

            if not self._should_consider_scope(scope_str):
                nncf_logger.info("Ignored adding Weight Sparsifier in scope: {}".format(scope_str))
                continue

            nncf_logger.info("Adding Weight Sparsifier in scope: {}".format(scope_str))
            operation = self.create_weight_sparsifying_operation(module)
            hook = UpdateWeight(operation).to(device)
            insertion_commands.append(InsertionCommand(InsertionPoint(
                InputAgnosticOperationExecutionContext("", module_scope, 0),
                InsertionType.NNCF_MODULE_PRE_OP),
                                                       hook,
                                                       OperationPriority.SPARSIFICATION_PRIORITY))
            self._sparsified_module_info.append(
                SparseModuleInfo(scope_str, module, hook.operand))

        return insertion_commands

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return BaseSparsityAlgoController(target_model, self._sparsified_module_info)

    def create_weight_sparsifying_operation(self, target_module):
        raise NotImplementedError


class BaseSparsityAlgoController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork,
                 sparsified_module_info: List[SparseModuleInfo]):
        super().__init__(target_model)
        self.sparsified_module_info = sparsified_module_info

    def freeze(self):
        raise NotImplementedError

    def set_sparsity_level(self, sparsity_level):
        raise NotImplementedError

    @property
    def sparsified_weights_count(self):
        count = 0
        for minfo in self.sparsified_module_info:
            count = count + minfo.module.weight.view(-1).size(0)
        return max(count, 1)

    @property
    def sparsity_rate_for_sparsified_modules(self):
        nonzero = 0
        count = 0

        for minfo in self.sparsified_module_info:
            mask = minfo.operand.apply_binary_mask(minfo.module.weight)
            nonzero = nonzero + mask.nonzero().size(0)
            count = count + mask.view(-1).size(0)

        return 1 - nonzero / max(count, 1)

    @property
    def sparsity_rate_for_model(self):
        nonzero = 0
        count = 0

        for m in self._model.modules():
            if isinstance(m, tuple(COMPRESSION_MODULES.registry_dict.values())):
                continue

            sparsified_module = False
            for minfo in self.sparsified_module_info:
                if minfo.module == m:
                    mask = minfo.operand.apply_binary_mask(m.weight)
                    nonzero = nonzero + mask.nonzero().size(0)
                    count = count + mask.numel()

                    if not m.bias is None:
                        nonzero = nonzero + m.bias.nonzero().size(0)
                        count = count + m.bias.numel()

                    sparsified_module = True

            if not sparsified_module:
                for param in m.parameters(recurse=False):
                    nonzero = nonzero + param.nonzero().size(0)
                    count = count + param.numel()

        return 1 - nonzero / max(count, 1)

    def statistics(self):
        stats = super().statistics()
        table = Texttable()
        header = ["Name", "Weight's Shape", "SR", "% weights"]
        data = [header]

        sparsified_weights_count = self.sparsified_weights_count

        for minfo in self.sparsified_module_info:
            drow = {h: 0 for h in header}
            drow["Name"] = minfo.module_name
            drow["Weight's Shape"] = list(minfo.module.weight.size())
            mask = minfo.operand.apply_binary_mask(minfo.module.weight)
            nonzero = mask.nonzero().size(0)
            drow["SR"] = 1.0 - nonzero / max(mask.view(-1).size(0), 1)
            drow["% weights"] = mask.view(-1).size(0) / sparsified_weights_count
            row = [drow[h] for h in header]
            data.append(row)
        table.add_rows(data)

        stats["sparsity_statistic_by_module"] = table
        stats["sparsity_rate_for_sparsified_modules"] = self.sparsity_rate_for_sparsified_modules
        stats["sparsity_rate_for_model"] = self.sparsity_rate_for_model

        return self.add_algo_specific_stats(stats)

    def add_algo_specific_stats(self, stats):
        return stats
