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

from collections import namedtuple

from texttable import Texttable

from nncf.compression_method_api import CompressionAlgorithm
from nncf.dynamic_graph.transform_graph import replace_modules_by_nncf_modules
from nncf.layers import NNCF_MODULES, COMPRESSION_MODULES
from nncf.operations import UpdateWeight
from nncf.utils import get_all_modules_by_type, in_scope_list

SparseModuleInfo = namedtuple('SparseModuleInfo', ['module_name', 'module', 'operand'])


class BaseSparsityAlgo(CompressionAlgorithm):
    def freeze(self):
        raise NotImplementedError

    def set_sparsity_level(self, sparsity_level):
        raise NotImplementedError

    def _replace_sparsifying_modules_by_nncf_modules(self, device, ignored_scope, logger):
        self._model = replace_modules_by_nncf_modules(self._model, ignored_scope=ignored_scope, logger=logger)
        self._model.to(device)

    def _register_weight_sparsifying_operations(self, device, ignored_scope, logger):
        sparsified_modules = get_all_modules_by_type(self._model, NNCF_MODULES)
        self.sparsified_module_info = []
        for module_name, module in sparsified_modules.items():
            if in_scope_list(module_name, ignored_scope):
                logger.info("Ignored adding Weight Sparsifier in scope: {}".format(module_name))
                continue

            logger.info("Adding Weight Sparsifier in scope: {}".format(module_name))
            operation = self.create_weight_sparsifying_operation(module)
            opid = module.register_pre_forward_operation(UpdateWeight(operation).to(device))
            self.sparsified_module_info.append(SparseModuleInfo(module_name, module, module.get_pre_op(opid).operand))

    def create_weight_sparsifying_operation(self, module):
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

        for m in self.model.modules():
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
