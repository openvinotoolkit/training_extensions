"""
 Copyright (c) 2020 Intel Corporation
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
from typing import List, Dict

from functools import partial, update_wrapper
from texttable import Texttable
from torch import nn

from nncf.compression_method_api import CompressionAlgorithmBuilder, \
    CompressionAlgorithmController
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext
from nncf.module_operations import UpdateWeight
from nncf.nncf_network import NNCFNetwork, InsertionPoint, InsertionCommand, InsertionType, OperationPriority
from nncf.pruning.filter_pruning.layers import apply_filter_binary_mask
from nncf.pruning.utils import get_bn_for_module_scope, \
    get_first_pruned_modules, get_last_pruned_modules, is_conv_with_downsampling

from nncf.nncf_logger import logger as nncf_logger


class PrunedModuleInfo:
    BN_MODULE_NAME = 'bn_module'

    def __init__(self, module_name: str, module: nn.Module, operand, related_modules: Dict):
        self.module_name = module_name
        self.module = module
        self.operand = operand
        self.related_modules = related_modules


class BasePruningAlgoBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)
        params = config.get('params', {})
        self._params = params

        self.prune_first = params.get('prune_first_conv', False)
        self.prune_last = params.get('prune_last_conv', False)
        self.prune_batch_norms = params.get('prune_batch_norms', False)
        self.prune_downsample_convs = params.get('prune_downsample_convs', False)

        self._pruned_module_info = []

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        insertion_commands = self._prune_weights(target_model)
        for command in insertion_commands:
            target_model.register_insertion_command(command)
        target_model.register_algorithm(self)
        return target_model

    def _prune_weights(self, target_model: NNCFNetwork):
        device = next(target_model.parameters()).device
        modules_to_prune = target_model.get_nncf_modules()
        insertion_commands = []

        input_non_pruned_modules = get_first_pruned_modules(target_model,
                                                            self.get_types_of_pruned_modules() + ['linear'])
        output_non_pruned_modules = get_last_pruned_modules(target_model,
                                                            self.get_types_of_pruned_modules() + ['linear'])

        for module_scope, module in modules_to_prune.items():
            # Check that we need to prune weights in this op
            if not self._is_pruned_module(module):
                continue

            module_scope_str = str(module_scope)
            if not self._should_consider_scope(module_scope_str):
                nncf_logger.info("Ignored adding Weight Pruner in scope: {}".format(module_scope_str))
                continue

            if not self.prune_first and module in input_non_pruned_modules:
                nncf_logger.info("Ignored adding Weight Pruner in scope: {} because"
                                 " this scope is one of the first convolutions".format(module_scope_str))
                continue
            if not self.prune_last and module in output_non_pruned_modules:
                nncf_logger.info("Ignored adding Weight Pruner in scope: {} because"
                                 " this scope is one of the last convolutions".format(module_scope_str))
                continue

            if not self.prune_downsample_convs and is_conv_with_downsampling(module):
                nncf_logger.info("Ignored adding Weight Pruner in scope: {} because"
                                 " this scope is convolution with downsample".format(module_scope_str))
                continue

            nncf_logger.info("Adding Weight Pruner in scope: {}".format(module_scope_str))
            operation = self.create_weight_pruning_operation(module)
            hook = UpdateWeight(operation).to(device)
            insertion_commands.append(
                InsertionCommand(
                    InsertionPoint(
                        InputAgnosticOperationExecutionContext("", module_scope, 0),
                        InsertionType.NNCF_MODULE_PRE_OP
                    ),
                    hook,
                    OperationPriority.PRUNING_PRIORITY
                )
            )

            related_modules = {}
            if self.prune_batch_norms:
                related_modules['bn_module'] = get_bn_for_module_scope(target_model, module_scope)

            self._pruned_module_info.append(
                PrunedModuleInfo(module_scope_str, module, hook.operand, related_modules))

        return insertion_commands

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return BasePruningAlgoController(target_model, self._pruned_module_info, self._params)

    def create_weight_pruning_operation(self, module):
        raise NotImplementedError

    def _is_pruned_module(self, module: nn.Module):
        """
        Return whether this module should be pruned or not.
        """
        raise NotImplementedError

    def get_types_of_pruned_modules(self):
        """
        Returns list of operation types that should be pruned.
        """
        raise NotImplementedError


class BasePruningAlgoController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork,
                 pruned_module_info: List[PrunedModuleInfo], params: dict):
        super().__init__(target_model)
        self.pruned_module_info = pruned_module_info
        self.prune_first = params.get('prune_first_conv', False)
        self.prune_last = params.get('prune_last_conv', False)
        self.prune_batch_norms = params.get('prune_batch_norms', False)
        self.zero_grad = params.get('zero_grad', True)
        self._hooks = []

    def freeze(self):
        raise NotImplementedError

    def set_pruning_rate(self, pruning_rate):
        raise NotImplementedError

    def zero_grads_for_pruned_modules(self):
        """
        This function register hook that will set gradients for pruned filters to zero.
        """
        self._clean_hooks()

        def hook(grad, mask):
            mask = mask.to(grad.device)
            return apply_filter_binary_mask(mask, grad)

        for minfo in self.pruned_module_info[:1]:
            mask = minfo.operand.binary_filter_pruning_mask
            weight = minfo.module.weight
            partial_hook = update_wrapper(partial(hook, mask=mask), hook)
            self._hooks.append(weight.register_hook(partial_hook))
            if minfo.module.bias is not None:
                bias = minfo.module.bias
                partial_hook = update_wrapper(partial(hook, mask=mask), hook)
                self._hooks.append(bias.register_hook(partial_hook))

    def _clean_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _get_mask(self, minfo: PrunedModuleInfo):
        """
        Returns pruning mask for minfo.module.
        """
        raise NotImplementedError

    @staticmethod
    def pruning_rate_for_weight(minfo: PrunedModuleInfo):
        """
        Calculates sparsity rate for all weight elements.
        """
        weight = minfo.module.weight
        pruning_rate = 1 - weight.nonzero().size(0) / weight.view(-1).size(0)
        return pruning_rate

    @staticmethod
    def pruning_rate_for_filters(minfo: PrunedModuleInfo):
        """
        Calculates sparsity rate for weight filter-wise.
        """
        weight = minfo.module.weight
        filters_sum = weight.view(weight.size(0), -1).sum(axis=1)
        pruning_rate = 1 - len(filters_sum.nonzero()) / filters_sum.size(0)
        return pruning_rate

    def pruning_rate_for_mask(self, minfo: PrunedModuleInfo):
        mask = self._get_mask(minfo)
        pruning_rate = mask.nonzero().size(0) / max(mask.view(-1).size(0), 1)
        return pruning_rate

    def mask_shape(self, minfo: PrunedModuleInfo):
        mask = self._get_mask(minfo)
        return mask.shape

    def statistics(self):
        stats = super().statistics()
        table = Texttable()
        header = ["Name", "Weight's Shape", "Mask Shape", "Mask zero %", "PR", "Filter PR"]
        data = [header]

        for minfo in self.pruned_module_info:
            drow = {h: 0 for h in header}
            drow["Name"] = minfo.module_name
            drow["Weight's Shape"] = list(minfo.module.weight.size())

            drow["Mask Shape"] = list(self.mask_shape(minfo))

            drow["Mask zero %"] = 1.0 - self.pruning_rate_for_mask(minfo)

            drow["PR"] = self.pruning_rate_for_weight(minfo)

            drow["Filter PR"] = self.pruning_rate_for_filters(minfo)

            row = [drow[h] for h in header]
            data.append(row)
        table.add_rows(data)

        stats["pruning_statistic_by_module"] = table
        return self.add_algo_specific_stats(stats)

    @staticmethod
    def add_algo_specific_stats(stats):
        return stats
