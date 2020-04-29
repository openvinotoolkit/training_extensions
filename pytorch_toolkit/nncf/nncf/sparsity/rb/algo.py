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
from typing import List

import torch
import torch.distributed as dist

from nncf.utils import get_world_size
from nncf.nncf_network import NNCFNetwork
from nncf.compression_method_api import CompressionAlgorithmController
from nncf.sparsity.rb.layers import RBSparsifyingWeight
from nncf.sparsity.rb.loss import SparseLoss
from nncf.sparsity.base_algo import BaseSparsityAlgoBuilder, BaseSparsityAlgoController, SparseModuleInfo
from nncf.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.algo_selector import COMPRESSION_ALGORITHMS


@COMPRESSION_ALGORITHMS.register('rb_sparsity')
class RBSparsityBuilder(BaseSparsityAlgoBuilder):
    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        target_model = super().apply_to(target_model)
        return target_model

    def create_weight_sparsifying_operation(self, module):
        return RBSparsifyingWeight(module.weight.size(), sparsify=True)

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        params = self.config.get("params", {})
        return RBSparsityController(target_model, self._sparsified_module_info,
                                    params)


class RBSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork,
                 sparsified_module_info: List[SparseModuleInfo],
                 params):
        super().__init__(target_model, sparsified_module_info)

        self._distributed = False
        self._loss = SparseLoss()  # type: SparseLoss
        scheduler_cls = SPARSITY_SCHEDULERS.get(params.get("schedule", "exponential"))
        self._scheduler = scheduler_cls(self, params)
        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        self._loss.set_layers(sparsify_operations)
        self._check_sparsity_masks = params.get("check_sparsity_masks", False)

    def set_sparsity_level(self, sparsity_level):
        self._loss.target = 1 - sparsity_level

    def freeze(self):
        self._loss.disable()

    def distributed(self):
        if not dist.is_initialized():
            raise KeyError('Could not set distributed mode for the compression algorithm '
                           'because the default process group has not been initialized.')

        if next(self._model.parameters()).is_cuda:
            state = torch.cuda.get_rng_state()
            if dist.get_backend() == dist.Backend.NCCL:
                state = state.cuda()
            torch.distributed.broadcast(state, src=0)
            torch.cuda.set_rng_state(state.cpu())
        else:
            state = torch.get_rng_state()
            torch.distributed.broadcast(state, src=0)
            torch.set_rng_state(state)

        self._distributed = True

    def check_distributed_masks(self):
        if not self._distributed or get_world_size() == 1:
            return 1

        nvalues = 0
        ncor_values = 0
        eps = 1e-4
        for minfo in self.sparsified_module_info:
            mask = minfo.operand.mask

            mask_list = [torch.empty_like(mask) for _ in range(get_world_size())]
            # nccl does not support gather, send, recv operations
            dist.all_gather(mask_list, mask)

            for i in range(1, len(mask_list)):
                rel_error = (mask_list[0] - mask_list[i]) / mask_list[0]
                ncor_values = ncor_values + (rel_error.abs() < eps).sum(dtype=mask.dtype)
                nvalues = nvalues + mask_list[i].numel()

        return ncor_values / nvalues

    def add_algo_specific_stats(self, stats):
        stats["target_sparsity_rate"] = self.loss.target_sparsity_rate
        if self._distributed and self._check_sparsity_masks:
            stats["masks_consistents"] = self.check_distributed_masks()
        return stats
