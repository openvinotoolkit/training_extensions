"""Algorithm to find a proper batch size which is fit to current GPU device for tasks using mmcv."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Callable, Dict, List
from copy import deepcopy

from otx.algorithms.common.adapters.torch.utils import adapt_batch_size as adapt_torch_model_bs
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


def adapt_batch_size(train_func: Callable, cfg, meta: Dict, datasets: List):
    """Decrease batch size if default batch size isn't fit to current GPU device.

    This function just setup for single iteration training to reduce time for adapting.
    The core part of adapting batch size is done in adapt_batch_size in the torch.utils package.

    Args:
        train_func (Callable): The function to train a model.
            Only cfg, dataset and meta are passed to the function when invoking it.
        cfg: Configuration of a training.
        meta (Dict): A dict records some meta information of a training.
        datasets (List): List of datasets.
    """
    def train_func_single_iter(batch_size):
        copied_cfg = deepcopy(cfg)
        copied_meta = deepcopy(meta)

        copied_cfg.data.train_dataloader['samples_per_gpu'] = batch_size
        
        # setup for training a single iter to reduce time
        copied_cfg.runner["max_epochs"] = 1
        copied_meta["run_single_iter"] = True
        for hook in copied_cfg.custom_hooks:
            if hook["type"] == "AdaptiveTrainSchedulingHook":
                hook["enable_eval_before_run"] = False

        train_func(
            dataset=datasets,
            cfg=copied_cfg,
            meta=copied_meta,
        )

    available_bs =  adapt_torch_model_bs(
        train_func=train_func_single_iter,
        default_bs=cfg.data.train_dataloader['samples_per_gpu'],
        trainset_size=len(datasets[0])
    )
    cfg.data.train_dataloader['samples_per_gpu'] = available_bs
    logger.info(f"batch size is set as {available_bs} after adapting.")
