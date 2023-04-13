from typing import Callable, Dict
from copy import deepcopy

from otx.algorithms.common.adapters.torch.utils import adapt_batch_size as adapt_torch_model_bs
from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


def adapt_batch_size(train_func: Callable, cfg, meta: Dict, datasets):
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
