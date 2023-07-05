import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from otx.v2.adapters.torch.mmcv.registry import MMCVRegistry
from otx.v2.api.core.engine import Engine
from otx.v2.api.utils.decorators import set_default_argument
from otx.v2.api.utils.logger import get_logger
from torch.utils.data import DataLoader

from mmcv.runner import BaseRunner, Hook, wrap_fp16_model
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.utils import collect_env

logger = get_logger()


class MMXEngine(Engine):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.module_registry = MMCVRegistry()

    @set_default_argument(BaseRunner.__init__)
    def train(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        work_dir: str,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = 20,
        distributed: bool = False,
        seed: Optional[int] = None,
        deterministric: bool = False,
        precision: str = "32",
        # Validation (Dependent on val_dataloader)
        eval_interval: int = 1,
        eval_metric: Optional[Union[str, List[str]]] = ["accuracy", "class_accuracy"],
        # Hooks for mmX
        lr_hooks: Optional[Union[Dict, Hook]] = None,
        optimizer_hooks: Optional[Union[Dict, Hook]] = None,
        checkpoint_hooks: Optional[Union[Dict, Hook]] = None,
        log_hooks: Optional[Dict] = None,
        momentum_hooks: Optional[Union[Dict, Hook]] = None,
        # timer_hooks: Optional[Union[Dict, Hook]] = dict(type="IterTimerHook"),  # TODO: This only exists for EpochRunnerWithCancel.
        custom_hooks: Optional[Union[List, Dict, Hook]] = None,
        **kwargs,
    ):
        # Validation of params
        if max_iters is not None and max_epochs is not None:
            raise ValueError("Only one of `max_epochs` or `max_iters` can be set.")

        # Set Logger
        env_info_dict = collect_env()
        "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])

        # Set Runners
        if max_epochs:
            base_runner = self.module_registry.get("EpochRunnerWithCancel")
            self._by_epoch = True
            kwargs["max_epochs"] = max_epochs
        else:
            base_runner = self.module_registry.get("IterBasedRunnerWithCancel")
            self._by_epoch = False
            kwargs["max_iters"] = max_iters
        if base_runner is None:
            raise ValueError("Need to check Module Registry.")

        # Model Wrapping
        if precision.upper() in ["FP16", "16"]:
            wrap_fp16_model(model)

        if checkpoint_hooks is None:
            checkpoint_hooks = CheckpointHook(interval=1, out_dir=work_dir)
        if hasattr(checkpoint_hooks, "by_epoch"):
            checkpoint_hooks.by_epoch = self._by_epoch

        # Set kwargs
        kwargs["work_dir"] = work_dir
        kwargs["logger"] = logger

        runner = base_runner(model=model, optimizer=optimizer, **kwargs)
        runner.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # Register Training Hooks
        if val_dataloader:
            base_eval_hook = (
                self.module_registry.get("DistCustomEvalHook")
                if distributed
                else self.module_registry.get("CustomEvalHook")
            )
            eval_hook = base_eval_hook(
                dataloader=val_dataloader,
                start=kwargs.get("eval_start", None),
                interval=eval_interval,
                by_epoch=self._by_epoch,
                test_fn=kwargs.get("test_fn", None),
                greater_keys=kwargs.get("greater_keys", None),
                less_keys=kwargs.get("less_keys", None),
                out_dir=work_dir,
                file_client_args=kwargs.get("file_client_args", None),
                ema_eval_start_epoch=kwargs.get("ema_eval_start_epoch", 10),
                metric=eval_metric,
            )
            runner.register_hook(eval_hook, priority="ABOVE_NORMAL")
        runner.register_training_hooks(
            lr_config=lr_hooks,
            optimizer_config=optimizer_hooks,
            checkpoint_config=checkpoint_hooks,
            log_config=log_hooks,
            momentum_config=momentum_hooks,
            # timer_config=timer_hooks,  # TODO: This only exists for EpochRunnerWithCancel.
            custom_hooks_config=custom_hooks,
        )
        if distributed and self._by_epoch:
            dist_sampler_seed_hook = self.module_registry.get("DistSamplerSeedHook")
            runner.register_hook(dist_sampler_seed_hook())

        runner.run(
            data_loaders=[train_dataloader],
            workflow=[("train", 1)],
        )
        # Save outputs
        output_ckpt_path = Path(work_dir) / "latest.pth"
        best_ckpt_path = glob.glob(str(Path(work_dir) / "best_*.pth"))
        if len(best_ckpt_path) > 0:
            output_ckpt_path = best_ckpt_path[0]
        return output_ckpt_path

    def infer(self, model: torch.nn.Module, img):
        raise NotImplementedError()

    def evaluate(self, **params):
        raise NotImplementedError()
