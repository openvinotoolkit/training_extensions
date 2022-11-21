# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import time

from mmcv.runner import RUNNERS, EpochBasedRunner
from mmcv.runner.hooks.lr_updater import LrUpdaterHook
from mmcv.runner.utils import get_host_info

from otx.algorithms.common.adapters.mmcv.nncf.hooks import CompressionHook
from otx.algorithms.common.adapters.nncf import (
    AccuracyAwareLrUpdater,
    check_nncf_is_enabled,
)


# Try monkey patching to steal validation result
#from mmcv.runner.hooks import EvalHook
import mmcv.runner.hooks
old_evaluate = mmcv.runner.EvalHook.evaluate
def new_evaluate(self, runner, result):
    ret = old_evaluate(self, runner, result)
    setattr(runner, "all_metrics", copy.deepcopy(runner.log_buffer.output))
    return ret
mmcv.runner.EvalHook.evaluate = new_evaluate


@RUNNERS.register_module()
class AccuracyAwareRunner(EpochBasedRunner):
    """
    An mmcv training runner to be used with NNCF-based accuracy-aware training.
    Inherited from the standard EpochBasedRunner with the overridden "run" method.
    This runner does not use the "workflow" and "max_epochs" parameters that are
    used by the EpochBasedRunner since the training is controlled by NNCF's
    AdaptiveCompressionTrainingLoop that does the scheduling of the compression-aware
    training loop using the parameters specified in the "accuracy_aware_training".
    """

    def __init__(self, *args, target_metric_name, nncf_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_metric_name = target_metric_name
        self.nncf_config = nncf_config
        self.compression_ctrl = None

    def run(self, data_loaders, *args, **kwargs):
        check_nncf_is_enabled()

        from nncf.common.accuracy_aware_training import (
            create_accuracy_aware_training_loop,
        )

        assert isinstance(data_loaders, list)

        lr_update_hook = []
        found_compression_hook = False
        for hook in self.hooks:
            if isinstance(hook, LrUpdaterHook):
                lr_update_hook.append(hook)
            if isinstance(hook, CompressionHook):
                found_compression_hook = True
        assert (
            found_compression_hook
        ), f"{CompressionHook} must be registered to {self}."
        assert len(lr_update_hook) <= 1, (
            f"More than 1 lr update hooks ({len(lr_update_hook)} "
            f"are registered to {self}"
        )

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info(
            "Start running, host: %s, work_dir: %s", get_host_info(), work_dir
        )
        self.logger.warning(
            "Note that the workflow and max_epochs parameters "
            "are not used in NNCF-based accuracy-aware training"
        )

        # taking only the first data loader for NNCF training
        self.train_data_loader = data_loaders[0]
        # Maximum possible number of iterations, needs for progress tracking
        params = self.nncf_config["accuracy_aware_training"]["params"]
        self._max_epochs = params["maximal_total_epochs"]
        self._max_iters = self._max_epochs * len(self.train_data_loader)

        self.call_hook("before_run")

        def configure_optimizers_fn():
            return self.optimizer, None

        if len(lr_update_hook) == 1:
            lr_update_hook = lr_update_hook[0]

            def configure_optimizers_fn():
                return self.optimizer, AccuracyAwareLrUpdater(lr_update_hook)

        acc_aware_training_loop = create_accuracy_aware_training_loop(
            self.nncf_config, self.compression_ctrl, verbose=False
        )

        model = acc_aware_training_loop.run(
            self.model,
            train_epoch_fn=self.train_fn,
            validate_fn=self.validation_fn,
            configure_optimizers_fn=configure_optimizers_fn,
        )

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook("after_run")
        return model

    def train_fn(self, *args, **kwargs):
        """
        Train the model for a single epoch.
        This method is used in NNCF-based accuracy-aware training.
        """
        self.train(self.train_data_loader)

    def validation_fn(self, *args, **kwargs):
        """
        Return the target metric value on the validation dataset.
        Evaluation is assumed to be already done at this point since EvalHook was called.
        This method is used in NNCF-based accuracy-aware training.
        """
        # Get metric from runner's attributes that set in EvalHook.evaluate() function
        # metric = getattr(self, self.target_metric_name, None)
        all_metrics = getattr(self, "all_metrics", {})
        if len(all_metrics) == 0:
            return 0.0
        metric = all_metrics.get(self.target_metric_name, None)
        if metric is None:
            raise RuntimeError(f"Could not find the {self.target_metric_name} key")
        return metric
