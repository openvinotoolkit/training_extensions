"""AccuracyAwareRunner for NNCF task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import time
from dataclasses import asdict

from mmcv.runner import RUNNERS
from mmcv.runner.hooks.evaluation import EvalHook
from mmcv.runner.hooks.lr_updater import LrUpdaterHook
from mmcv.runner.utils import get_host_info

from otx.algorithms.common.adapters.mmcv.nncf.hooks import CompressionHook
from otx.algorithms.common.adapters.mmcv.runner import EpochRunnerWithCancel
from otx.algorithms.common.adapters.nncf import (
    AccuracyAwareLrUpdater,
    check_nncf_is_enabled,
)
from otx.algorithms.common.adapters.nncf.compression import NNCFMetaState

NNCF_META_KEY = "nncf_meta"


# TODO: refactoring
@RUNNERS.register_module()
class AccuracyAwareRunner(EpochRunnerWithCancel):  # pylint: disable=too-many-instance-attributes
    """AccuracyAwareRunner for NNCF task.

    An mmcv training runner to be used with NNCF-based accuracy-aware training.
    Inherited from the standard EpochBasedRunner with the overridden "run" method.
    This runner does not use the "workflow" and "max_epochs" parameters that are
    used by the EpochBasedRunner since the training is controlled by NNCF's
    AdaptiveCompressionTrainingLoop that does the scheduling of the compression-aware
    training loop using the parameters specified in the "accuracy_aware_training".
    """

    def __init__(self, *args, nncf_config, nncf_meta=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nncf_config = nncf_config

        if nncf_meta is None:
            nncf_meta = NNCFMetaState()
        self.nncf_meta = nncf_meta

        self.compression_ctrl = None
        self._target_metric_name = nncf_config["target_metric_name"]
        self._train_data_loader = None
        self._eval_hook = None

    def run(self, data_loaders, *args, **kwargs):  # pylint: disable=unused-argument
        """run."""
        check_nncf_is_enabled()

        from nncf.common.accuracy_aware_training import (
            create_accuracy_aware_training_loop,
        )

        assert isinstance(data_loaders, list)

        lr_update_hook = []
        eval_hook = []
        found_compression_hook = False
        for hook in self.hooks:
            if isinstance(hook, LrUpdaterHook):
                lr_update_hook.append(hook)
            if isinstance(hook, CompressionHook):
                found_compression_hook = True
            if isinstance(hook, EvalHook):
                eval_hook.append(hook)
        assert found_compression_hook, f"{CompressionHook} must be registered to {self}."
        assert len(lr_update_hook) <= 1, (
            f"More than 1 lr update hooks ({len(lr_update_hook)} " f"are registered to {self}"
        )
        assert len(eval_hook) == 1, f"{EvalHook} must be registered to {self}"
        self._eval_hook = eval_hook[0]
        assert self._eval_hook.save_best == self.nncf_config.target_metric_name, (
            "'target_metric_name' from nncf_config is not identical to 'save_best' in 'EvalHook'. "
            f"({self._eval_hook.save_best} != {self.nncf_config.target_metric_name})"
        )

        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info("Start running, host: %s, work_dir: %s", get_host_info(), work_dir)
        self.logger.warning(
            "Note that the workflow and max_epochs parameters are not used in NNCF-based accuracy-aware training"
        )

        # taking only the first data loader for NNCF training
        self._train_data_loader = data_loaders[0]
        # Maximum possible number of iterations, needs for progress tracking
        params = self.nncf_config["accuracy_aware_training"]["params"]
        self._max_epochs = params["maximal_total_epochs"]
        self._max_iters = self._max_epochs * len(self._train_data_loader)

        self.logger.info("Start running, host: %s, work_dir: %s", get_host_info(), work_dir)
        self.logger.info("Hooks will be executed in the following order:\n%s", self.get_hook_info())
        self.call_hook("before_run")

        def configure_optimizers_fn():
            return self.optimizer, None

        if len(lr_update_hook) == 1:
            lr_update_hook = lr_update_hook[0]

            def configure_optimizers_fn():  # noqa: F811  # pylint: disable=function-redefined
                return self.optimizer, AccuracyAwareLrUpdater(lr_update_hook)

        # pylint: disable-next=unused-argument
        def dump_checkpoint_fn(model, compression_ctrl, nncf_runner, save_dir):
            # pylint: disable-next=protected-access
            self._eval_hook._save_ckpt(self, nncf_runner.best_val_metric_value)
            return self._eval_hook.best_ckpt_path

        if hasattr(self.model, "module"):
            uncompressed_model_accuracy = self.model.module.nncf._uncompressed_model_accuracy
        else:
            uncompressed_model_accuracy = self.model.nncf._uncompressed_model_accuracy

        acc_aware_training_loop = create_accuracy_aware_training_loop(
            self.nncf_config,
            self.compression_ctrl,
            verbose=False,
            uncompressed_model_accuracy=uncompressed_model_accuracy,
        )

        model = acc_aware_training_loop.run(
            self.model,
            train_epoch_fn=self.train_fn,
            validate_fn=self.validation_fn,
            configure_optimizers_fn=configure_optimizers_fn,
            dump_checkpoint_fn=dump_checkpoint_fn,
            log_dir=self.work_dir,
        )

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook("after_run")
        return model

    def train_fn(self, *args, **kwargs):  # pylint: disable=unused-argument
        """train_fn.

        Train the model for a single epoch.
        This method is used in NNCF-based accuracy-aware training.
        """
        self.train(self._train_data_loader)

    def validation_fn(self, *args, **kwargs):  # pylint: disable=unused-argument
        """validation_fn.

        Return the target metric value on the validation dataset.
        This method is used in NNCF-based accuracy-aware training.
        """

        # make sure evaluation hook is in a 'should_evaluate' state
        interval_bak = self._eval_hook.interval
        self._eval_hook.interval = 1
        self._eval_hook._do_evaluate(self)  # pylint: disable=protected-access
        self._eval_hook.interval = interval_bak
        # Get metric from runner's attributes that set in EvalHook.evaluate() function
        all_metrics = getattr(self, "all_metrics", {})
        metric = all_metrics.get(self._target_metric_name, None)
        if metric is None:
            raise RuntimeError(f"Could not find the {self._target_metric_name} key")
        return metric

    def save_checkpoint(self, *args, **kwargs) -> None:
        """Save checkpoint with NNCF meta state."""

        compression_state = self.compression_ctrl.get_compression_state()
        for algo_state in compression_state.get("ctrl_state", {}).values():
            if not algo_state.get("scheduler_state"):
                algo_state["scheduler_state"] = {"current_step": 0, "current_epoch": 0}

        nncf_meta = NNCFMetaState(
            **{**asdict(self.nncf_meta), "compression_ctrl": compression_state},
        )

        meta = kwargs.pop("meta", {})
        meta[NNCF_META_KEY] = nncf_meta
        meta["nncf_enable_compression"] = True
        super().save_checkpoint(*args, **kwargs, meta=meta)
