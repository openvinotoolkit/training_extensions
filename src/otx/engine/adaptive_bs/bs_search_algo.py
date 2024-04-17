"""Algorithm to find a proper batch size which is fit to current device."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from copy import deepcopy
import lightning.pytorch as pl
from lightning.pytorch.utilities.exceptions import MisconfigurationException, _TunerExitException
from typing import Any, TYPE_CHECKING

import torch
from lightning.pytorch.callbacks.callback import Callback

from otx.utils.utils import is_xpu_available

if TYPE_CHECKING:
    from lightning import Trainer

    from otx.core.model.module.base import OTXLitModule
    from otx.core.data.module import OTXDataModule

logger = logging.getLogger(__name__)


def _run_trial(trainer: Trainer, model, datamodule, bs: int, trial_queue: mp.Queue) -> None:
    mp.set_start_method(None, True)  # reset mp start method

    datamodule.config.train_subset.batch_size = bs

    oom = False
    try:
        batch_size_finder: Callback = BatchSizeFinder(steps_per_trial=3)
        # do not continue with the loop in case Tuner is used
        batch_size_finder._early_exit = True
        trainer.callbacks = [batch_size_finder] + trainer.callbacks

        trainer.fit(model=model, datamodule=datamodule)
    except RuntimeError as e:
        if str(e).startswith("CUDA out of memory.") or str(e).startswith(  # CUDA OOM
            "Allocation is out of device memory on current platform."  # XPU OOM
        ):
            oom = True
        else:
            raise e

    print("*"*100, _get_max_memory_reserved())
    trial_queue.put(
        {
            "oom": oom,
            "max_memory_reserved": _get_max_memory_reserved(),
        }
    )


class BsSearchAlgo:
    """Algorithm class to find optimal batch size.

    Args:
        train_func (Callable[[int], None]): Training function with single arugment to set batch size.
        train_func_kwargs (Dict[str, Any]): Keyword arguments for train_func.
        default_bs (int): Default batch size. It should be bigger than 0.
        max_bs (int): Maximum batch size. It should be bigger than 0.
    """

    def __init__(self, trainer: Trainer, model: OTXLitModule, datamodule: OTXDataModule, max_bs: int):
        if default_bs <= 0:
            raise ValueError("Batch size should be bigger than 0.")
        if max_bs <= 0:
            raise ValueError("train data set size should be bigger than 0.")

        if max_bs < default_bs:
            default_bs = max_bs

        self._trainer = trainer
        self._model = model
        self._datamodule = datamodule
        self._default_bs = default_bs
        self._max_bs = max_bs
        self._bs_try_history: dict[int, int] = {}
        self._total_mem = _get_total_memory_size()
        self._mem_lower_bound = 0.8 * self._total_mem
        self._mem_upper_bound = 0.85 * self._total_mem
        self._mp_ctx = mp.get_context("spawn")

    def _try_batch_size(self, bs: int) -> tuple[bool, int]:
        trial_queue = self._mp_ctx.Queue()
        proc = self._mp_ctx.Process(
            target=_run_trial, args=(self._trainer, self._model, self._datamodule, bs, trial_queue)
        )
        proc.start()
        output = None
        while proc.is_alive():
            try:
                output = trial_queue.get(timeout=1)
                break
            except queue.Empty:
                pass
        proc.join()
        if output is None:
            msg = "There is no output from the trial for adaptive batch size."
            raise RuntimeError(msg)

        oom = output["oom"]
        max_memory_reserved = output["max_memory_reserved"]

        if not oom:
            self._bs_try_history[bs] = max_memory_reserved

        # logger.debug(
        logger.warning(
            f"Adapting Batch size => bs : {bs}, OOM : {oom}, "
            f"memory usage : {max_memory_reserved / self._total_mem}%"
        )

        return oom, max_memory_reserved

    @staticmethod
    def _get_even_center_val(val1: int, val2: int) -> int:
        ret = (val1 + val2) // 2
        if ret % 2 == 1:
            ret += 1
        return ret

    def auto_decrease_batch_size(self) -> int:
        """Decrease batch size if default batch size isn't fit to current device.

        Returns:
            int: Proper batch size possibly decreased as default value isn't fit
        """
        available_bs = 0
        current_bs = self._default_bs
        lowest_unavailable_bs = self._default_bs + 2

        while True:
            oom, max_memory_reserved = self._try_batch_size(current_bs)

            # If memory usage is too close to limit, OOM can be raised during training
            if oom or max_memory_reserved > self._mem_upper_bound:
                if current_bs < lowest_unavailable_bs:
                    lowest_unavailable_bs = current_bs
                current_bs = self._get_even_center_val(current_bs, available_bs)
            else:
                available_bs = current_bs
                current_bs = self._get_even_center_val(current_bs, lowest_unavailable_bs)

            if lowest_unavailable_bs - available_bs <= 2:
                break

        if available_bs == 0:
            raise RuntimeError("Current device can't train model even with 2.")

        return available_bs

    def find_big_enough_batch_size(self, drop_last: bool = False) -> int:
        """Find a big enough batch size.

        This function finds a big enough batch size by training with various batch sizes.
        It estimate a batch size using equation is estimated using training history.
        The reason why using the word "big enough" is that it tries to find not maxmium but big enough value which uses
        memory between lower and upper bound.

        Args:
            drop_last (bool): Whether to drop the last incomplete batch.

        Raises:
            RuntimeError: If training with batch size 2 can't be run, raise an error.

        Returns:
            int: Big enough batch size.
        """
        estimated_bs = self._default_bs

        # try default batch size
        oom, bs_mem_usage = self._try_batch_size(estimated_bs)
        if oom or bs_mem_usage > self._mem_upper_bound:
            self._default_bs -= 2
            if self._default_bs <= 0:
                raise RuntimeError("Current device can't train model even with 2.")

            return self.auto_decrease_batch_size()

        # try default batch size + 2
        estimated_bs += 2
        if estimated_bs > self._max_bs:
            return self._default_bs
        oom, bs_mem_usage = self._try_batch_size(estimated_bs)
        if oom or bs_mem_usage > self._mem_upper_bound:
            return self._default_bs

        # estimate batch size using equation
        estimation_pct = 0.82
        while True:
            estimated_bs = self._estimate_batch_size(estimation_pct)
            if estimated_bs in self._bs_try_history:
                break
            oom, mem_usage = self._try_batch_size(estimated_bs)

            if oom:
                estimation_pct -= 0.1
                if estimation_pct <= 0:
                    estimated_bs = self._default_bs + 2
                    break
            elif self._mem_lower_bound <= mem_usage <= self._mem_upper_bound:
                break
            else:
                estimation_pct = 0.82

        if drop_last and (self._max_bs // 2 < estimated_bs < self._max_bs):
            estimated_bs = self._max_bs // 2

        return estimated_bs

    def _estimate_batch_size(self, estimation_pct: float) -> int:
        if len(self._bs_try_history) < 2:
            raise RuntimeError("At least two trials should be done without OOM to estimate batch size.")

        def distance_from_bound(val):
            if val[1] < self._mem_lower_bound:
                # if memory usage is same, then higher batch size is preferred
                return self._mem_lower_bound - val[1] - val[0] / 10000
            elif self._mem_upper_bound < val[1]:
                # if memory usage is same, then lower batch size is preferred
                return val[1] - self._mem_upper_bound + val[0] / 10000
            else:
                return min(abs(self._mem_lower_bound - val[1], abs(val[1] - self._mem_upper_bound)))

        bs_arr = sorted([(bs, mem_usage) for bs, mem_usage in self._bs_try_history.items()], key=lambda x: x[0])
        for idx in range(len(bs_arr) - 1, -1, -1):
            if bs_arr[idx][1] < self._mem_upper_bound:
                cur_max_bs_idx = idx
                break
        else:
            logger.warning("All batch size tried used more memory size than upper bound.")
            return bs_arr[0][0]

        def check_bs_suitable(estimated_bs) -> bool:
            # Check batch size is between largest bs which uses lower memory than uppper bound
            # and smallest bs which uses higher memory than upper bound.
            if estimated_bs >= bs_arr[cur_max_bs_idx][0]:
                if cur_max_bs_idx + 1 < len(bs_arr):
                    if estimated_bs < bs_arr[cur_max_bs_idx + 1][0]:
                        return True
                else:
                    return True
            return False

        x_idx, y_idx = 0, len(bs_arr) - 1

        while x_idx < y_idx:
            graident = (bs_arr[y_idx][1] - bs_arr[x_idx][1]) / (bs_arr[y_idx][0] - bs_arr[x_idx][0])
            b = bs_arr[y_idx][1] - graident * bs_arr[y_idx][0]
            if graident != 0:
                estimated_bs = round(((self._total_mem * estimation_pct) - b) / (graident * 2)) * 2
                if check_bs_suitable(estimated_bs):
                    break

            if distance_from_bound(bs_arr[x_idx + 1]) < distance_from_bound(bs_arr[y_idx - 1]):
                x_idx += 1
            else:
                y_idx -= 1

        if x_idx == y_idx:
            if check_bs_suitable(bs_arr[cur_max_bs_idx][0] + 2):
                estimated_bs = bs_arr[cur_max_bs_idx][0] + 2
            else:
                estimated_bs = bs_arr[cur_max_bs_idx][0]

        if estimated_bs > self._max_bs:
            estimated_bs = self._max_bs

        return estimated_bs


def _get_max_memory_reserved() -> int:
    if is_xpu_available():
        return torch.xpu.max_memory_reserved(device=None)
    return torch.cuda.max_memory_reserved(device=None)


def _get_total_memory_size() -> int:
    if is_xpu_available():
        return torch.xpu.get_device_properties(0).total_memory
    _, total_mem = torch.cuda.mem_get_info()
    return total_mem


class BatchSizeFinder(Callback):
    """The ``BatchSizeFinder`` callback tries to find the largest batch size for a given model that does not give an
    out of memory (OOM) error. All you need to do is add it as a callback inside Trainer and call
    ``trainer.{fit,validate,test,predict}``. Internally it calls the respective step function ``steps_per_trial`` times
    for each batch size until one of the batch sizes generates an OOM error.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        mode: search strategy to update the batch size:

            - ``'power'``: Keep multiplying the batch size by 2, until we get an OOM error.
            - ``'binsearch'``: Initially keep multiplying by 2 and after encountering an OOM error
              do a binary search between the last successful batch size and the batch size that failed.

        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs,
            however in practice a few are needed.

        init_val: initial batch size to start the search with.

        max_trials: max number of increases in batch size done before
            algorithm is terminated

        batch_arg_name: name of the attribute that stores the batch size.
            It is expected that the user has provided a model or datamodule that has a hyperparameter
            with that name. We will look for this attribute name in the following places

            - ``model``
            - ``model.hparams``
            - ``trainer.datamodule`` (the datamodule passed to the tune method)

    Example::

        # 1. Customize the BatchSizeFinder callback to run at different epochs. This feature is
        # useful while fine-tuning models since you can't always use the same batch size after
        # unfreezing the backbone.
        from lightning.pytorch.callbacks import BatchSizeFinder


        class FineTuneBatchSizeFinder(BatchSizeFinder):
            def __init__(self, milestones, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.milestones = milestones

            def on_fit_start(self, *args, **kwargs):
                return

            def on_train_epoch_start(self, trainer, pl_module):
                if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                    self.scale_batch_size(trainer, pl_module)


        trainer = Trainer(callbacks=[FineTuneBatchSizeFinder(milestones=(5, 10))])
        trainer.fit(...)

    Example::

        # 2. Run batch size finder for validate/test/predict.
        from lightning.pytorch.callbacks import BatchSizeFinder


        class EvalBatchSizeFinder(BatchSizeFinder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def on_fit_start(self, *args, **kwargs):
                return

            def on_test_start(self, trainer, pl_module):
                self.scale_batch_size(trainer, pl_module)


        trainer = Trainer(callbacks=[EvalBatchSizeFinder()])
        trainer.test(...)

    """

    def __init__(
        self,
        steps_per_trial: int = 3,
    ) -> None:
        self._steps_per_trial = steps_per_trial
        self._early_exit = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str | None = None) -> None:
        if trainer._accelerator_connector.is_distributed:
            raise MisconfigurationException("The Batch size finder is not supported with distributed strategies.")
        # TODO: check if this can be enabled (#4040)
        if not trainer.fit_loop._data_source.is_module():
            raise MisconfigurationException(
                "The Batch size finder cannot be used with dataloaders passed directly to `.fit()`. Please disable"
                " the feature or incorporate the dataloader into your LightningModule or LightningDataModule."
            )

        # TODO: Add support for multiple eval dataloader
        if stage != "fit":
            loop = trainer._active_loop
            assert loop is not None
            loop.setup_data()
            combined_loader = loop._combined_loader
            assert combined_loader is not None
            if len(combined_loader.flattened) > 1:
                stage = trainer.state.stage
                assert stage is not None
                raise MisconfigurationException(
                    f"The Batch size finder cannot be used with multiple {stage.dataloader_prefix} dataloaders."
                )


    def scale_batch_size(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        params = _scale_batch_dump_params(trainer)
        _scale_batch_reset_params(trainer, 3)
        _try_loop_run(trainer, params)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking or trainer.state.fn != "validate":
            return

        self.scale_batch_size(trainer, pl_module)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)


def _scale_batch_dump_params(trainer: Trainer) -> dict[str, Any]:
    dumped_params = {
        "loggers": trainer.loggers,
        "callbacks": trainer.callbacks,
    }
    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        dumped_params["max_steps"] = trainer.max_steps
        dumped_params["limit_train_batches"] = trainer.limit_train_batches
        dumped_params["limit_val_batches"] = trainer.limit_val_batches
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        dumped_params["limit_eval_batches"] = getattr(trainer, f"limit_{stage.dataloader_prefix}_batches")
        dumped_params["loop_verbose"] = loop.verbose

    dumped_params["loop_state_dict"] = deepcopy(loop.state_dict())
    return dumped_params


def _try_loop_run(trainer: Trainer, params: dict[str, Any]) -> None:
    loop = trainer._active_loop
    assert loop is not None
    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    loop.run()


def _scale_batch_reset_params(trainer: Trainer, steps_per_trial: int) -> None:
    from lightning.pytorch.loggers.logger import DummyLogger

    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.callbacks = []

    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        trainer.limit_train_batches = 1.0
        trainer.limit_val_batches = steps_per_trial
        trainer.fit_loop.epoch_loop.max_steps = steps_per_trial
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        setattr(trainer, f"limit_{stage.dataloader_prefix}_batches", steps_per_trial)
        loop.verbose = False
