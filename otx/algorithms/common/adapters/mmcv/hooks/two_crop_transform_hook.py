"""Two crop transform hook."""
from typing import List

from mmcv.runner import BaseRunner
from mmcv.runner.hooks import HOOKS, Hook

from otx.algorithms.common.utils.logger import get_logger
from otx.api.utils.argument_checks import check_input_parameters_type

logger = get_logger()


@HOOKS.register_module()
class TwoCropTransformHook(Hook):
    """TwoCropTransformHook with every specific interval.

    This hook decides whether using single pipeline or two pipelines
    implemented in `TwoCropTransform` for the current iteration.

    Args:
        interval (int): If `interval` == 1, both pipelines is used.
            If `interval` > 1, the first pipeline is used and then
            both pipelines are used every `interval`. Defaults to 1.
        by_epoch (bool): (TODO) Use `interval` by epoch. Defaults to False.
    """

    @check_input_parameters_type()
    def __init__(self, interval: int = 1, by_epoch: bool = False):
        assert interval > 0, f"interval (={interval}) must be positive value."
        if by_epoch:
            raise NotImplementedError("by_epoch is not implemented.")

        self.interval = interval
        self.cnt = 0

    @check_input_parameters_type()
    def _get_dataset(self, runner: BaseRunner):
        """Get dataset to handle `is_both`."""
        if hasattr(runner.data_loader.dataset, "dataset"):
            # for RepeatDataset
            dataset = runner.data_loader.dataset.dataset
        else:
            dataset = runner.data_loader.dataset

        return dataset

    # pylint: disable=inconsistent-return-statements
    @check_input_parameters_type()
    def _find_two_crop_transform(self, transforms: List[object]):
        """Find TwoCropTransform among transforms."""
        for transform in transforms:
            if transform.__class__.__name__ == "TwoCropTransform":
                return transform

    @check_input_parameters_type()
    def before_train_epoch(self, runner: BaseRunner):
        """Called before_train_epoch in TwoCropTransformHook."""
        # Always keep `TwoCropTransform` enabled.
        if self.interval == 1:
            return

        dataset = self._get_dataset(runner)
        two_crop_transform = self._find_two_crop_transform(dataset.pipeline.transforms)
        if self.cnt == self.interval - 1:
            # start using both pipelines
            two_crop_transform.is_both = True
        else:
            two_crop_transform.is_both = False

    @check_input_parameters_type()
    def after_train_iter(self, runner: BaseRunner):
        """Called after_train_iter in TwoCropTransformHook."""
        # Always keep `TwoCropTransform` enabled.
        if self.interval == 1:
            return

        if self.cnt < self.interval - 1:
            # Instead of using `runner.every_n_iters` or `runner.every_n_inner_iters`,
            # this condition is used to compare `self.cnt` with `self.interval` throughout the entire epochs.
            self.cnt += 1

        if self.cnt == self.interval - 1:
            dataset = self._get_dataset(runner)
            two_crop_transform = self._find_two_crop_transform(dataset.pipeline.transforms)
            if not two_crop_transform.is_both:
                # If `self.cnt` == `self.interval`-1, there are two cases,
                # 1. `self.cnt` was updated in L709, so `is_both` must be on for the next iter.
                # 2. if the current iter was already conducted, `is_both` must be off.
                two_crop_transform.is_both = True
            else:
                two_crop_transform.is_both = False
                self.cnt = 0
