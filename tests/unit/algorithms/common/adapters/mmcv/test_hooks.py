from typing import List
from unittest.mock import MagicMock

import pytest
from mmcv.runner import BaseRunner

from otx.algorithms.common.adapters.mmcv.hooks import TwoCropTransformHook
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@pytest.fixture
def mock_iter_runner(mocker):
    _mock_iter_runner = mocker.patch("mmcv.runner.IterBasedRunner", autospec=True)
    _mock_iter_runner.data_loader = MagicMock()

    return _mock_iter_runner


@pytest.mark.usefixtures("mock_iter_runner")
class TestTwoCropTransformHook:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.two_crop_transform_hook = TwoCropTransformHook(interval=1)

    def set_mock_name(self, name: str, is_both: bool = True) -> MagicMock:
        mock_class = MagicMock()
        mock_class.__class__.__name__ = name
        if name == "TwoCropTransform":
            mock_class.is_both = is_both
        return mock_class

    @e2e_pytest_unit
    def test_use_not_implemented_by_epoch(self) -> None:
        with pytest.raises(NotImplementedError):
            TwoCropTransformHook(interval=1, by_epoch=True)

    @e2e_pytest_unit
    def test_get_dataset(self, mock_iter_runner: BaseRunner) -> None:
        """Test _get_dataset."""
        mock_iter_runner.data_loader.dataset = True
        assert not hasattr(mock_iter_runner.data_loader.dataset, "dataset")

        results = self.two_crop_transform_hook._get_dataset(mock_iter_runner)
        assert results

    @e2e_pytest_unit
    def test_get_dataset_repeat_dataset(self, mock_iter_runner: BaseRunner) -> None:
        """Test _get_dataset when dataset includes child dataset (e.g. RepeatDataset)."""
        mock_iter_runner.data_loader.dataset.dataset = True
        assert hasattr(mock_iter_runner.data_loader.dataset, "dataset")

        results = self.two_crop_transform_hook._get_dataset(mock_iter_runner)
        assert results

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "transforms_order",
        [["TwoCropTransform", "A", "B"], ["A", "TwoCropTransform", "B"], ["A", "B", "TwoCropTransform"]],
    )
    def test_find_two_crop_transform(self, transforms_order: List[object]) -> None:
        """Test _find_two_crop_transform."""
        transforms = [self.set_mock_name(name) for name in transforms_order]

        results = self.two_crop_transform_hook._find_two_crop_transform(transforms)
        assert results.__class__.__name__ == "TwoCropTransform"

    @e2e_pytest_unit
    @pytest.mark.parametrize("interval,cnt,expected", [(1, 0, True), (2, 0, False), (2, 1, True)])
    def test_before_train_epoch(self, mock_iter_runner: BaseRunner, interval: int, cnt: int, expected: bool) -> None:
        """Test before_train_epoch."""
        if hasattr(mock_iter_runner.data_loader.dataset, "dataset"):
            del mock_iter_runner.data_loader.dataset.dataset

        setattr(self.two_crop_transform_hook, "interval", interval)
        setattr(self.two_crop_transform_hook, "cnt", cnt)

        transforms_order = ["TwoCropTransform", "A", "B"]
        transforms = [self.set_mock_name(name=name) for name in transforms_order]
        mock_iter_runner.data_loader.dataset.pipeline.transforms = transforms

        self.two_crop_transform_hook.before_train_epoch(mock_iter_runner)

        assert transforms[0].is_both == expected

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "is_both,interval,cnt,expected",
        [
            (False, 1, 0, False),
            (True, 1, 0, True),
            (False, 3, 0, False),
            (True, 3, 0, True),
            (False, 3, 1, True),
            (True, 3, 1, False),
            (True, 2, 0, False),
        ],
    )
    def test_after_train_iter(
        self, mock_iter_runner: BaseRunner, is_both: bool, interval: int, cnt: int, expected: bool
    ) -> None:
        """Test after_train_iter."""
        if hasattr(mock_iter_runner.data_loader.dataset, "dataset"):
            del mock_iter_runner.data_loader.dataset.dataset

        setattr(self.two_crop_transform_hook, "interval", interval)
        setattr(self.two_crop_transform_hook, "cnt", cnt)

        transforms_order = ["TwoCropTransform", "A", "B"]
        transforms = [self.set_mock_name(name=name, is_both=is_both) for name in transforms_order]
        mock_iter_runner.data_loader.dataset.pipeline.transforms = transforms

        self.two_crop_transform_hook.after_train_iter(mock_iter_runner)

        assert transforms[0].is_both == expected
        if is_both and interval - cnt == 2:
            # test cnt initialization
            assert self.two_crop_transform_hook.cnt == 0
