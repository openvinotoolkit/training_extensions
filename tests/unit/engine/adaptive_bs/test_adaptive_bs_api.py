from __future__ import annotations

from math import sqrt
from typing import Any
from unittest.mock import MagicMock

import pytest
from lightning.pytorch.loggers.logger import DummyLogger
from otx.core.types.task import OTXTaskType
from otx.engine.adaptive_bs import adaptive_bs_api as target_file
from otx.engine.adaptive_bs.adaptive_bs_api import BatchSizeFinder, _adjust_train_args, _train_model, adapt_batch_size


@pytest.fixture()
def mock_is_cuda_available(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "is_cuda_available", return_value=True)


@pytest.fixture()
def mock_is_xpu_available(mocker) -> MagicMock:
    return mocker.patch.object(target_file, "is_xpu_available", return_value=False)


@pytest.fixture()
def default_bs() -> int:
    return 8


@pytest.fixture()
def train_set_size() -> int:
    return 10


@pytest.fixture()
def default_lr() -> float:
    return 0.01


@pytest.fixture()
def mock_engine(default_bs: int, train_set_size: int, default_lr: float) -> MagicMock:
    engine = MagicMock()
    engine.datamodule.train_subset.batch_size = default_bs
    engine.datamodule.train_subset.subset_name = "train"
    engine.datamodule.subsets = {"train": range(train_set_size)}
    engine.device.devices = 1
    engine._cache = {"devices": 1}
    engine.model.optimizer_callable.optimizer_kwargs = {"lr": default_lr}
    return engine


@pytest.fixture()
def mock_bs_search_algo_ins() -> MagicMock:
    bs_search_algo_ins = MagicMock()
    bs_search_algo_ins.auto_decrease_batch_size.return_value = 4
    bs_search_algo_ins.find_big_enough_batch_size.return_value = 16
    return bs_search_algo_ins


@pytest.fixture()
def mock_bs_search_algo_cls(mocker, mock_bs_search_algo_ins) -> MagicMock:
    return mocker.patch.object(target_file, "BsSearchAlgo", return_value=mock_bs_search_algo_ins)


@pytest.fixture()
def train_args() -> dict[str, Any]:
    return {
        "self": MagicMock(),
        "run_hpo": True,
        "adaptive_bs": True,
        "kwargs": {"kwargs_a": "kwargs_a"},
    }


def get_bs(engine) -> int:
    return engine.datamodule.train_subset.batch_size


def get_lr(engine) -> float:
    return engine.model.optimizer_callable.optimizer_kwargs["lr"]


@pytest.mark.parametrize("not_increase", [True, False])
def test_adapt_batch_size(
    not_increase,
    mock_is_cuda_available,
    mock_is_xpu_available,
    mock_engine,
    mock_bs_search_algo_cls,
    mock_bs_search_algo_ins,
    default_bs,
    train_set_size,
    default_lr,
    train_args,
):
    adapt_batch_size(mock_engine, not_increase, **train_args)

    # check patch_optimizer_and_scheduler_for_hpo is invoked
    mock_engine.model.patch_optimizer_and_scheduler_for_hpo.assert_called_once()
    # check BsSearchAlgo is initialized well
    mock_bs_search_algo_cls.assert_called_once()
    assert mock_bs_search_algo_cls.call_args.kwargs["default_bs"] == default_bs
    assert mock_bs_search_algo_cls.call_args.kwargs["max_bs"] == train_set_size
    # check proper method is invkoed depending on value of not_increase
    if not_increase:
        mock_bs_search_algo_ins.auto_decrease_batch_size.assert_called_once()
    else:
        mock_bs_search_algo_ins.find_big_enough_batch_size.assert_called_once()
    # check lr and bs is changed well
    cur_bs = get_bs(mock_engine)
    assert default_bs != cur_bs
    assert get_lr(mock_engine) == pytest.approx(default_lr * sqrt(cur_bs / default_bs))


@pytest.fixture()
def mock_os(mocker) -> MagicMock:
    os = mocker.patch.object(target_file, "os")
    os.environ = {}  # noqa: B003
    return os


@pytest.mark.parametrize("not_increase", [True, False])
def test_adapt_batch_size_dist_main_proc(
    not_increase,
    mock_is_cuda_available,
    mock_is_xpu_available,
    mock_engine,
    mock_bs_search_algo_cls,
    mock_bs_search_algo_ins,
    default_bs,
    train_set_size,
    default_lr,
    train_args,
    mock_os,
):
    num_devices = 2
    mock_engine.device.devices = num_devices
    adapt_batch_size(mock_engine, not_increase, **train_args)

    # same as test_adapt_batch_size
    mock_engine.model.patch_optimizer_and_scheduler_for_hpo.assert_called_once()
    mock_bs_search_algo_cls.assert_called_once()
    assert mock_bs_search_algo_cls.call_args.kwargs["default_bs"] == default_bs
    assert mock_bs_search_algo_cls.call_args.kwargs["max_bs"] == train_set_size // num_devices
    if not_increase:
        mock_bs_search_algo_ins.auto_decrease_batch_size.assert_called_once()
    else:
        mock_bs_search_algo_ins.find_big_enough_batch_size.assert_called_once()
    cur_bs = get_bs(mock_engine)
    assert default_bs != cur_bs
    assert get_lr(mock_engine) == pytest.approx(default_lr * sqrt(cur_bs / default_bs))
    # check ADAPTIVE_BS_FOR_DIST is set for other processors
    assert int(mock_os.environ["ADAPTIVE_BS_FOR_DIST"]) == cur_bs


def test_adapt_batch_size_dist_sub_proc(
    mock_is_cuda_available,
    mock_is_xpu_available,
    mock_engine,
    mock_bs_search_algo_cls,
    default_bs,
    default_lr,
    train_args,
    mock_os,
):
    mock_engine.device.devices = 2
    mock_os.environ["ADAPTIVE_BS_FOR_DIST"] = 4
    adapt_batch_size(mock_engine, **train_args)

    mock_bs_search_algo_cls.assert_not_called()
    cur_bs = get_bs(mock_engine)
    assert default_bs != cur_bs
    assert get_lr(mock_engine) == pytest.approx(default_lr * sqrt(cur_bs / default_bs))
    assert int(mock_os.environ["ADAPTIVE_BS_FOR_DIST"]) == cur_bs


def test_adapt_batch_size_no_accelerator(
    mock_is_cuda_available,
    mock_is_xpu_available,
    mock_engine,
    train_args,
):
    mock_is_cuda_available.return_value = False
    with pytest.raises(RuntimeError, match="Adaptive batch size supports CUDA or XPU."):
        adapt_batch_size(mock_engine, **train_args)


def test_adapt_batch_size_zvp_task(
    mock_is_cuda_available,
    mock_is_xpu_available,
    mock_engine,
    train_args,
):
    mock_engine.task = OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING
    with pytest.raises(RuntimeError, match="Zero shot visual prompting task doesn't support adaptive batch size."):
        adapt_batch_size(mock_engine, **train_args)


def test_adjust_train_args(train_args):
    adjusted_args = _adjust_train_args(train_args)

    assert "self" not in adjusted_args
    assert "run_hpo" not in adjusted_args
    assert "adaptive_bs" not in adjusted_args
    assert adjusted_args["kwargs_a"] == "kwargs_a"


def test_train_model(mock_engine):
    mock_engine.device.devices = 2
    mock_engine._cahce["devices"] = 2
    batch_size = 16
    train_args = {"a": 1, "b": 2}
    _train_model(bs=16, engine=mock_engine, **train_args)

    # check batch size is set
    assert get_bs(mock_engine) == batch_size
    # check turn off distributed training
    assert mock_engine._cache["devices"] == 1
    # check train is invoked
    mock_engine.train.assert_called_once()
    # check BatchSizeFinder callback is registered
    assert isinstance(mock_engine.train.call_args.kwargs["callbacks"][0], BatchSizeFinder)
    # check train_args is passed to train function
    for key, val in train_args.items():
        assert mock_engine.train.call_args.kwargs[key] == val


@pytest.mark.parametrize("bs", [-10, 0])
def test_train_model_wrong_bs(mock_engine, bs):
    with pytest.raises(ValueError, match="Batch size should be greater than 0"):
        _train_model(bs=bs, engine=mock_engine)


class TestBatchSizeFinder:
    def test_init(self):
        BatchSizeFinder()

    @pytest.fixture()
    def mock_active_loop(self):
        return MagicMock()

    @pytest.fixture()
    def mock_trainer(self, mock_active_loop) -> MagicMock:
        trainer = MagicMock()
        trainer.limit_val_batches = 100
        trainer.fit_loop.epoch_loop.max_steps = -1
        trainer.fit_loop.max_epochs = 200
        trainer._active_loop = mock_active_loop
        return trainer

    def test_setup(self, mock_trainer):
        bs_finder = BatchSizeFinder()
        bs_finder.setup(trainer=mock_trainer, pl_module=MagicMock(), stage="fit")

    @pytest.mark.parametrize("stage", ["validate", "test"])
    def test_setup_not_fit(self, stage: str, mock_trainer):
        bs_finder = BatchSizeFinder()
        with pytest.raises(RuntimeError, match="Adaptive batch size supports only training."):
            bs_finder.setup(trainer=mock_trainer, pl_module=MagicMock(), stage=stage)

    def test_on_fit_start(self, mock_trainer, mock_active_loop):
        steps_per_trial = 3
        bs_finder = BatchSizeFinder(steps_per_trial=steps_per_trial)
        bs_finder.on_fit_start(trainer=mock_trainer, pl_module=MagicMock())

        # check steps_per_trial is set well
        assert mock_trainer.limit_val_batches == steps_per_trial
        assert mock_trainer.fit_loop.epoch_loop.max_steps == -1
        assert mock_trainer.fit_loop.max_epochs == 1
        assert mock_trainer.limit_train_batches == steps_per_trial
        # check active_loop is run
        assert mock_active_loop.restarting is False
        mock_active_loop.run.assert_called_once()
        # check callback and logger is removed
        assert mock_trainer.callbacks == []
        assert isinstance(mock_trainer.logger, DummyLogger) or mock_trainer.logger is None

    def test_on_fit_start_no_val(self, mock_trainer, mock_active_loop):
        steps_per_trial = 3
        mock_trainer.limit_val_batches = 0
        bs_finder = BatchSizeFinder(steps_per_trial=steps_per_trial)
        bs_finder.on_fit_start(trainer=mock_trainer, pl_module=MagicMock())

        # check steps_per_trial is set well
        assert mock_trainer.limit_val_batches == 0
        assert mock_trainer.fit_loop.epoch_loop.max_steps == -1
        assert mock_trainer.fit_loop.max_epochs == 1
        assert mock_trainer.limit_train_batches == steps_per_trial
        # check active_loop is run
        assert mock_active_loop.restarting is False
        mock_active_loop.run.assert_called_once()
        # check callback and logger is removed
        assert mock_trainer.callbacks == []
        assert isinstance(mock_trainer.logger, DummyLogger) or mock_trainer.logger is None

    def test_on_fit_start_iter_based_loop(self, mock_trainer, mock_active_loop):
        mock_trainer.fit_loop.epoch_loop.max_steps = 200
        steps_per_trial = 3
        bs_finder = BatchSizeFinder(steps_per_trial=steps_per_trial)
        bs_finder.on_fit_start(trainer=mock_trainer, pl_module=MagicMock())

        # check steps_per_trial is set well
        assert mock_trainer.limit_val_batches == steps_per_trial
        assert mock_trainer.fit_loop.epoch_loop.max_steps == steps_per_trial
        assert mock_trainer.limit_train_batches == 1.0
        # check active_loop is run
        assert mock_active_loop.restarting is False
        mock_active_loop.run.assert_called_once()
        # check callback and logger is removed
        assert mock_trainer.callbacks == []
        assert isinstance(mock_trainer.logger, DummyLogger) or mock_trainer.logger is None

    def test_on_fit_start_no_loop(self, mock_trainer):
        mock_trainer._active_loop = None
        steps_per_trial = 3
        bs_finder = BatchSizeFinder(steps_per_trial=steps_per_trial)

        with pytest.raises(RuntimeError, match="There is no active loop."):
            bs_finder.on_fit_start(trainer=mock_trainer, pl_module=MagicMock())
