import pytest
from math import sqrt

from otx.algorithms.common.adapters.mmcv.utils import automatic_bs
from otx.algorithms.common.adapters.mmcv.utils import adapt_batch_size
from otx.algorithms.common.adapters.mmcv.utils.automatic_bs import SubDataset

DEFAULT_BS = 8
DEFAULT_LR = 0.001
TRAINSET_SIZE = 100


class MockBsSearchAlgo:
    def __init__(self, train_func, default_bs: int, max_bs: int):
        self.train_func = train_func
        self.default_bs = default_bs
        self.max_bs = max_bs

    def auto_decrease_batch_size(self):
        self.train_func(self.default_bs)
        self.train_func(self.default_bs // 2)
        return self.default_bs // 2

    def find_big_enough_batch_size(self, drop_last: bool):
        self.train_func(self.default_bs)
        self.train_func(self.default_bs + 2)
        return self.default_bs + 2


@pytest.fixture
def mock_adapt_algo_cls(mocker):
    return mocker.patch.object(automatic_bs, "BsSearchAlgo", side_effect=MockBsSearchAlgo)


@pytest.fixture
def common_cfg(mocker):
    mock_cfg = mocker.MagicMock()
    mock_cfg.runner = {"type": "EpochRunnerWithCancel", "max_epochs": 100}
    mock_cfg.custom_hooks = [
        {"type": "AdaptiveTrainSchedulingHook", "enable_eval_before_run": True},
        {"type": "OTXProgressHook"},
    ]
    mock_cfg.optimizer.lr = DEFAULT_LR
    return mock_cfg


def set_mock_cfg_not_action(common_cfg):
    common_cfg.data.train_dataloader = {"samples_per_gpu": DEFAULT_BS}
    return common_cfg


def set_mock_cfg_action(common_cfg):
    common_cfg.data.videos_per_gpu = DEFAULT_BS
    common_cfg.domain = "ACTION_CLASSIFICATION"
    return common_cfg


@pytest.fixture
def mock_dataset(mocker):
    mock_ds = [mocker.MagicMock()]
    mock_ds[0].__len__.return_value = TRAINSET_SIZE
    return mock_ds


@pytest.mark.parametrize("not_increase", [True, False])
@pytest.mark.parametrize("is_action_task", [True, False])
@pytest.mark.parametrize("is_iter_based_runner", [True, False])
def test_adapt_batch_size(
    mocker, mock_adapt_algo_cls, common_cfg, mock_dataset, not_increase, is_action_task, is_iter_based_runner
):
    # prepare
    mock_train_func = mocker.MagicMock()
    new_bs = DEFAULT_BS // 2 if not_increase else DEFAULT_BS + 2

    max_eph_name = "max_epochs"
    if is_iter_based_runner:
        common_cfg.runner = {"type": "IterBasedRunnerWithCancel", "max_iters": 100}
        max_eph_name = "max_iters"

    if is_action_task:
        mock_config = set_mock_cfg_action(common_cfg)
    else:
        mock_config = set_mock_cfg_not_action(common_cfg)

    # execute
    adapt_batch_size(mock_train_func, mock_config, mock_dataset, False, not_increase)

    # check adapted batch size is applied
    if is_action_task:
        assert mock_config.data.videos_per_gpu == new_bs
    else:
        assert mock_config.data.train_dataloader["samples_per_gpu"] == new_bs
    # check leanring rate is updated depending on adapted batch size
    bs_change_ratio = new_bs / DEFAULT_BS
    assert mock_config.optimizer.lr == pytest.approx(DEFAULT_LR * sqrt(bs_change_ratio))
    # check adapt function gets proper arguments
    assert mock_adapt_algo_cls.call_args.kwargs["default_bs"] == DEFAULT_BS
    assert mock_adapt_algo_cls.call_args.kwargs["max_bs"] == TRAINSET_SIZE
    # check length of dataset is decreased to reduce time
    assert len(mock_train_func.call_args_list[0].kwargs["dataset"][0]) == DEFAULT_BS
    assert len(mock_train_func.call_args_list[1].kwargs["dataset"][0]) == new_bs
    # check max epoch is set as 1 to reduce time
    assert mock_train_func.call_args_list[0].kwargs["cfg"].runner[max_eph_name] == 1
    assert mock_train_func.call_args_list[1].kwargs["cfg"].runner[max_eph_name] == 1
    # check eval before run is disabled to reduce time
    assert not mock_train_func.call_args_list[0].kwargs["cfg"].custom_hooks[0]["enable_eval_before_run"]
    assert not mock_train_func.call_args_list[1].kwargs["cfg"].custom_hooks[0]["enable_eval_before_run"]
    # check OTXProgressHook is removed
    assert len(mock_train_func.call_args_list[0].kwargs["cfg"].custom_hooks) == 1


class TestSubDataset:
    @pytest.fixture(autouse=True)
    def set_up(self, mocker):
        self.num_samples = 3
        self.fullset = mocker.MagicMock()
        self.sub_dataset = SubDataset(self.fullset, self.num_samples)

    def test_init(self, mocker):
        fullset = mocker.MagicMock()
        subset = SubDataset(fullset, 3)

        # test for class incremental case. If below assert can't be passed, ClsIncrSampler can't work well.
        assert len(subset.img_indices["new"]) / len(subset.img_indices["old"]) + 1 <= self.num_samples

    @pytest.mark.parametrize("num_samples", [-1, 0])
    def test_init_w_wrong_num_samples(self, mocker, num_samples):
        fullset = mocker.MagicMock()
        with pytest.raises(ValueError):
            SubDataset(fullset, num_samples)

    def test_len(self):
        assert len(self.sub_dataset) == self.num_samples

    def test_getitem(self):
        self.sub_dataset[0]
        self.fullset.__getitem__.assert_called_once_with(0)

    def test_getattr(self):
        self.fullset.data = "data"
        assert self.sub_dataset.data == "data"

    def test_flag(self):
        assert len(self.sub_dataset.flag) == self.num_samples
