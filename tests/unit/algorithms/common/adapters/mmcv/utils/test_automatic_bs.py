import pytest

from otx.algorithms.common.adapters.mmcv.utils import automatic_bs
from otx.algorithms.common.adapters.mmcv.utils import adapt_batch_size
from otx.algorithms.common.adapters.mmcv.utils.automatic_bs import SubDataset

DEFAULT_BS = 8
DEFAULT_LR = 0.001
TRAINSET_SIZE = 100


def bs_adapt_func(train_func, current_bs, trainset_size):
    train_func(current_bs)
    train_func(current_bs // 2)
    return current_bs // 2


@pytest.fixture
def mock_adapt_func(mocker):
    mock_func = mocker.patch.object(automatic_bs, "adapt_torch_model_bs")
    mock_func.side_effect = bs_adapt_func
    return mock_func


@pytest.fixture
def common_cfg(mocker):
    mock_cfg = mocker.MagicMock()
    mock_cfg.runner = {"type": "EpochRunnerWithCancel", "max_epochs": 100}
    mock_cfg.custom_hooks = [{"type": "AdaptiveTrainSchedulingHook", "enable_eval_before_run": True}]
    mock_cfg.optimizer.lr = DEFAULT_LR
    return mock_cfg


@pytest.fixture
def mock_cfg_not_action(common_cfg):
    common_cfg.data.train_dataloader = {"samples_per_gpu": DEFAULT_BS}
    return common_cfg


@pytest.fixture
def mock_cfg_action(common_cfg):
    common_cfg.data.videos_per_gpu = DEFAULT_BS
    common_cfg.domain = "ACTION_CLASSIFICATION"
    return common_cfg


@pytest.fixture
def mock_dataset(mocker):
    mock_ds = [mocker.MagicMock()]
    mock_ds[0].__len__.return_value = TRAINSET_SIZE
    return mock_ds


def test_adapt_batch_size_not_action_task(mocker, mock_adapt_func, mock_cfg_not_action, mock_dataset):
    # prepare
    mock_train_func = mocker.MagicMock()

    # execute
    adapt_batch_size(mock_train_func, mock_cfg_not_action, mock_dataset, False)

    # check adapted batch size is applied
    assert mock_cfg_not_action.data.train_dataloader["samples_per_gpu"] == DEFAULT_BS // 2
    # check leanring rate is updated depending on adapted batch size
    assert mock_cfg_not_action.optimizer.lr == pytest.approx(DEFAULT_LR / 2)
    # check adapt function gets proper arguments
    assert mock_adapt_func.call_args.kwargs["current_bs"] == DEFAULT_BS
    assert mock_adapt_func.call_args.kwargs["trainset_size"] == TRAINSET_SIZE
    # check length of dataset is decreased to reduce time
    assert len(mock_train_func.call_args_list[0].kwargs["dataset"][0]) == DEFAULT_BS
    assert len(mock_train_func.call_args_list[1].kwargs["dataset"][0]) == DEFAULT_BS // 2
    # check max epoch is set as 1 to reduce time
    assert mock_train_func.call_args_list[0].kwargs["cfg"].runner["max_epochs"] == 1
    assert mock_train_func.call_args_list[1].kwargs["cfg"].runner["max_epochs"] == 1
    # check eval before run is disabled to reduce time
    assert not mock_train_func.call_args_list[0].kwargs["cfg"].custom_hooks[0]["enable_eval_before_run"]
    assert not mock_train_func.call_args_list[1].kwargs["cfg"].custom_hooks[0]["enable_eval_before_run"]


def test_adapt_batch_size_action_task(mocker, mock_adapt_func, mock_cfg_action, mock_dataset):
    # prepare
    mock_train_func = mocker.MagicMock()

    # execute
    adapt_batch_size(mock_train_func, mock_cfg_action, mock_dataset, True)

    # check adapted batch size is applied
    assert mock_cfg_action.data.videos_per_gpu == DEFAULT_BS // 2
    # check leanring rate is updated depending on adapted batch size
    assert mock_cfg_action.optimizer.lr == pytest.approx(DEFAULT_LR / 2)
    # check adapt function gets proper arguments
    assert mock_adapt_func.call_args.kwargs["current_bs"] == DEFAULT_BS
    assert mock_adapt_func.call_args.kwargs["trainset_size"] == TRAINSET_SIZE
    # check length of dataset is decreased to reduce time
    assert len(mock_train_func.call_args_list[0].kwargs["dataset"][0]) == DEFAULT_BS
    assert len(mock_train_func.call_args_list[1].kwargs["dataset"][0]) == DEFAULT_BS // 2
    # check max epoch is set as 1 to reduce time
    assert mock_train_func.call_args_list[0].kwargs["cfg"].runner["max_epochs"] == 1
    assert mock_train_func.call_args_list[1].kwargs["cfg"].runner["max_epochs"] == 1
    # check eval before run is enabled if validate is set as True
    assert mock_train_func.call_args_list[0].kwargs["cfg"].custom_hooks[0]["enable_eval_before_run"]
    assert mock_train_func.call_args_list[1].kwargs["cfg"].custom_hooks[0]["enable_eval_before_run"]


class TestSubDataset:
    @pytest.fixture(autouse=True)
    def set_up(self, mocker):
        self.num_samples = 3
        self.fullset = mocker.MagicMock()
        self.sub_dataset = SubDataset(self.fullset, self.num_samples)

    def test_init(self, mocker):
        fullset = mocker.MagicMock()
        SubDataset(fullset, 3)

    @pytest.mark.parametrize("num_sampels", [-1, 0])
    def test_init_w_wrong_num_sampels(self, mocker, num_sampels):
        fullset = mocker.MagicMock()
        with pytest.raises(ValueError):
            SubDataset(fullset, num_sampels)

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
