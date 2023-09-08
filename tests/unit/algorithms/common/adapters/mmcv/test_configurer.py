import pytest
from mmcv.utils import Config
from otx.algorithms.common.adapters.mmcv import configurer
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
class TestBaseConfigurer:
    def test_get_input_size_to_fit_dataset(self, mocker):
        data_cfg = Config({"data": {"train": {"otx_dataset": None}}})
        input_size = configurer.BaseConfigurer.get_input_size_to_fit_dataset(data_cfg)
        assert input_size is None

        data_cfg = Config({"data": {"train": {"otx_dataset": True}}})
        mock_stat = mocker.patch.object(configurer, "compute_robust_dataset_statistics")

        mock_stat.return_value = {}
        input_size = configurer.BaseConfigurer.get_input_size_to_fit_dataset(data_cfg)
        assert input_size is None

        mock_stat.return_value = dict(
            image=dict(
                robust_max=150,
            ),
        )
        input_size = configurer.BaseConfigurer.get_input_size_to_fit_dataset(data_cfg)
        assert input_size == (128, 128)

        mock_stat.return_value = dict(
            image=dict(
                robust_max=150,
            ),
            annotation=dict()
        )
        input_size = configurer.BaseConfigurer.get_input_size_to_fit_dataset(data_cfg, use_annotations=True)
        assert input_size == (128, 128)

        mock_stat.return_value = dict(
            image=dict(
                robust_max=150,
            ),
            annotation=dict(
                size_of_shape=dict(
                    robust_min=64
                )
            )
        )
        input_size = configurer.BaseConfigurer.get_input_size_to_fit_dataset(data_cfg, use_annotations=True)
        assert input_size == (64, 64)
