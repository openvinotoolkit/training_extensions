import re

import pytest

from otx.algorithms.common.adapters.mmcv.utils import config_utils
from otx.algorithms.common.adapters.mmcv.utils.config_utils import (
    get_adaptive_num_workers,
    InputSizeManager,
    get_configured_input_size,
)
from otx.algorithms.common.configs.configuration_enums import InputSizePreset

from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_adaptive_num_workers(mocker):
    num_gpu = 5
    mock_torch = mocker.patch.object(config_utils, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mock_multiprocessing = mocker.patch.object(config_utils, "multiprocessing")
    mock_multiprocessing.cpu_count.return_value = num_cpu

    assert get_adaptive_num_workers() == num_cpu // num_gpu


@e2e_pytest_unit
def test_get_adaptive_num_workers_no_gpu(mocker):
    num_gpu = 0
    mock_torch = mocker.patch.object(config_utils, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mock_multiprocessing = mocker.patch.object(config_utils, "multiprocessing")
    mock_multiprocessing.cpu_count.return_value = num_cpu

    assert get_adaptive_num_workers() is None


@pytest.fixture
def mock_data_pipeline():
    image_size = (400, 400)

    return [
        dict(type="Mosaic", img_scale=image_size),
        dict(
            type="RandomAffine",
            border=image_size,
        ),
        dict(type="Resize", img_scale=image_size, keep_ratio=True),
        dict(type="Pad", pad_to_square=True),
        dict(
            type="MultiScaleFlipAug",
            img_scale=image_size,
            flip=False,
            transforms=[
                dict(type="Resize", keep_ratio=True),
                dict(type="RandomFlip"),
                dict(type="Pad", size=image_size),
            ],
        ),
        dict(type="RandomResizedCrop", size=image_size[0], efficientnet_style=True),
        dict(
            type="AutoAugment",
            policies=[
                [
                    dict(
                        type="Resize",
                        img_scale=[
                            image_size,
                        ],
                        multiscale_mode="value",
                        keep_ratio=True,
                    )
                ],
                [
                    dict(
                        type="Resize",
                        img_scale=[
                            image_size,
                        ],
                        multiscale_mode="value",
                        keep_ratio=True,
                    ),
                    dict(type="RandomCrop", crop_type="absolute_range", crop_size=image_size),
                    dict(
                        type="Resize",
                        img_scale=[
                            image_size,
                        ],
                        multiscale_mode="value",
                        override=True,
                        keep_ratio=True,
                    ),
                ],
            ],
        ),
        dict(
            type="TwoCropTransform",
            view0=[
                dict(type="RandomResizedCrop", size=image_size),
            ],
            view1=[
                dict(type="RandomResizedCrop", size=image_size),
            ],
            pipeline=[
                dict(type="Resize", size=image_size),
            ],
        ),
    ]


mock_data_pipeline_to_estimate = {
    "pad_smaller_than_resize": {
        "pipeline": [
            dict(type="Resize", img_scale=(300, 300), keep_ratio=True),
            dict(type="Pad", size=(200, 200)),
        ],
        "input_size": (300, 300),
    },
    "crop_bigger_than_resize": {
        "pipeline": [
            dict(type="Resize", img_scale=(300, 300), keep_ratio=True),
            dict(type="crop", size=(400, 400)),
        ],
        "input_size": (300, 300),
    },
    "multi_scale_flip_aug": {
        "pipeline": [
            dict(
                type="MultiScaleFlipAug",
                img_scale=(232, 232),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Pad", size=(500, 500)),
                ],
            ),
        ],
        "input_size": (500, 500),
    },
    "multi_scale_flip_aug_pad_bigger_than_resize": {
        "pipeline": [
            dict(
                type="MultiScaleFlipAug",
                img_scale=(232, 232),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Pad", size=(200, 200)),
                ],
            ),
        ],
        "input_size": (232, 232),
    },
    "resize_crop_pad": {
        "pipeline": [
            dict(type="Resize", img_scale=(300, 300), keep_ratio=True),
            dict(type="crop", size=(200, 200)),
            dict(type="Pad", size=(400, 400)),
        ],
        "input_size": (400, 400),
    },
    "auto_augment": {
        "pipeline": [
            dict(
                type="AutoAugment",
                policies=[
                    [
                        dict(
                            type="Resize",
                            img_scale=[
                                (500, 500),
                                (900, 900),
                            ],
                            multiscale_mode="value",
                            keep_ratio=True,
                        )
                    ],
                    [
                        dict(
                            type="Resize",
                            img_scale=[(600, 600), (400, 400)],
                            multiscale_mode="value",
                            keep_ratio=True,
                        ),
                        dict(type="RandomCrop", crop_type="absolute_range", crop_size=(200, 300)),
                        dict(
                            type="Resize",
                            img_scale=[
                                (100, 200),
                            ],
                            multiscale_mode="value",
                            override=True,
                            keep_ratio=True,
                        ),
                    ],
                ],
            ),
        ],
        "input_size": (500, 500),
    },
    "two_crop_transform": {
        "pipeline": [
            dict(type="Resize", img_scale=(300, 300), keep_ratio=True),
            dict(type="crop", size=(200, 200)),
            dict(type="Pad", size=(400, 400)),
            dict(
                type="TwoCropTransform",
                view0=[
                    dict(type="RandomResizedCrop", size=(600, 600)),
                ],
                view1=[
                    dict(type="RandomResizedCrop", size=(500, 500)),
                ],
            ),
        ],
        "input_size": (600, 600),
    },
    "load_resize_data_from_otxdataset": {
        "pipeline": [
            dict(
                type="LoadResizeDataFromOTXDataset",
                resize_cfg=dict(type="Resize", size=(100, 100)),
            ),
        ],
        "input_size": (100, 100),
    },
    "resize_to": {
        "pipeline": [
            dict(
                type="LoadResizeDataFromOTXDataset",
                resize_cfg=dict(type="Resize", size=(100, 100), downscale_only=True),
            ),
            dict(type="ResizeTo", size=(100, 100)),
        ],
        "input_size": (100, 100),
    },
}


@e2e_pytest_unit
class TestInputSizeManager:
    @pytest.mark.parametrize("base_input_size", [None, 100, [100, 200], {"train": 100}])
    def test_init(self, base_input_size):
        # prepare
        mock_data_config = {"train": {"pipeline": []}}

        # check
        InputSizeManager(mock_data_config, base_input_size)

    def test_init_insufficient_base_input_size(self):
        # prepare
        mock_data_config = {"train": {"pipeline": []}}
        base_input_size = {"val": 100}

        # check if data pipeline has train but base_input_size doesn't have it, error is raised
        with pytest.raises(ValueError):
            InputSizeManager(mock_data_config, base_input_size)

    @pytest.mark.parametrize("input_size", [200, (200, 100)])
    def test_set_input_size(self, mock_data_pipeline, input_size):
        # prepare
        base_input_size = 400
        if isinstance(input_size, tuple):
            expected_input_size_tuple = input_size
            expected_input_size_int = input_size[0]
        elif isinstance(input_size, int):
            expected_input_size_tuple = (input_size, input_size)
            expected_input_size_int = input_size

        mock_data_config = {"train": {"pipeline": mock_data_pipeline}}

        # execute
        InputSizeManager(mock_data_config, base_input_size).set_input_size(input_size)

        # check all input sizes are updated as expected
        def check_val_changed(pipelines):
            if isinstance(pipelines, list):
                for pipeline in pipelines:
                    check_val_changed(pipeline)
            elif isinstance(pipelines, dict):
                for value in pipelines.values():
                    check_val_changed(value)
            elif isinstance(pipelines, tuple):
                assert pipelines == expected_input_size_tuple
            elif not isinstance(pipelines, bool) and isinstance(pipelines, int):
                assert pipelines == expected_input_size_int

        check_val_changed(mock_data_pipeline)

    @pytest.mark.parametrize("base_input_size", [100, [100, 200], {"train": 100}])
    def test_base_input_size_with_given_args(self, base_input_size):
        # prepare
        mock_data_config = {"train": {"pipeline": []}}
        if isinstance(base_input_size, int):
            base_input_size = [base_input_size, base_input_size]
        elif isinstance(base_input_size, dict):
            for task in base_input_size.keys():
                if isinstance(base_input_size[task], int):
                    base_input_size[task] = [base_input_size[task], base_input_size[task]]

        # execute
        input_size_manager = InputSizeManager(mock_data_config, base_input_size)

        # check base_input_size attribute is same as argument given when class initialization
        assert input_size_manager.base_input_size == base_input_size

    def test_base_input_size_without_given_args(self, mocker):
        # prepare
        input_size_manager = InputSizeManager(mocker.MagicMock())
        estimated_input_size = [100, 100]

        # execute
        input_size_manager.get_input_size_from_cfg = mocker.MagicMock(return_value=estimated_input_size)

        # check if base_input_size argument isn't given, input size is estimated
        assert input_size_manager.base_input_size == estimated_input_size

    @pytest.mark.parametrize("test_case", list(mock_data_pipeline_to_estimate.keys()))
    def test_get_input_size_from_cfg(self, test_case):
        # prepare
        pipeline = mock_data_pipeline_to_estimate[test_case]["pipeline"]
        input_size = mock_data_pipeline_to_estimate[test_case]["input_size"]
        mock_data_config = {"train": {"pipeline": pipeline}}
        input_size_manager = InputSizeManager(mock_data_config)

        # check input size is estimated as expected
        assert input_size_manager.get_input_size_from_cfg("train") == input_size


def get_mock_model_ckpt(case):
    if case == "none":
        return None
    if case == "no_input_size":
        return {"config": {}}
    if case == "input_size_default":
        return {"config": {"learning_parameters": {"input_size": {"value": "Default"}}}}
    if case == "input_size_exist":
        return {"config": {"learning_parameters": {"input_size": {"value": "512x512"}}}}


@e2e_pytest_unit
@pytest.mark.parametrize("input_size_config", [InputSizePreset.DEFAULT, InputSizePreset._1024x1024])
@pytest.mark.parametrize("model_ckpt_case", ["none", "no_input_size", "input_size_default", "input_size_exist"])
def test_get_configured_input_size(mocker, input_size_config, model_ckpt_case):
    # prepare
    mock_torch = mocker.patch.object(config_utils, "torch")
    mock_torch.load.return_value = get_mock_model_ckpt(model_ckpt_case)
    input_size_parser = re.compile("(\d+)x(\d+)")

    if input_size_config == InputSizePreset.DEFAULT:
        if model_ckpt_case == "none" or model_ckpt_case == "no_input_size" or model_ckpt_case == "input_size_default":
            expeted_value = None
        elif model_ckpt_case == "input_size_exist":
            input_size = get_mock_model_ckpt(model_ckpt_case)["config"]["learning_parameters"]["input_size"]["value"]
            pattern = input_size_parser.search(input_size)
            expeted_value = (int(pattern.group(1)), int(pattern.group(2)))
    else:
        pattern = input_size_parser.search(input_size_config.value)
        expeted_value = (int(pattern.group(1)), int(pattern.group(2)))

    # check expected value is returned
    assert (
        get_configured_input_size(input_size_config, None if model_ckpt_case == "none" else mocker.MagicMock())
        == expeted_value
    )
