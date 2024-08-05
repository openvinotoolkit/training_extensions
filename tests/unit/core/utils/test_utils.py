# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import pytest
from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem
from otx.core.utils import utils as target_file
from otx.core.utils.utils import (
    get_adaptive_num_workers,
    get_idx_list_per_classes,
    import_object_from_module,
    is_ckpt_for_finetuning,
    is_ckpt_from_otx_v1,
    remove_state_dict_prefix,
)


@pytest.mark.parametrize("num_dataloader", [1, 2, 4])
def test_get_adaptive_num_workers(mocker, num_dataloader):
    num_gpu = 5
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mocker.patch.object(target_file, "cpu_count", return_value=num_cpu)

    assert get_adaptive_num_workers(num_dataloader) == num_cpu // (num_gpu * num_dataloader)


def test_get_adaptive_num_workers_no_gpu(mocker):
    num_gpu = 0
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mocker.patch.object(target_file, "cpu_count", return_value=num_cpu)

    assert get_adaptive_num_workers() is None


def test_is_ckpt_from_otx_v1():
    ckpt = {"model": "some_model", "VERSION": 1}
    assert is_ckpt_from_otx_v1(ckpt)

    ckpt = {"model": "another_model", "VERSION": 2}
    assert not is_ckpt_from_otx_v1(ckpt)


def test_is_ckpt_for_finetuning():
    ckpt = {"state_dict": {"param1": 1, "param2": 2}}
    assert is_ckpt_for_finetuning(ckpt)

    ckpt = {"other_key": "value"}
    assert not is_ckpt_for_finetuning(ckpt)

    ckpt = {}
    assert not is_ckpt_for_finetuning(ckpt)


@pytest.fixture()
def fxt_dm_dataset() -> DmDataset:
    dataset_items = [
        DatasetItem(
            id=f"item00{i}_0",
            subset="train",
            media=None,
            annotations=[
                Label(label=0),
            ],
        )
        for i in range(1, 101)
    ] + [
        DatasetItem(
            id=f"item00{i}_1",
            subset="train",
            media=None,
            annotations=[
                Label(label=1),
            ],
        )
        for i in range(1, 9)
    ]

    return DmDataset.from_iterable(dataset_items, categories=["0", "1"])


def test_get_idx_list_per_classes(fxt_dm_dataset):
    # Call the function under test
    result = get_idx_list_per_classes(fxt_dm_dataset)

    # Assert the expected output
    expected_result = defaultdict(list)
    expected_result[0] = list(range(100))
    expected_result[1] = list(range(100, 108))
    assert result == expected_result

    # Call the function under test with use_string_label
    result = get_idx_list_per_classes(fxt_dm_dataset, use_string_label=True)

    # Assert the expected output
    expected_result = defaultdict(list)
    expected_result["0"] = list(range(100))
    expected_result["1"] = list(range(100, 108))
    assert result == expected_result


def test_import_object_from_module():
    obj_path = "otx.core.utils.utils.get_idx_list_per_classes"
    obj = import_object_from_module(obj_path)
    assert obj == get_idx_list_per_classes


def test_remove_state_dict_prefix():
    state_dict = {
        "model._orig_mod.backbone.0.weight": 1,
        "model._orig_mod.backbone.0.bias": 2,
        "model._orig_mod.backbone.1.weight": 3,
        "model._orig_mod.backbone.1.bias": 4,
    }
    new_state_dict = remove_state_dict_prefix(state_dict=state_dict, prefix="_orig_mod.")
    expected = {
        "model.backbone.0.weight": 1,
        "model.backbone.0.bias": 2,
        "model.backbone.1.weight": 3,
        "model.backbone.1.bias": 4,
    }
    assert new_state_dict == expected
