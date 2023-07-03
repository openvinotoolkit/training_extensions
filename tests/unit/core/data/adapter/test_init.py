import os

import pytest

from otx.algorithms.common.configs.training_base import TrainType
from otx.api.entities.subset import Subset
from otx.core.data.adapter import get_dataset_adapter
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)

from pathlib import Path
import shutil


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", TASK_NAME_TO_TASK_TYPE.keys())
@pytest.mark.parametrize("train_type", [TrainType.Incremental.value])
def test_get_dataset_adapter_incremental(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]
    if str(task_type).upper() == "VISUAL_PROMPTING":
        data_root = data_root.get("coco")

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
        val_data_roots=os.path.join(root_path, data_root["val"]),
    )

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        test_data_roots=os.path.join(root_path, data_root["test"]),
    )


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", ["classification"])
@pytest.mark.parametrize("train_type", [TrainType.Selfsupervised.value])
def test_get_dataset_adapter_selfsl_classification(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
    )

    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        test_data_roots=os.path.join(root_path, data_root["test"]),
    )


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", ["segmentation"])
@pytest.mark.parametrize("train_type", [TrainType.Selfsupervised.value])
def test_get_dataset_adapter_selfsl_segmentation(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]

    with pytest.raises(ValueError, match=r"pseudo_mask_dir must be set."):
        get_dataset_adapter(
            task_type=task_type,
            train_type=train_type,
            train_data_roots=os.path.join(root_path, data_root["train"]),
        )

        get_dataset_adapter(
            task_type=task_type,
            train_type=train_type,
            test_data_roots=os.path.join(root_path, data_root["test"]),
        )

    tmp_supcon_mask_dir = Path("/tmp/selfsl_supcon_unit_test")
    get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
        pseudo_mask_dir=tmp_supcon_mask_dir,
    )
    shutil.rmtree(str(tmp_supcon_mask_dir))


# TODO: direct annotation function is only supported in COCO format for now.
@e2e_pytest_unit
@pytest.mark.parametrize("task_name", ["detection"])
@pytest.mark.parametrize("train_type", [TrainType.Incremental.value])
def test_direct_annotation(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]

    t_adapter = get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
        train_ann_files="tests/assets/car_tree_bug/annotations/instances_train_5_imgs.json",
        val_data_roots=os.path.join(root_path, data_root["val"]),
    )
    assert t_adapter.dataset[Subset.TRAINING].get_subset("train_5_imgs").get_annotated_items() == 5

    v_adapter = get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
        val_data_roots=os.path.join(root_path, data_root["val"]),
        val_ann_files="tests/assets/car_tree_bug/annotations/instances_val_1_imgs.json",
    )
    assert v_adapter.dataset[Subset.VALIDATION].get_subset("val_1_imgs").get_annotated_items() == 1

    tv_adapter = get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
        train_ann_files="tests/assets/car_tree_bug/annotations/instances_train_5_imgs.json",
        val_data_roots=os.path.join(root_path, data_root["val"]),
        val_ann_files="tests/assets/car_tree_bug/annotations/instances_val_1_imgs.json",
    )
    assert tv_adapter.dataset[Subset.TRAINING].get_subset("train_5_imgs").get_annotated_items() == 5
    assert tv_adapter.dataset[Subset.VALIDATION].get_subset("val_1_imgs").get_annotated_items() == 1


@e2e_pytest_unit
@pytest.mark.parametrize("task_name", ["classification", "detection", "segmentation"])
@pytest.mark.parametrize("train_type", [TrainType.Incremental.value])
def test_unlabeled_file_list(task_name, train_type):
    root_path = os.getcwd()
    task_type = TASK_NAME_TO_TASK_TYPE[task_name]
    data_root = TASK_NAME_TO_DATA_ROOT[task_name]

    unlabeled_data_roots = "tests/assets/unlabeled_dataset"
    unlabeled_file_list = "tests/assets/unlabeled_dataset/unlabeled_file_list.txt"

    adapter = get_dataset_adapter(
        task_type=task_type,
        train_type=train_type,
        train_data_roots=os.path.join(root_path, data_root["train"]),
        val_data_roots=os.path.join(root_path, data_root["val"]),
        unlabeled_data_roots=unlabeled_data_roots,
        unlabeled_file_list=unlabeled_file_list,
    )

    assert len(adapter.dataset[Subset.UNLABELED]) == 8
