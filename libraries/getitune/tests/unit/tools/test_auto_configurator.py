# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from getitune.backend.lightning.models.base import DataInputParams, LightningModel
from getitune.data.module import DataModule
from getitune.tools.auto_configurator import (
    DEFAULT_CONFIG_PER_TASK,
    AutoConfigurator,
)
from getitune.types.device import DeviceType
from getitune.types.label import LabelInfo, SegLabelInfo
from getitune.types.task import TaskType
from getitune.utils.utils import should_pass_label_info


@pytest.fixture
def fxt_data_root_per_task_type() -> dict:
    return {
        TaskType.MULTI_CLASS_CLS: "tests/assets/classification_cifar10",
        TaskType.MULTI_LABEL_CLS: "tests/assets/multilabel_classification_coco",
        TaskType.DETECTION: "tests/assets/detection_coco",
        TaskType.KEYPOINT_DETECTION: "tests/assets/keypoint_detection_coco",
        TaskType.INSTANCE_SEGMENTATION: "tests/assets/instance_segmentation_coco",
        TaskType.SEMANTIC_SEGMENTATION: "tests/assets/segmentation_pets",
    }


class TestAutoConfigurator:
    def test_check_task(self) -> None:
        # None inputs
        with pytest.raises(ValueError, match="Either task or model must be provided."):
            auto_configurator = AutoConfigurator(task=None, model=None)

        # data_root is None & task is not None
        auto_configurator = AutoConfigurator(data_root=None, task="MULTI_CLASS_CLS")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # instantiate with model_config_path
        model_config_path = "src/getitune/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml"
        auto_configurator = AutoConfigurator(data_root=None, task=None, model=model_config_path)
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # instantiate with model_config_path
        with pytest.raises(
            ValueError,
            match="If model is provided as a name, task must be provided to find the model.",
        ):
            auto_configurator = AutoConfigurator(data_root=None, task=None, model="mobilenet_v3_large")

        auto_configurator = AutoConfigurator(data_root=None, task="MULTI_CLASS_CLS", model="mobilenet_v3_large")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

        # data_root is not None & task is None
        data_root = "tests/assets/classification_cifar10"
        auto_configurator = AutoConfigurator(data_root=data_root, task="MULTI_CLASS_CLS")
        assert auto_configurator.task == "MULTI_CLASS_CLS"

    def test_load_default_config(self) -> None:
        # Test the load_default_config function
        data_root = "tests/assets/classification_cifar10"
        task = TaskType.MULTI_CLASS_CLS
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        # Default Config
        default_config = auto_configurator._load_default_config()
        target_config = DEFAULT_CONFIG_PER_TASK[task].resolve()
        assert isinstance(default_config, dict)
        assert len(default_config) > 0
        assert "config" in default_config
        assert len(default_config["config"]) > 0
        assert str(default_config["config"][0]) == str(target_config)

        # getitune-Mobilenet-v2
        # new_config
        model_name = "vit_tiny"
        new_config = auto_configurator._load_default_config(
            config_path="src/getitune/recipe/classification/multi_class_cls/vit_tiny.yaml",
        )
        new_path = str(target_config).split("/")
        new_path[-1] = f"{model_name}.yaml"
        new_target_config = Path("/".join(new_path))
        assert isinstance(new_config, dict)
        assert len(new_config) > 0
        assert "config" in new_config
        assert len(new_config["config"]) > 0
        assert Path(new_config["config"][0]).name == new_target_config.name
        assert Path(new_config["config"][0]).exists()

    def test_get_datamodule(self) -> None:
        data_root = None
        task = TaskType.DETECTION
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        # data_root is None
        with pytest.raises(ValueError, match="No data root provided."):
            assert auto_configurator.get_datamodule() is None

        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task=task)

        datamodule = auto_configurator.get_datamodule()
        assert isinstance(datamodule, DataModule)
        assert datamodule.task == task

    def test_get_model(self, fxt_task: TaskType, fxt_data_root_per_task_type) -> None:
        auto_configurator = AutoConfigurator(task=fxt_task, data_root=fxt_data_root_per_task_type[fxt_task])

        # With label_info
        label_names = ["class1", "class2", "class3"]
        label_info = (
            LabelInfo(label_names=label_names, label_groups=[label_names], label_ids=label_names)
            if fxt_task != TaskType.SEMANTIC_SEGMENTATION
            else SegLabelInfo(label_names=label_names, label_groups=[label_names], label_ids=label_names)
        )
        model = auto_configurator.get_model(
            label_info=label_info,
            data_input_params=DataInputParams((288, 288), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert isinstance(model, LightningModel)

        model_cls = model.__class__

        if should_pass_label_info(model_cls):
            with pytest.raises(ValueError, match="Given model class (.*) requires a valid label_info to instantiate."):
                _ = auto_configurator.get_model(label_info=None)

    def test_get_model_set_input_size(self) -> None:
        auto_configurator = AutoConfigurator(task=TaskType.MULTI_CLASS_CLS)
        label_names = ["class1", "class2", "class3"]
        label_info = LabelInfo(label_names=label_names, label_groups=[label_names], label_ids=label_names)

        model = auto_configurator.get_model(
            label_info=label_info,
            data_input_params=DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        assert model.data_input_params.input_size == (300, 300)

    def test_update_ov_subset_pipeline(self) -> None:
        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task="DETECTION")

        datamodule = auto_configurator.get_datamodule()
        # The detection base config has augmentations_cpu with Resize
        assert any("Resize" in aug.get("class_path", "") for aug in datamodule.test_subset.augmentations_cpu)

        ov_config_path = DEFAULT_CONFIG_PER_TASK[TaskType.DETECTION].parent / "openvino_model.yaml"
        ov_recipe_batch_size = auto_configurator._load_default_config(config_path=ov_config_path)["data"][
            "test_subset"
        ]["batch_size"]

        updated_datamodule = auto_configurator.update_ov_subset_pipeline(datamodule, subset="test")
        # OV recipes now use Resize (preprocessing moved from ModelAPI to getitune)
        assert len(updated_datamodule.test_subset.augmentations_cpu) == 1
        assert "Resize" in updated_datamodule.test_subset.augmentations_cpu[0]["class_path"]
        assert not updated_datamodule.tile_config.enable_tiler
        # Without tiling, the OV recipe's batch_size is used as-is (no override).
        assert updated_datamodule.test_subset.batch_size == ov_recipe_batch_size

    def test_update_ov_subset_pipeline_tiling_keeps_tiler_and_strips_resize(self) -> None:
        """With tiling enabled, the OV pipeline keeps the tiler on and removes the tile Resize.

        ModelAPI's tiler resizes native crops to the model input internally, so the
        DataModule must not resize tiles (that would both duplicate the resize and
        corrupt the tile-to-image coordinate mapping used by OVModel.forward_tiles).
        """
        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task="DETECTION")

        datamodule = auto_configurator.get_datamodule()
        datamodule.tile_config.enable_tiler = True

        # Sanity-check the precondition this regression test relies on: the OV recipe's
        # non-tiled batch_size must be > 1, otherwise the batch_size==1 assertion below
        # would pass trivially even if the override logic were removed.
        ov_config_path = DEFAULT_CONFIG_PER_TASK[TaskType.DETECTION].parent / "openvino_model.yaml"
        ov_recipe_batch_size = auto_configurator._load_default_config(config_path=ov_config_path)["data"][
            "test_subset"
        ]["batch_size"]
        assert ov_recipe_batch_size > 1, (
            "This test expects the OV recipe's test_subset.batch_size to be > 1 "
            "(currently 64) so it can verify that tiling forces it down to 1."
        )

        updated_datamodule = auto_configurator.update_ov_subset_pipeline(datamodule, subset="test")

        # Tiling forces the loader batch_size down to 1 regardless of the OV recipe's
        # configured batch_size: with tiling enabled, a single dataset item expands into
        # every grid tile of the native-resolution image, so a larger loader batch_size
        # would collate many images' worth of tiles into one in-memory batch, which can
        # hang or OOM (see DataModule._eval_loader_kwargs and update_ov_subset_pipeline).
        assert updated_datamodule.test_subset.batch_size == 1

        # Tiling stays enabled: ModelAPI performs tiled inference via forward_tiles.
        assert updated_datamodule.tile_config.enable_tiler
        # The model-input Resize is stripped so native-resolution crops reach ModelAPI.
        assert all(
            "Resize" not in aug.get("class_path", "") for aug in updated_datamodule.test_subset.augmentations_cpu
        )

    def test_update_ov_subset_pipeline_from_pre_constructed_datasets(self) -> None:
        """Test that update_ov_subset_pipeline works when the datamodule was created via from_vision_datasets (no data_root)."""
        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task=TaskType.DETECTION)

        # Create a normal datamodule first, then rebuild it via from_vision_datasets
        # to simulate what the quantization pipeline does
        datamodule = auto_configurator.get_datamodule()
        pre_constructed_datamodule = DataModule.from_vision_datasets(
            train_dataset=datamodule.subsets["train"],
            val_dataset=datamodule.subsets["val"],
            test_dataset=datamodule.subsets.get("test"),
            train_subset=datamodule.train_subset,
            val_subset=datamodule.val_subset,
            test_subset=datamodule.test_subset,
        )
        assert pre_constructed_datamodule.data_root == ""

        # This should NOT raise ValueError about dataset format detection
        updated_datamodule = auto_configurator.update_ov_subset_pipeline(pre_constructed_datamodule, subset="train")
        assert len(updated_datamodule.train_subset.augmentations_cpu) == 1
        assert "Resize" in updated_datamodule.train_subset.augmentations_cpu[0]["class_path"]
        assert not updated_datamodule.tile_config.enable_tiler
        # Verify subsets are preserved
        assert "train" in updated_datamodule.subsets
        assert "val" in updated_datamodule.subsets

    @pytest.mark.parametrize("training_device", [DeviceType.xpu, DeviceType.gpu, DeviceType.auto])
    def test_update_ov_subset_pipeline_resets_device_to_cpu(self, training_device: DeviceType) -> None:
        """The rebuilt datamodule must run on CPU, whatever device training used.

        OpenVINO evaluation always runs on CPU, so the rebuilt datamodule must not
        inherit the training accelerator. If it did, ``DataLoader(pin_memory=True)``
        would page-lock every batch through the accelerator runtime (Level Zero on
        XPU, CUDA on NVIDIA) for a workload that never transfers anything to the
        device - pure overhead, and it drags the GPU stack into a CPU-only step.

        This covers the ``data_root``-backed reconstruction path.
        """
        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task=TaskType.DETECTION)

        datamodule = auto_configurator.get_datamodule()
        datamodule.device = training_device

        # Precondition: the source datamodule *would* pin. Without this the assertions
        # below could pass trivially even if the device reset were removed.
        assert datamodule._pin_memory is True

        updated_datamodule = auto_configurator.update_ov_subset_pipeline(datamodule, subset="test")

        assert updated_datamodule.device == DeviceType.cpu
        assert updated_datamodule._pin_memory is False
        # The consequence that actually matters: the loader handed to OV eval does not pin.
        assert updated_datamodule.test_dataloader().pin_memory is False

    @pytest.mark.parametrize("training_device", [DeviceType.xpu, DeviceType.gpu, DeviceType.auto])
    def test_update_ov_subset_pipeline_resets_device_to_cpu_pre_constructed(
        self,
        training_device: DeviceType,
    ) -> None:
        """Same CPU/pinned-memory guarantee for the pre-constructed-dataset path.

        This branch returns via ``DataModule.from_vision_datasets`` instead of the
        ``DataModule`` constructor, so it needs its own coverage.
        """
        data_root = "tests/assets/detection_coco"
        auto_configurator = AutoConfigurator(data_root=data_root, task=TaskType.DETECTION)

        datamodule = auto_configurator.get_datamodule()
        pre_constructed_datamodule = DataModule.from_vision_datasets(
            train_dataset=datamodule.subsets["train"],
            val_dataset=datamodule.subsets["val"],
            test_dataset=datamodule.subsets.get("test"),
            train_subset=datamodule.train_subset,
            val_subset=datamodule.val_subset,
            test_subset=datamodule.test_subset,
            device=training_device,
        )
        # Confirm the pre-constructed branch is the one under test.
        assert pre_constructed_datamodule.data_root == ""
        assert pre_constructed_datamodule._pin_memory is True

        updated_datamodule = auto_configurator.update_ov_subset_pipeline(pre_constructed_datamodule, subset="test")

        assert updated_datamodule.device == DeviceType.cpu
        assert updated_datamodule._pin_memory is False
        assert updated_datamodule.test_dataloader().pin_memory is False
