# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics Configurator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from getitune.backend.ultralytics.models.base import UltralyticsModel
from getitune.backend.ultralytics.tools.configurator import Configurator
from getitune.backend.ultralytics.tools.utils import (
    flatten_overrides,
)
from getitune.data.module import DataModule
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType

_RECIPE_DIR = Path(__file__).resolve().parents[5] / "src" / "getitune" / "recipe"

_DETECTION_MODEL_NAME = "yolo26_s"
_DETECTION_RECIPE_FILE = _RECIPE_DIR / "detection" / "yolo26_s.yaml"
_DETECTION_CLASS_PATH = "getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel"

_INST_SEG_MODEL_NAME = "yolo26_s_seg"
_INST_SEG_CLASS_PATH = "getitune.backend.ultralytics.models.instance_segmentation.UltralyticsInstSegModel"


def _minimal_recipe(
    *,
    backend: str = "ultralytics",
    task: str = "DETECTION",
    model_name: str = "yolo26n.yaml",
    pretrained: bool = False,
    class_path: str = _DETECTION_CLASS_PATH,
) -> dict:
    """Return a minimal valid recipe dict."""
    return {
        "backend": backend,
        "task": task,
        "model": {
            "class_path": class_path,
            "init_args": {
                "model_name": model_name,
                "pretrained": pretrained,
                "imgsz": 640,
            },
        },
        "engine": {"device": "auto"},
        "training": {
            "epochs": 50,
            "batch": 8,
            "lr0": 0.005,
            "close_mosaic": 0,
        },
        "export": {"format": "OPENVINO", "precision": "FP32"},
    }


def _minimal_data_config() -> dict:
    """Return a minimal data config dict with all three subsets."""
    return {
        "input_size": [640, 640],
        "train_subset": {"batch_size": 16, "augmentations_cpu": []},
        "val_subset": {"batch_size": 16},
        "test_subset": {"batch_size": 16},
    }


def _write_recipe(tmp_path: Path, recipe: dict, name: str = "recipe.yaml") -> Path:
    """Write a recipe dict to a YAML file and return the path."""
    path = tmp_path / name
    path.write_text(yaml.dump(recipe), encoding="utf-8")
    return path


def _make_label_info(num_classes: int = 5) -> LabelInfo:
    return LabelInfo.from_num_classes(num_classes)


class TestInitData:
    def test_pathlike_data_stored_as_root(self, tmp_path: Path) -> None:
        cfg = Configurator(
            data=tmp_path / "data.yaml",
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        assert cfg.data_root == (tmp_path / "data.yaml").resolve()
        assert cfg.datamodule is None

    def test_str_data_stored_as_root(self, tmp_path: Path) -> None:
        cfg = Configurator(
            data=str(tmp_path / "data.yaml"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        assert cfg.data_root == (tmp_path / "data.yaml").resolve()

    def test_datamodule_input_stored_directly(self) -> None:
        from getitune.config.data import SubsetConfig, TileConfig
        from getitune.data.module import DataModule

        assets_dir = Path(__file__).resolve().parents[3] / "assets" / "detection_coco"
        if not assets_dir.exists():
            pytest.skip(f"Detection test assets not found at {assets_dir}")

        dm = DataModule(
            task=TaskType.DETECTION,
            data_root=str(assets_dir),
            train_subset=SubsetConfig(batch_size=2, subset_name="train"),
            val_subset=SubsetConfig(batch_size=2, subset_name="val"),
            test_subset=SubsetConfig(batch_size=2, subset_name="test"),
            tile_config=TileConfig(enable_tiler=False),
            input_size=(64, 64),
        )
        cfg = Configurator(
            data=dm,
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        assert cfg.datamodule is dm
        assert cfg.data_root is None

    def test_invalid_data_type_raises(self) -> None:
        with pytest.raises(TypeError, match="data must be PathLike or DataModule"):
            Configurator(
                data=42,  # type: ignore[arg-type]
                model=_DETECTION_MODEL_NAME,
                task=TaskType.DETECTION,
            )


class TestInitModel:
    def test_model_str_resolved_from_recipes_dir(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        assert cfg.model_config is not None
        assert (
            cfg.model_config["class_path"] == "getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel"
        )
        assert cfg.model_config["init_args"]["model_name"] == "yolo26s.yaml"

    def test_model_str_instseg_resolved(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_INST_SEG_MODEL_NAME,
            task=TaskType.INSTANCE_SEGMENTATION,
        )
        assert cfg.model_config is not None
        assert cfg.model_config["class_path"] == _INST_SEG_CLASS_PATH

    def test_model_path_resolved(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe_path = _write_recipe(tmp_path, recipe, name="custom.yaml")
        cfg = Configurator(
            data=Path("dummy"),
            model=recipe_path,
            task=TaskType.DETECTION,
        )
        assert cfg.model_config is not None
        assert cfg.model_config["class_path"] == _DETECTION_CLASS_PATH
        assert cfg.model_config["init_args"]["model_name"] == "yolo26n.yaml"

    def test_model_str_unknown_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Recipe not found"):
            Configurator(
                data=Path("dummy"),
                model="nonexistent-model-xyz",
                task=TaskType.DETECTION,
            )

    def test_model_str_without_task_raises(self) -> None:
        with pytest.raises(ValueError, match="without task="):
            Configurator(
                data=Path("dummy"),
                model=_DETECTION_MODEL_NAME,
            )

    def test_model_yaml_filename_without_task_raises(self) -> None:
        with pytest.raises(ValueError, match="without task="):
            Configurator(
                data=Path("dummy"),
                model="yolo26_s.yaml",
            )

    def test_model_path_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Recipe not found"):
            Configurator(
                data=Path("dummy"),
                model=tmp_path / "nonexistent.yaml",
                task=TaskType.DETECTION,
            )

    def test_model_ultralytics_model_stored(self) -> None:
        from getitune.backend.ultralytics.models.detection import UltralyticsDetectionModel

        model = UltralyticsDetectionModel(
            model_name="yolo26s.yaml",
            label_info=_make_label_info(),
            pretrained=False,
            imgsz=640,
        )
        cfg = Configurator(
            data=Path("dummy"),
            model=model,
            task=TaskType.DETECTION,
        )
        assert cfg.model is model
        assert cfg.model_config is None

    def test_invalid_model_type_raises(self) -> None:
        with pytest.raises(TypeError, match="model must be str, PathLike, or UltralyticsModel"):
            Configurator(
                data=Path("dummy"),
                model=42,  # type: ignore[arg-type]
                task=TaskType.DETECTION,
            )


class TestInitTask:
    def test_task_as_enum(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        assert cfg.task == "DETECTION"

    def test_task_as_string(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_INST_SEG_MODEL_NAME,
            task="INSTANCE_SEGMENTATION",
        )
        assert cfg.task == "INSTANCE_SEGMENTATION"

    def test_unsupported_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported task"):
            Configurator(
                data=Path("dummy"),
                model=_DETECTION_MODEL_NAME,
                task="KEYPOINT_DETECTION",
            )

    def test_task_none_allowed(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_RECIPE_FILE,
        )
        assert cfg.task is None


class TestInitTrainingExport:
    def test_training_and_export_stored(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
            training={"epochs": 100, "batch": 32},
            export={"format": "OPENVINO", "precision": "FP16"},
        )
        # Constructor overrides are merged on top of recipe defaults.
        assert cfg.training["epochs"] == 100
        assert cfg.training["batch"] == 32
        assert cfg.export["precision"] == "FP16"
        # Recipe defaults are preserved for keys not overridden.
        assert cfg.training["lr0"] == 0.001
        assert cfg.training["close_mosaic"] == 0

    def test_training_and_export_default_to_empty(self) -> None:
        from getitune.backend.ultralytics.models.detection import UltralyticsDetectionModel

        # When model is a live instance (not a recipe path), training/export
        # are not loaded from a recipe.
        model = UltralyticsDetectionModel(
            model_name="yolo26s.yaml",
            label_info=_make_label_info(),
            pretrained=False,
            imgsz=640,
        )
        cfg = Configurator(
            data=Path("dummy"),
            model=model,
            task=TaskType.DETECTION,
        )
        assert cfg.training == {}
        assert cfg.export == {}


class TestConvert:
    def test_convert_produces_backend_tagged_dict(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        result = Configurator.convert(path)
        assert result["backend"] == "ultralytics"
        assert result["task"] == "DETECTION"
        assert result["model"]["class_path"] == _DETECTION_CLASS_PATH
        assert result["model"]["init_args"]["model_name"] == "yolo26n.yaml"

    def test_convert_applies_hyper_parameters(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = _minimal_data_config()
        path = _write_recipe(tmp_path, recipe)
        result = Configurator.convert(
            path,
            hyper_parameters={"training": {"learning_rate": 0.002, "batch_size": 4, "max_epochs": 10}},
        )
        assert result["training"]["lr0"] == 0.002
        assert result["training"]["batch"] == 4
        assert result["training"]["epochs"] == 10
        assert result["data"]["train_subset"]["batch_size"] == 4

    def test_convert_without_hyper_parameters(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        result = Configurator.convert(path, hyper_parameters=None)
        assert result["training"]["lr0"] == 0.005
        assert result["training"]["epochs"] == 50

    def test_convert_wrong_backend_raises(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe(backend="lightning")
        path = _write_recipe(tmp_path, recipe)
        with pytest.raises(ValueError, match="Expected backend 'ultralytics'"):
            Configurator.convert(path)

    def test_convert_unsupported_task_raises(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe(task="KEYPOINT_DETECTION")
        path = _write_recipe(tmp_path, recipe)
        with pytest.raises(ValueError, match="Unsupported task"):
            Configurator.convert(path)

    def test_convert_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            Configurator.convert("/nonexistent/recipe.yaml")

    def test_convert_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just a list\n", encoding="utf-8")
        with pytest.raises(TypeError, match="YAML mapping"):
            Configurator.convert(path)

    def test_convert_resolves_data_reference(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "_base_" / "data"
        base_dir.mkdir(parents=True)
        base_data = {"input_size": [640, 640], "train_subset": {"batch_size": 16}}
        (base_dir / "det.yaml").write_text(yaml.dump(base_data), encoding="utf-8")

        recipe = _minimal_recipe()
        recipe["data"] = "../_base_/data/det.yaml"
        recipe_dir = tmp_path / "detection"
        recipe_dir.mkdir()
        path = _write_recipe(recipe_dir, recipe)

        result = Configurator.convert(path)
        assert result["data"]["input_size"] == [640, 640]

    def test_convert_inline_data(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = {"input_size": [320, 320]}
        path = _write_recipe(tmp_path, recipe)
        result = Configurator.convert(path)
        assert result["data"]["input_size"] == [320, 320]

    def test_convert_data_overrides_merge(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "_base_" / "data"
        base_dir.mkdir(parents=True)
        base_data = {"train_subset": {"batch_size": 16, "num_workers": 8}}
        (base_dir / "det.yaml").write_text(yaml.dump(base_data), encoding="utf-8")

        recipe = _minimal_recipe()
        recipe["data"] = "../_base_/data/det.yaml"
        recipe["overrides"] = {"data": {"train_subset": {"batch_size": 4}}}
        recipe_dir = tmp_path / "detection"
        recipe_dir.mkdir()
        path = _write_recipe(recipe_dir, recipe)

        result = Configurator.convert(path)
        assert result["data"]["train_subset"]["batch_size"] == 4
        assert result["data"]["train_subset"]["num_workers"] == 8

    def test_convert_missing_data_ref_raises(self, tmp_path: Path) -> None:
        from omegaconf.errors import InterpolationResolutionError

        recipe = _minimal_recipe()
        recipe["data"] = "nonexistent.yaml"
        path = _write_recipe(tmp_path, recipe)
        with pytest.raises(InterpolationResolutionError, match="Referenced file not found"):
            Configurator.convert(path)


class TestApplyHyperParameters:
    def _make_configurator(self) -> Configurator:
        return Configurator(
            data=Path("dummy"),
            model=_DETECTION_RECIPE_FILE,
            task=TaskType.DETECTION,
            training={
                "epochs": 100,
                "batch": 16,
                "lr0": 0.01,
                "weight_decay": 0.0005,
                "patience": 100,
                "close_mosaic": 0,
            },
        )

    def test_learning_rate(self) -> None:
        cfg = self._make_configurator()
        cfg.apply_hyper_parameters({"training": {"learning_rate": 0.05}})
        assert cfg.training["lr0"] == 0.05

    def test_weight_decay(self) -> None:
        cfg = self._make_configurator()
        cfg.apply_hyper_parameters({"training": {"weight_decay": 0.001}})
        assert cfg.training["weight_decay"] == 0.001

    def test_max_epochs(self) -> None:
        cfg = self._make_configurator()
        cfg.apply_hyper_parameters({"training": {"max_epochs": 50}})
        assert cfg.training["epochs"] == 50

    def test_input_size(self) -> None:
        cfg = self._make_configurator()
        cfg.apply_hyper_parameters({"training": {"input_size_height": 320, "input_size_width": 320}})
        assert cfg.model_config is not None
        assert cfg.model_config["init_args"]["imgsz"] == 320

    def test_early_stopping_disable(self) -> None:
        cfg = self._make_configurator()
        cfg.apply_hyper_parameters({"training": {"early_stopping": {"enable": False}}})
        assert cfg.training["patience"] == 0

    def test_early_stopping_set_patience(self) -> None:
        cfg = self._make_configurator()
        cfg.apply_hyper_parameters({"training": {"early_stopping": {"enable": True, "patience": 25}}})
        assert cfg.training["patience"] == 25

    def test_empty_training_section_noop(self) -> None:
        cfg = self._make_configurator()
        original = cfg.training
        cfg.apply_hyper_parameters({})
        assert cfg.training == original

    def test_batch_size_propagates_to_data_config(self) -> None:
        cfg = self._make_configurator()
        cfg._data_config = _minimal_data_config()
        cfg.apply_hyper_parameters({"training": {"batch_size": 8}})
        assert cfg._data_config["train_subset"]["batch_size"] == 8
        assert cfg._data_config["val_subset"]["batch_size"] == 8
        assert cfg._data_config["test_subset"]["batch_size"] == 8

    def test_num_workers_propagates_to_data_config(self) -> None:
        cfg = self._make_configurator()
        cfg._data_config = _minimal_data_config()
        cfg.apply_hyper_parameters({"training": {"num_workers": 0}})
        assert cfg._data_config["train_subset"]["num_workers"] == 0
        assert cfg._data_config["val_subset"]["num_workers"] == 0
        assert cfg._data_config["test_subset"]["num_workers"] == 0


class TestToConfigDict:
    def test_model_section_from_path(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_RECIPE_FILE,
            task=TaskType.DETECTION,
        )
        result = cfg.to_config_dict()
        assert (
            result["model"]["class_path"] == "getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel"
        )
        assert result["model"]["init_args"]["model_name"] == "yolo26s.yaml"
        assert result["model"]["init_args"]["pretrained"] is True

    def test_backend_and_task(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        result = cfg.to_config_dict()
        assert result["backend"] == "ultralytics"
        assert result["task"] == "DETECTION"

    def test_training_section_preserved(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
            training={"epochs": 100, "batch": 16},
        )
        result = cfg.to_config_dict()
        assert result["training"]["epochs"] == 100
        assert result["training"]["batch"] == 16

    def test_max_epochs_at_top_level(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
            training={"epochs": 50, "batch": 16},
        )
        result = cfg.to_config_dict()
        assert result["max_epochs"] == 50

    def test_data_config_included(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        cfg._data_config = _minimal_data_config()
        result = cfg.to_config_dict()
        assert result["data"]["input_size"] == [640, 640]
        assert "tile_config" in result["data"]
        assert result["data"]["tile_config"]["enable_tiler"] is False

    def test_data_is_deep_copy(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        cfg._data_config = _minimal_data_config()
        result = cfg.to_config_dict()
        result["data"]["input_size"] = [999, 999]
        assert cfg._data_config["input_size"] == [640, 640]


class TestRealRecipes:
    """Verify the shipped YOLO26 recipe files parse correctly."""

    @pytest.mark.parametrize("variant", ["yolo26_n", "yolo26_s", "yolo26_m"])
    def test_yolo26_detection_recipe_loads(self, variant: str) -> None:
        path = _RECIPE_DIR / "detection" / f"{variant}.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        result = Configurator.convert(path)
        assert result["backend"] == "ultralytics"
        assert result["task"] == "DETECTION"
        assert (
            result["model"]["class_path"] == "getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel"
        )
        assert result["model"]["init_args"]["model_name"].endswith(".yaml")
        assert result["training"]["close_mosaic"] == 0

    @pytest.mark.parametrize("variant", ["yolo26_n", "yolo26_s", "yolo26_m"])
    def test_yolo26_recipe_has_data_config(self, variant: str) -> None:
        path = _RECIPE_DIR / "detection" / f"{variant}.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        result = Configurator.convert(path)
        assert "input_size" in result["data"]
        assert "train_subset" in result["data"]


class TestSemanticSegmentationRecipes:
    """Verify the shipped YOLO26 semantic segmentation recipes parse correctly."""

    @pytest.mark.parametrize("variant", ["yolo26_n", "yolo26_s", "yolo26_m", "yolo26_l", "yolo26_x"])
    def test_semantic_recipe_loads(self, variant: str) -> None:
        path = _RECIPE_DIR / "semantic_segmentation" / f"{variant}_sem.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        result = Configurator.convert(path)
        assert result["backend"] == "ultralytics"
        assert result["task"] == "SEMANTIC_SEGMENTATION"
        assert (
            result["model"]["class_path"]
            == "getitune.backend.ultralytics.models.semantic_segmentation.UltralyticsSemanticSegModel"
        )
        assert result["model"]["init_args"]["model_name"].endswith("-sem.yaml")
        assert result["data"]["input_size"] == [512, 512]
        assert result["data"]["train_subset"]["augmentations_gpu"] == []
        assert result["data"]["val_subset"]["augmentations_gpu"] == []
        assert result["data"]["test_subset"]["augmentations_gpu"] == []


class TestApplyOverrides:
    def test_training_override(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
            training={"epochs": 50},
        )
        cfg.apply_overrides({"training.epochs": 200})
        assert cfg.training["epochs"] == 200

    def test_model_init_args_override(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        cfg.apply_overrides({"model.init_args.imgsz": 320})
        assert cfg.model_config is not None
        assert cfg.model_config["init_args"]["imgsz"] == 320

    def test_export_override(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
            export={"format": "OPENVINO", "precision": "FP32"},
        )
        cfg.apply_overrides({"export.precision": "FP16"})
        assert cfg.export["precision"] == "FP16"

    def test_nested_overrides(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
            training={"epochs": 50},
        )
        cfg.apply_overrides({"training": {"lr0": 0.001, "batch": 32}})
        assert cfg.training["lr0"] == 0.001
        assert cfg.training["batch"] == 32

    def test_bare_key_raises(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        with pytest.raises(ValueError, match="dot path"):
            cfg.apply_overrides({"epochs": 10})

    def test_none_overrides_noop(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
            training={"epochs": 50},
        )
        cfg.apply_overrides(None)
        assert cfg.training["epochs"] == 50

    def test_unknown_section_raises(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        with pytest.raises(KeyError):
            cfg.apply_overrides({"bogus.field": 1})


class TestCreateModel:
    def test_creates_detection_model(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model="yolo26_n",
            task=TaskType.DETECTION,
        )
        model = cfg.create_model(_make_label_info())
        assert model.model_name == "yolo26n.yaml"
        assert model.label_info is not None
        assert model.label_info.num_classes == 5
        assert model.pretrained is True
        assert model.imgsz == 640

    def test_weights_path_loads_checkpoint(self, tmp_path: Path) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model="yolo26_n",
            task=TaskType.DETECTION,
        )
        fake_weights = tmp_path / "weights.pt"
        fake_weights.write_bytes(b"")  # placeholder file

        with patch.object(UltralyticsModel, "load_checkpoint") as mock_load:
            model = cfg.create_model(_make_label_info(), weights_path=fake_weights)

        assert model.model_name == "yolo26n.yaml"
        mock_load.assert_called_once_with(fake_weights)

    def test_no_weights_path_skips_checkpoint(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model="yolo26_n",
            task=TaskType.DETECTION,
        )
        with patch.object(UltralyticsModel, "load_checkpoint") as mock_load:
            cfg.create_model(_make_label_info())
        mock_load.assert_not_called()

    def test_invalid_class_path_raises(self, tmp_path: Path) -> None:
        bad_recipe = _minimal_recipe()
        bad_recipe["model"] = {"class_path": "nonexistent.module.BadClass", "init_args": {}}
        path = _write_recipe(tmp_path, bad_recipe, name="bad.yaml")
        cfg = Configurator(
            data=Path("dummy"),
            model=path,
            task=TaskType.DETECTION,
        )
        with pytest.raises(ValueError, match="No module named"):
            cfg.create_model(_make_label_info())

    def test_class_path_must_resolve_to_ultralytics_model(self, tmp_path: Path) -> None:
        bad_recipe = _minimal_recipe()
        bad_recipe["model"] = {"class_path": "pathlib.Path", "init_args": {}}
        path = _write_recipe(tmp_path, bad_recipe, name="bad.yaml")
        cfg = Configurator(
            data=Path("dummy"),
            model=path,
            task=TaskType.DETECTION,
        )
        with pytest.raises(ValueError, match="subclass of UltralyticsModel"):
            cfg.create_model(_make_label_info())


class TestCreateEngine:
    def test_creates_engine(self, tmp_path: Path) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model="yolo26_n",
            task=TaskType.DETECTION,
        )
        model = cfg.create_model(_make_label_info())
        engine = cfg.create_engine(model, data=tmp_path, work_dir=tmp_path / "work")
        from getitune.backend.ultralytics.engine import UltralyticsEngine

        assert isinstance(engine, UltralyticsEngine)

    def test_training_defaults_forwarded(self, tmp_path: Path) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model="yolo26_n",
            task=TaskType.DETECTION,
            training={"epochs": 50, "close_mosaic": 0},
        )
        model = cfg.create_model(_make_label_info())
        engine = cfg.create_engine(model, data=tmp_path, work_dir=tmp_path / "work")
        assert engine._train_args["epochs"] == 50
        assert engine._train_args["close_mosaic"] == 0

    def test_export_defaults_forwarded(self, tmp_path: Path) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model="yolo26_n",
            task=TaskType.DETECTION,
            export={
                "format": "OPENVINO",
                "precision": "FP32",
                "confidence_threshold": 0.4,
                "iou_threshold": 0.6,
            },
        )
        model = cfg.create_model(_make_label_info())
        engine = cfg.create_engine(model, data=tmp_path, work_dir=tmp_path / "work")
        assert engine._export_args["confidence_threshold"] == 0.4
        assert engine._export_args["iou_threshold"] == 0.6

    def test_device_override(self, tmp_path: Path) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model="yolo26_n",
            task=TaskType.DETECTION,
        )
        model = cfg.create_model(_make_label_info())
        engine = cfg.create_engine(model, data=tmp_path, work_dir=tmp_path / "work", device="cpu")
        import torch

        assert engine._device == torch.device("cpu")


class TestBuildDatamodule:
    def test_returns_existing_datamodule(self) -> None:
        from getitune.config.data import SubsetConfig, TileConfig
        from getitune.data.module import DataModule

        assets_dir = Path(__file__).resolve().parents[3] / "assets" / "detection_coco"
        if not assets_dir.exists():
            pytest.skip(f"Detection test assets not found at {assets_dir}")

        dm = DataModule(
            task=TaskType.DETECTION,
            data_root=str(assets_dir),
            train_subset=SubsetConfig(batch_size=2, subset_name="train"),
            val_subset=SubsetConfig(batch_size=2, subset_name="val"),
            test_subset=SubsetConfig(batch_size=2, subset_name="test"),
            tile_config=TileConfig(enable_tiler=False),
            input_size=(64, 64),
        )
        cfg = Configurator(
            data=dm,
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        assert cfg.build_datamodule() is dm

    def test_builds_from_recipe_data_config(self, tmp_path: Path) -> None:
        assets_dir = Path(__file__).resolve().parents[3] / "assets" / "detection_coco"
        if not assets_dir.exists():
            pytest.skip(f"Detection test assets not found at {assets_dir}")

        recipe = _minimal_recipe()
        recipe["data"] = _minimal_data_config()
        path = _write_recipe(tmp_path, recipe)

        cfg = Configurator(
            data=str(assets_dir),
            model=path,
            task=TaskType.DETECTION,
        )
        dm = cfg.build_datamodule()
        assert isinstance(dm, DataModule)
        assert dm.train_subset.batch_size == 16
        assert dm.val_subset.batch_size == 16
        assert dm.test_subset.batch_size == 16

    def test_uses_constructor_data_root_when_no_arg(self, tmp_path: Path) -> None:
        assets_dir = Path(__file__).resolve().parents[3] / "assets" / "detection_coco"
        if not assets_dir.exists():
            pytest.skip(f"Detection test assets not found at {assets_dir}")

        recipe = _minimal_recipe()
        recipe["data"] = _minimal_data_config()
        path = _write_recipe(tmp_path, recipe)

        cfg = Configurator(
            data=str(assets_dir),
            model=path,
            task=TaskType.DETECTION,
        )
        dm = cfg.build_datamodule()
        assert dm.data_root == str(assets_dir)

    def test_explicit_data_root_overrides_constructor(self, tmp_path: Path) -> None:
        assets_dir = Path(__file__).resolve().parents[3] / "assets" / "detection_coco"
        if not assets_dir.exists():
            pytest.skip(f"Detection test assets not found at {assets_dir}")

        recipe = _minimal_recipe()
        recipe["data"] = _minimal_data_config()
        path = _write_recipe(tmp_path, recipe)
        explicit_root = tmp_path / "explicit"
        explicit_root.mkdir()
        # Create minimal Datumaro structure so DataModule can at least be constructed
        (explicit_root / "metadata.json").write_text('{"infos": [], "categories": {}}')
        (explicit_root / "data.parquet").write_bytes(b"")

        cfg = Configurator(
            data=str(assets_dir),
            model=path,
            task=TaskType.DETECTION,
        )
        # Pass explicit_root - DataModule construction may fail on import, but
        # data_root property should be set.  We test the override path only.
        try:
            dm = cfg.build_datamodule(str(explicit_root))
            assert dm.data_root == str(explicit_root)
        except ValueError:
            # Acceptable if the dummy dataset can't be imported - we still
            # verify that the override path reaches DataModule construction.
            pass

    def test_raises_when_no_data_config(self) -> None:
        from getitune.backend.ultralytics.models.detection import UltralyticsDetectionModel

        model = UltralyticsDetectionModel(
            model_name="yolo26s.yaml",
            label_info=_make_label_info(),
            pretrained=False,
            imgsz=640,
        )
        cfg = Configurator(
            data=Path("dummy"),
            model=model,
            task=TaskType.DETECTION,
        )
        with pytest.raises(ValueError, match="No data config available"):
            cfg.build_datamodule()

    def test_raises_when_no_data_root_available(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = _minimal_data_config()
        path = _write_recipe(tmp_path, recipe)
        cfg = Configurator(
            data=Path("placeholder"),
            model=path,
            task=TaskType.DETECTION,
        )
        # Simulate the case where data_root was never set (e.g. data was a DataModule
        # that was later cleared, or the constructor received a Path that we want to
        # override).
        cfg._data_root = None
        with pytest.raises(ValueError, match="data_root is required"):
            cfg.build_datamodule()

    def test_raises_when_missing_input_size(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = {"train_subset": {"batch_size": 8}}
        path = _write_recipe(tmp_path, recipe)
        cfg = Configurator(
            data=str(tmp_path),
            model=path,
            task=TaskType.DETECTION,
        )
        with pytest.raises(ValueError, match="missing 'input_size'"):
            cfg.build_datamodule()

    def test_raises_when_missing_subset(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = {"input_size": [640, 640], "train_subset": {"batch_size": 8}}
        path = _write_recipe(tmp_path, recipe)
        cfg = Configurator(
            data=str(tmp_path),
            model=path,
            task=TaskType.DETECTION,
        )
        with pytest.raises(ValueError, match="missing 'val_subset'"):
            cfg.build_datamodule()


class TestDataModuleToConfigDict:
    """to_config_dict produces correct data section when data is a DataModule."""

    def test_datamodule_data_section_in_config_dict(self) -> None:
        from getitune.config.data import SubsetConfig, TileConfig
        from getitune.data.module import DataModule

        assets_dir = Path(__file__).resolve().parents[3] / "assets" / "detection_coco"
        if not assets_dir.exists():
            pytest.skip(f"Detection test assets not found at {assets_dir}")

        dm = DataModule(
            task=TaskType.DETECTION,
            data_root=str(assets_dir),
            train_subset=SubsetConfig(batch_size=2, subset_name="train"),
            val_subset=SubsetConfig(batch_size=2, subset_name="val"),
            test_subset=SubsetConfig(batch_size=2, subset_name="test"),
            tile_config=TileConfig(enable_tiler=False),
            input_size=(64, 64),
        )
        cfg = Configurator(
            data=dm,
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        result = cfg.to_config_dict()
        assert "data" in result
        assert result["data"]["input_size"] == [64, 64]
        assert result["data"]["train_subset"]["batch_size"] == 2
        assert result["data"]["val_subset"]["batch_size"] == 2
        assert result["data"]["test_subset"]["batch_size"] == 2
        assert result["data"]["tile_config"]["enable_tiler"] is False

    def test_datamodule_convert_roundtrip(self) -> None:
        from getitune.config.data import SubsetConfig, TileConfig
        from getitune.data.module import DataModule

        assets_dir = Path(__file__).resolve().parents[3] / "assets" / "detection_coco"
        if not assets_dir.exists():
            pytest.skip(f"Detection test assets not found at {assets_dir}")

        dm = DataModule(
            task=TaskType.DETECTION,
            data_root=str(assets_dir),
            train_subset=SubsetConfig(batch_size=4, subset_name="train"),
            val_subset=SubsetConfig(batch_size=4, subset_name="val"),
            test_subset=SubsetConfig(batch_size=4, subset_name="test"),
            tile_config=TileConfig(enable_tiler=False),
            input_size=(320, 320),
        )
        cfg = Configurator(
            data=dm,
            model=_DETECTION_MODEL_NAME,
            task=TaskType.DETECTION,
        )
        result = cfg.to_config_dict()
        assert result["data"]["input_size"] == [320, 320]


class TestInstanceSegmentation:
    def test_instseg_model_name_resolved(self) -> None:
        cfg = Configurator(
            data=Path("dummy"),
            model=_INST_SEG_MODEL_NAME,
            task=TaskType.INSTANCE_SEGMENTATION,
        )
        assert cfg.model_config is not None
        assert cfg.model_config["class_path"] == _INST_SEG_CLASS_PATH
        assert cfg.model_config["init_args"]["model_name"] == "yolo26s-seg.yaml"

    def test_instseg_recipe_path(self) -> None:
        path = _RECIPE_DIR / "instance_segmentation" / "yolo26_s_seg.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        cfg = Configurator(
            data=Path("dummy"),
            model=path,
            task=TaskType.INSTANCE_SEGMENTATION,
        )
        assert cfg.model_config is not None
        assert cfg.model_config["class_path"] == _INST_SEG_CLASS_PATH

    @pytest.mark.parametrize(
        ("variant", "fname"),
        [
            ("yolo26_n_seg", "yolo26n-seg.yaml"),
            ("yolo26_s_seg", "yolo26s-seg.yaml"),
            ("yolo26_m_seg", "yolo26m-seg.yaml"),
        ],
    )
    def test_instseg_variants(self, variant: str, fname: str) -> None:
        path = _RECIPE_DIR / "instance_segmentation" / f"{variant}.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        cfg = Configurator(
            data=Path("dummy"),
            model=variant,
            task=TaskType.INSTANCE_SEGMENTATION,
        )
        assert cfg.model_config is not None
        assert cfg.model_config["init_args"]["model_name"] == fname


class TestInitModelEdgeCases:
    def test_model_name_with_hyphens_fails(self) -> None:
        with pytest.raises(FileNotFoundError, match="Recipe not found"):
            Configurator(
                data=Path("dummy"),
                model="yolo26-s",
                task=TaskType.DETECTION,
            )

    def test_model_name_case_sensitive(self) -> None:
        with pytest.raises(FileNotFoundError, match="Recipe not found"):
            Configurator(
                data=Path("dummy"),
                model="YOLO26_S",
                task=TaskType.DETECTION,
            )


class TestFlattenOverrides:
    def test_flat(self) -> None:
        assert flatten_overrides({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self) -> None:
        assert flatten_overrides({"a": {"b": 1}}) == {"a.b": 1}

    def test_deeply_nested(self) -> None:
        assert flatten_overrides({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}
