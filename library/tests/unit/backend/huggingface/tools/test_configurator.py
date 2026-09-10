# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hugging Face recipe Configurator.

Uses offline, network-free recipes: ``checkpoint`` is a nested
``transformers.PretrainedConfig`` class_path rather than a bare Hub id, the
same offline-construction path every other HF backend test already relies
on. The real, network-downloading recipes under ``recipe/`` (used by
``create_engine("mask2former_swin_t", data=...)`` in real usage) are
exercised manually against ``data/wgisd``, not in this fast unit suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import transformers as tf

from getitune.backend.huggingface.engine import HFEngine
from getitune.backend.huggingface.models import HFInstSegModel
from getitune.backend.huggingface.tools.configurator import Configurator
from getitune.data.module import DataModule
from getitune.types.task import TaskType


def _write_offline_recipe(tmp_path: Path) -> Path:
    """Write a recipe whose model builds from a bare ``PretrainedConfig``, not a Hub id."""
    base_data_config = (
        Path(__file__).resolve().parents[5]
        / "src"
        / "getitune"
        / "recipe"
        / "_base_"
        / "data"
        / "instance_segmentation.yaml"
    )
    recipe = tmp_path / "hf_offline.yaml"
    recipe.write_text(
        f"""
backend: huggingface
task: INSTANCE_SEGMENTATION

model:
  class_path: getitune.backend.huggingface.models.instance_segmentation.HFInstSegModel
  init_args:
    checkpoint:
      class_path: transformers.Mask2FormerConfig
      init_args:
        num_queries: 10
        decoder_layers: 2
        encoder_layers: 2
    pretrained: true
    data_input_params:
      input_size: [64, 64]

data: {base_data_config}
""".strip()
    )
    return recipe


def _write_offline_recipe_with_training(tmp_path: Path) -> Path:
    """Same as :func:`_write_offline_recipe`, plus a ``training:`` block."""
    base_data_config = (
        Path(__file__).resolve().parents[5]
        / "src"
        / "getitune"
        / "recipe"
        / "_base_"
        / "data"
        / "instance_segmentation.yaml"
    )
    recipe = tmp_path / "hf_offline_training.yaml"
    recipe.write_text(
        f"""
backend: huggingface
task: INSTANCE_SEGMENTATION

model:
  class_path: getitune.backend.huggingface.models.instance_segmentation.HFInstSegModel
  init_args:
    checkpoint:
      class_path: transformers.Mask2FormerConfig
      init_args:
        num_queries: 10
        decoder_layers: 2
        encoder_layers: 2
    pretrained: true
    data_input_params:
      input_size: [64, 64]

training:
  max_epochs: 3
  batch: 2
  learning_rate: 0.0005
  precision: "32"

data: {base_data_config}
""".strip()
    )
    return recipe


def _write_offline_semantic_segmentation_recipe(tmp_path: Path) -> Path:
    """A semantic-seg recipe — the one task whose ``label_info`` is a ``SegLabelInfo`` (G37)."""
    base_data_config = (
        Path(__file__).resolve().parents[5]
        / "src"
        / "getitune"
        / "recipe"
        / "_base_"
        / "data"
        / "semantic_segmentation.yaml"
    )
    recipe = tmp_path / "hf_offline_semantic_segmentation.yaml"
    recipe.write_text(
        f"""
backend: huggingface
task: SEMANTIC_SEGMENTATION

model:
  class_path: getitune.backend.huggingface.models.semantic_segmentation.HFSemanticSegModel
  init_args:
    checkpoint:
      class_path: transformers.SegformerConfig
      init_args:
        num_encoder_blocks: 2
        depths: [1, 1]
    pretrained: true
    data_input_params:
      input_size: [64, 64]

data: {base_data_config}
""".strip()
    )
    return recipe


@pytest.fixture
def datamodule() -> DataModule:
    return DataModule(task=TaskType.INSTANCE_SEGMENTATION, data_root="tests/assets/instance_segmentation_coco")


def test_configurator_accepts_an_already_built_model(datamodule: DataModule) -> None:
    model = HFInstSegModel(_pretrained_config(), datamodule.label_info)
    configurator = Configurator(data=datamodule, model=model)
    assert configurator.create_model(datamodule.label_info) is model


def test_configurator_rejects_bad_data_type() -> None:
    with pytest.raises(TypeError, match="data must be PathLike or DataModule"):
        Configurator(data=123, model="mask2former_swin_t", task="INSTANCE_SEGMENTATION")  # type: ignore[arg-type]


def test_configurator_rejects_bad_model_type(datamodule: DataModule) -> None:
    with pytest.raises(TypeError, match="model must be str, PathLike, or HFModel"):
        Configurator(data=datamodule, model=123)  # type: ignore[arg-type]


def test_configurator_rejects_unsupported_task(datamodule: DataModule) -> None:
    with pytest.raises(ValueError, match="Unsupported task"):
        Configurator(data=datamodule, model="mask2former_swin_t", task="KEYPOINT_DETECTION")


def test_configurator_resolves_a_bare_recipe_name_without_task_from_recipe(tmp_path: Path) -> None:
    """A full recipe path carries its own 'task:' field; task= isn't required."""
    recipe_path = _write_offline_recipe(tmp_path)
    configurator = Configurator(data="tests/assets/instance_segmentation_coco", model=recipe_path)
    assert configurator.task == "INSTANCE_SEGMENTATION"


def test_bare_model_name_requires_task_to_resolve() -> None:
    with pytest.raises(ValueError, match="Cannot resolve model name"):
        Configurator(data="tests/assets/instance_segmentation_coco", model="mask2former_swin_t")


def test_bare_model_name_resolves_against_the_recipe_tree() -> None:
    configurator = Configurator(
        data="tests/assets/instance_segmentation_coco", model="mask2former_swin_t", task="INSTANCE_SEGMENTATION"
    )
    assert configurator.task == "INSTANCE_SEGMENTATION"


def test_unknown_bare_model_name_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Recipe not found"):
        Configurator(
            data="tests/assets/instance_segmentation_coco", model="does_not_exist", task="INSTANCE_SEGMENTATION"
        )


def test_build_datamodule_uses_the_recipes_data_config(tmp_path: Path) -> None:
    recipe_path = _write_offline_recipe(tmp_path)
    configurator = Configurator(data="tests/assets/instance_segmentation_coco", model=recipe_path)
    dm = configurator.build_datamodule()
    assert set(dm.subsets) == {"train", "val", "test"}
    assert dm.subsets["train"].label_info.num_classes > 0


def test_build_datamodule_returns_the_datamodule_passed_in_directly(datamodule: DataModule) -> None:
    configurator = Configurator(data=datamodule, model=HFInstSegModel(_pretrained_config(), datamodule.label_info))
    assert configurator.build_datamodule() is datamodule


def test_create_model_builds_from_the_recipes_class_path_and_init_args(tmp_path: Path) -> None:
    recipe_path = _write_offline_recipe(tmp_path)
    configurator = Configurator(data="tests/assets/instance_segmentation_coco", model=recipe_path)
    dm = configurator.build_datamodule()

    model = configurator.create_model(dm.label_info)

    assert isinstance(model, HFInstSegModel)
    assert model.hf_model.config.num_queries == 10
    assert model.label_info.label_names == dm.label_info.label_names


def test_create_model_accepts_a_bare_num_classes_int(tmp_path: Path) -> None:
    recipe_path = _write_offline_recipe(tmp_path)
    configurator = Configurator(data="tests/assets/instance_segmentation_coco", model=recipe_path)

    model = configurator.create_model(3)

    assert model.label_info.num_classes == 3


def test_create_model_supports_seglabelinfo_for_semantic_segmentation(tmp_path: Path) -> None:
    """Regression test for G37.

    ``DataModule.label_info`` for semantic segmentation is a ``SegLabelInfo``,
    whose ``.as_dict()`` includes ``ignore_index`` — a field the plain
    ``LabelInfo | int | list[str]`` union has no slot for. This used to crash
    ``create_model()`` for this task only; the other four tasks' recipes
    (plain ``LabelInfo``) were unaffected, which is exactly why the bug went
    unnoticed until every task's recipe path was tested, not just one.
    """
    from getitune.backend.huggingface.models import HFSemanticSegModel
    from getitune.types.label import SegLabelInfo

    recipe_path = _write_offline_semantic_segmentation_recipe(tmp_path)
    configurator = Configurator(data="tests/assets/segmentation_pets", model=recipe_path)
    dm = configurator.build_datamodule()
    label_info = dm.label_info
    assert isinstance(label_info, SegLabelInfo)

    model = configurator.create_model(label_info)

    assert isinstance(model, HFSemanticSegModel)
    assert model.label_info.num_classes == label_info.num_classes
    assert model._ignore_index == label_info.ignore_index


def test_create_engine_forwards_the_recipes_training_block(tmp_path: Path) -> None:
    """The recipe's ``training:`` block reaches ``HFEngine.train()``'s defaults."""
    recipe_path = _write_offline_recipe_with_training(tmp_path)
    configurator = Configurator(data="tests/assets/instance_segmentation_coco", model=recipe_path)
    dm = configurator.build_datamodule()
    model = configurator.create_model(dm.label_info)

    engine = configurator.create_engine(model=model, work_dir=tmp_path / "wd")

    assert engine._training_defaults == {
        "max_epochs": 3,
        "batch": 2,
        "learning_rate": 0.0005,
        "precision": "32",
    }


def test_create_engine_builds_an_hf_engine_from_a_recipe_path(tmp_path: Path) -> None:
    recipe_path = _write_offline_recipe(tmp_path)
    configurator = Configurator(data="tests/assets/instance_segmentation_coco", model=recipe_path)
    dm = configurator.build_datamodule()
    model = configurator.create_model(dm.label_info)

    engine = configurator.create_engine(model=model, work_dir=tmp_path / "wd")

    assert isinstance(engine, HFEngine)
    assert engine.model is model
    assert engine.datamodule is dm


def test_create_engine_requires_data_when_none_was_configured(datamodule: DataModule) -> None:
    model = HFInstSegModel(_pretrained_config(), datamodule.label_info)
    configurator = Configurator(data=datamodule, model=model)
    configurator._datamodule = None
    configurator._data_root = None
    with pytest.raises(ValueError, match="No data available"):
        configurator.create_engine(model=model)


def _pretrained_config() -> tf.Mask2FormerConfig:
    return tf.Mask2FormerConfig(num_queries=10, decoder_layers=2, encoder_layers=2)
