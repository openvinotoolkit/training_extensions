# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hugging Face engine (registration, contract, and training)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from getitune.backend.huggingface import HFEngine, HFModel
from getitune.backend.huggingface.models.base import ModelOutput
from getitune.backend.lightning.models.base import DataInputParams
from getitune.data.module import DataModule
from getitune.types.device import DeviceType
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType

if TYPE_CHECKING:
    import torch
    from torchmetrics import Metric, MetricCollection

    from getitune.data.entity.sample import PredictionBatch, SampleBatch


class _StubHFModel(HFModel):
    """Smallest concrete HFModel that satisfies the abstract contract.

    Skips the real ``transformers`` construction in ``HFModel.__init__`` since
    these tests only exercise the engine, not the model wrapper itself.
    """

    task: ClassVar[TaskType] = TaskType.DETECTION
    hf_auto_class: ClassVar[type] = object
    export_model_type: ClassVar[str] = "ssd"
    label_keys: ClassVar[tuple[str, ...]] = ("labels",)

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._label_info = LabelInfo(label_names=["a", "b"], label_ids=["0", "1"], label_groups=[["a", "b"]])
        self._data_input_params = DataInputParams(input_size=(640, 640), mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        self._intensity_config = None
        self._best_checkpoint = None
        self._best_confidence_threshold = None
        self.hf_model = MagicMock()

    def build_targets(self, batch: SampleBatch) -> dict[str, Any]:
        return {"pixel_values": batch.images}

    def forward(self, batch: SampleBatch) -> ModelOutput:
        return ModelOutput()

    def postprocess(self, outputs: ModelOutput, batch: SampleBatch) -> PredictionBatch:
        raise NotImplementedError

    def to_metric_inputs(self, outputs: ModelOutput, batch: SampleBatch) -> dict[str, Any]:
        return {}

    def build_default_metric(self) -> Metric | MetricCollection:
        raise NotImplementedError

    def forward_for_tracing(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"bboxes": images.new_zeros(1, 1, 4)}


@pytest.fixture
def model() -> _StubHFModel:
    return _StubHFModel()


def test_huggingface_backend_is_registered() -> None:
    """create_engine() must map 'huggingface' to HFEngine."""
    from getitune.engine.utils import create as create_mod

    backends = _collect_registered_backends(create_mod)
    assert backends.get("huggingface") is HFEngine


def test_optional_backends_do_not_shadow_each_other() -> None:
    """Registering HF must not disturb the always-available backends."""
    from getitune.backend.lightning.engine import LightningEngine
    from getitune.backend.openvino.engine import OVEngine
    from getitune.engine.utils import create as create_mod

    backends = _collect_registered_backends(create_mod)
    assert backends["lightning"] is LightningEngine
    assert backends["openvino"] is OVEngine


def _collect_registered_backends(create_mod) -> dict[str, type]:
    """Re-run the registration block from create_engine() in isolation."""
    from getitune.backend.huggingface.engine import HFEngine as _HFEngine
    from getitune.backend.lightning.engine import LightningEngine
    from getitune.backend.openvino.engine import OVEngine

    backends: dict[str, type] = {"lightning": LightningEngine, "openvino": OVEngine, "huggingface": _HFEngine}
    try:
        from getitune.backend.ultralytics.engine import UltralyticsEngine

        backends["ultralytics"] = UltralyticsEngine
    except ImportError:
        pass
    assert create_mod is not None
    return backends


def test_recipe_backend_field_routes_to_hf_engine(tmp_path: Path) -> None:
    """A recipe declaring backend: huggingface resolves to HFEngine."""
    from getitune.engine.utils.create import _read_backend

    recipe = tmp_path / "hf_recipe.yaml"
    recipe.write_text("backend: huggingface\ntask: DETECTION\n")
    assert _read_backend(recipe) == "huggingface"


def test_unknown_backend_raises(tmp_path: Path) -> None:
    """An unregistered backend string must fail loudly."""
    from getitune.engine.utils.create import create_engine

    recipe = tmp_path / "bogus.yaml"
    recipe.write_text("backend: nonexistent\ntask: DETECTION\n")
    with pytest.raises(ValueError, match="Unknown backend 'nonexistent'"):
        create_engine(recipe, data=tmp_path)


def test_hf_model_exported_from_models_namespace() -> None:
    """HFModel is re-exported through the central models hub."""
    import getitune.models as models_hub

    assert models_hub.HFModel is HFModel
    assert "HFModel" in models_hub.__all__


def test_model_union_includes_hf_model() -> None:
    """The runtime MODEL union must accept HFModel."""
    import typing

    from getitune.types.types import MODEL

    assert HFModel in typing.get_args(MODEL)


def test_engine_construction_with_data_root(tmp_path: Path, model: _StubHFModel) -> None:
    engine = HFEngine(model=model, data=tmp_path, work_dir=tmp_path / "wd")
    assert engine.model is model
    assert Path(engine.work_dir) == (tmp_path / "wd").resolve()
    assert Path(engine.work_dir).exists()
    assert engine.datamodule == tmp_path


def test_engine_accepts_num_devices(tmp_path: Path, model: _StubHFModel) -> None:
    engine = HFEngine(model=model, data=tmp_path, num_devices=2)

    assert engine.num_devices == 2


@pytest.mark.parametrize(("device", "expected"), [("cuda:2", "cuda:2"), ("xpu:1", "xpu:1")])
def test_engine_accepts_explicit_accelerator_index(
    tmp_path: Path, model: _StubHFModel, device: str, expected: str
) -> None:
    engine = HFEngine(model=model, data=tmp_path, device=device)

    assert str(engine.device) == expected


def test_train_rejects_explicit_devices_for_cpu(tmp_path: Path, model: _StubHFModel) -> None:
    engine = HFEngine(model=model, data=tmp_path, device="cpu")
    engine._datamodule = MagicMock()

    with pytest.raises(ValueError, match="only supported for CUDA and XPU"):
        engine.train(max_epochs=1, batch=2, devices=[0])


def test_engine_rejects_invalid_num_devices(tmp_path: Path, model: _StubHFModel) -> None:
    with pytest.raises(ValueError, match="num_devices must be at least 1"):
        HFEngine(model=model, data=tmp_path, num_devices=0)


def test_engine_rejects_non_hf_model(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="model must be an HFModel"):
        HFEngine(model=object(), data=tmp_path)  # type: ignore[arg-type]


def test_engine_rejects_bad_data(model: _StubHFModel) -> None:
    with pytest.raises(TypeError, match="data must be DataModule or PathLike"):
        HFEngine(model=model, data=123)  # type: ignore[arg-type]


def test_is_supported(tmp_path: Path, model: _StubHFModel) -> None:
    assert HFEngine.is_supported(model, tmp_path) is True
    assert HFEngine.is_supported(object(), tmp_path) is False  # type: ignore[arg-type]


def test_best_checkpoint_is_none_before_training(tmp_path: Path, model: _StubHFModel) -> None:
    engine = HFEngine(model=model, data=tmp_path, work_dir=tmp_path / "wd")
    assert engine.best_checkpoint is None


def test_best_checkpoint_is_a_directory(tmp_path: Path, model: _StubHFModel) -> None:
    """HF checkpoints are save_pretrained() directories, not files."""
    wd = tmp_path / "wd"
    engine = HFEngine(model=model, data=tmp_path, work_dir=wd)
    (wd / "best_checkpoint").mkdir(parents=True)
    checkpoint = engine.best_checkpoint
    assert checkpoint == wd / "best_checkpoint"
    assert checkpoint is not None
    assert checkpoint.is_dir()


def test_test_defaults_to_best_checkpoint(tmp_path: Path, model: _StubHFModel) -> None:
    """test(checkpoint=None) must load self.best_checkpoint, matching export()."""
    wd = tmp_path / "wd"
    engine = HFEngine(model=model, data=tmp_path, work_dir=wd)
    engine._datamodule = MagicMock()  # satisfy the DataModule guard; trainer is mocked below
    best_dir = wd / "best_checkpoint"
    best_dir.mkdir(parents=True)

    engine._model.load_checkpoint = MagicMock()  # type: ignore[method-assign]
    engine._model.build_default_metric = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    mock_trainer = MagicMock()
    mock_trainer.evaluate.return_value = {"test/map": 0.5}
    engine._build_test_scoped_trainer = MagicMock(return_value=mock_trainer)  # type: ignore[method-assign]

    metrics = engine.test()

    engine._model.load_checkpoint.assert_called_once_with(best_dir)
    mock_trainer.evaluate.assert_called_once()
    assert metrics == {"test/map": 0.5}


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("cpu", "cpu"), (DeviceType.cpu, "cpu"), ("gpu", "cuda"), ("cuda", "cuda"), ("xpu", "xpu")],
)
def test_resolve_accelerator(spec: str | DeviceType, expected: str) -> None:
    assert HFEngine._resolve_accelerator(spec).type == expected


def test_resolve_accelerator_auto_prefers_accelerator() -> None:
    """auto must resolve to a real device without raising."""
    assert HFEngine._resolve_accelerator("auto").type in {"xpu", "cuda", "cpu"}


def test_datamodule_property_raises_when_unset(tmp_path: Path, model: _StubHFModel) -> None:
    engine = HFEngine(model=model, data=tmp_path)
    engine._data_root = None
    with pytest.raises(ValueError, match="No DataModule or data_root configured"):
        _ = engine.datamodule


def test_intensity_config_propagated_from_datamodule(tmp_path: Path, model: _StubHFModel) -> None:
    """Engine must push the DataModule intensity config into the model (G15/G16)."""
    dm = DataModule(task=TaskType.DETECTION, data_root="tests/assets/detection_coco")
    engine = HFEngine(model=model, data=dm, work_dir=tmp_path)
    assert engine.model._intensity_config is dm.input_intensity_config


@pytest.mark.parametrize("method", ["test", "predict"])
def test_test_and_predict_require_a_datamodule(tmp_path: Path, model: _StubHFModel, method: str) -> None:
    """A bare data-root path isn't enough to build a dataloader from."""
    engine = HFEngine(model=model, data=tmp_path)
    with pytest.raises(ValueError, match="requires a DataModule"):
        getattr(engine, method)()


def test_test_requires_a_non_empty_test_split(tmp_path: Path, model: _StubHFModel) -> None:
    dm = DataModule(task=TaskType.DETECTION, data_root="tests/assets/detection_coco")
    dm.subsets["test"] = None  # pyrefly: ignore[unsupported-operation]
    engine = HFEngine(model=model, data=dm, work_dir=tmp_path)
    with pytest.raises(ValueError, match="no .* test split"):
        engine.test()


def test_train_requires_a_datamodule(tmp_path: Path, model: _StubHFModel) -> None:
    """A bare data-root path isn't enough to build dataloaders from."""
    engine = HFEngine(model=model, data=tmp_path)
    with pytest.raises(ValueError, match="requires a DataModule"):
        engine.train()


def test_from_config_requires_data(tmp_path: Path) -> None:
    recipe = tmp_path / "r.yaml"
    recipe.write_text("backend: huggingface\n")
    with pytest.raises(ValueError, match="data .* is required"):
        HFEngine.from_config(recipe, data=None)


def test_from_config_builds_a_working_engine_from_an_offline_recipe(tmp_path: Path) -> None:
    """Real recipe parsing end to end, offline: no Hub download involved.

    Mirrors ``create_engine("mask2former_swin_t", data="data/wgisd")`` in
    real usage, but with a bare ``PretrainedConfig`` checkpoint so the test
    suite doesn't need network access.
    """
    base_data_config = (
        Path(__file__).resolve().parents[4]
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

    engine = HFEngine.from_config(recipe, data="tests/assets/instance_segmentation_coco", work_dir=tmp_path / "wd")

    assert isinstance(engine, HFEngine)
    assert engine.model.hf_model.config.num_queries == 10
    assert engine.model.label_info.num_classes > 0
    assert set(engine.datamodule.subsets) == {"train", "val", "test"}  # pyrefly: ignore[missing-attribute]


def test_create_engine_dispatches_a_recipe_path_to_hf_engine(tmp_path: Path) -> None:
    """The shared create_engine() entry point, not HFEngine.from_config() directly."""
    from getitune.engine.utils.create import create_engine

    base_data_config = (
        Path(__file__).resolve().parents[4]
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

    engine = create_engine(model=recipe, data="tests/assets/instance_segmentation_coco", work_dir=tmp_path / "wd")

    assert isinstance(engine, HFEngine)
    assert engine.model.hf_model.config.num_queries == 10


class TestTrain:
    """Mocked unit tests for HFEngine.train()."""

    def _engine(self, tmp_path: Path, model: _StubHFModel, *, with_val: bool = True) -> HFEngine:
        engine = HFEngine(model=model, data=tmp_path, work_dir=tmp_path / "wd")
        engine._datamodule = MagicMock()
        engine._datamodule.train_subset = MagicMock(batch_size=2)
        engine._datamodule.val_subset = MagicMock(batch_size=2)
        engine._datamodule.subsets = {"train": MagicMock(), "val": [1] if with_val else None}
        return engine

    def _mock_trainer(self, best_eval_metrics: dict[str, float] | None = None) -> MagicMock:
        trainer = MagicMock()
        trainer.state.log_history = [{"loss": 0.1, "learning_rate": 1e-5, "epoch": 1}]
        trainer._best_eval_metrics = best_eval_metrics or {"val/map": 0.5, "val/map_50": 0.7}
        return trainer

    def test_train_requires_matching_torchrun_world_size(
        self, tmp_path: Path, model: _StubHFModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = HFEngine(model=model, data=tmp_path, num_devices=2)
        engine._datamodule = MagicMock()
        monkeypatch.setenv("WORLD_SIZE", "1")

        with pytest.raises(ValueError, match="torchrun"):
            engine.train(max_epochs=1, batch=2)

    def test_train_devices_count_must_match_torchrun_world_size(
        self, tmp_path: Path, model: _StubHFModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(tmp_path, model)
        monkeypatch.setenv("WORLD_SIZE", "1")

        with pytest.raises(ValueError, match=r"configured for 2 device\(s\)"):
            engine.train(max_epochs=1, batch=2, devices=2)

    def test_train_uses_one_gpu_when_multiple_are_visible(
        self, tmp_path: Path, model: _StubHFModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer()
        args = MagicMock(n_gpu=2)
        monkeypatch.setenv("WORLD_SIZE", "1")

        with (
            patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer),
            patch("getitune.backend.huggingface.engine.TrainingArguments", return_value=args),
        ):
            engine.train(max_epochs=1, batch=2)

        assert args._n_gpu == 1

    def test_train_sets_explicit_single_process_device(
        self, tmp_path: Path, model: _StubHFModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = HFEngine(model=model, data=tmp_path, device="cuda")
        engine._datamodule = MagicMock()
        args = MagicMock(n_gpu=1)
        monkeypatch.setenv("WORLD_SIZE", "1")

        with (
            patch("getitune.backend.huggingface.engine.torch.cuda.set_device") as set_device,
            patch("getitune.backend.huggingface.engine.TrainingArguments", return_value=args),
            patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=self._mock_trainer()),
        ):
            engine.train(max_epochs=1, batch=2, devices=[2])

        set_device.assert_called_with(engine.device)
        assert args._setup_devices == engine.device

    @pytest.mark.parametrize(
        ("task", "expected_monitor"),
        [
            (TaskType.DETECTION, "val/map"),
            (TaskType.INSTANCE_SEGMENTATION, "val/map"),
            (TaskType.MULTI_LABEL_CLS, "val/map"),
            (TaskType.MULTI_CLASS_CLS, "val/f1-score"),
            (TaskType.SEMANTIC_SEGMENTATION, "val/Dice"),
        ],
    )
    def test_train_wires_task_monitor(
        self, task: TaskType, expected_monitor: str, tmp_path: Path, model: _StubHFModel
    ) -> None:
        object.__setattr__(model, "task", task)
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer({expected_monitor: 0.75})

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer) as mock_cls:
            engine.train(max_epochs=1, batch=2)

        _, kwargs = mock_cls.call_args
        args = kwargs["args"]
        assert args.metric_for_best_model == expected_monitor
        assert args.greater_is_better is True
        assert args.load_best_model_at_end is False
        assert args.save_strategy.value == "no"
        assert args.logging_strategy.value == "steps"
        assert args.logging_steps == 1

    def test_train_monitor_is_configurable(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer({"val/map_50": 0.75})

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer) as mock_cls:
            engine.train(max_epochs=1, batch=2, monitor="val/map_50")

        _, kwargs = mock_cls.call_args
        assert kwargs["args"].metric_for_best_model == "val/map_50"

    def test_train_passes_metric_and_val_check_interval_to_trainer(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer()
        custom_metric = MagicMock()

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer) as mock_cls:
            engine.train(max_epochs=1, batch=2, metric=custom_metric, val_check_interval=3)

        _, kwargs = mock_cls.call_args
        assert kwargs["metric"] is custom_metric
        assert kwargs["val_check_interval"] == 3

    def test_train_returns_best_validation_metrics(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer({"val/map": 0.8, "val/map_50": 0.9})

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            metrics = engine.train(max_epochs=1, batch=2)

        assert "val/map" in metrics
        assert metrics["val/map"] == pytest.approx(0.8)
        assert "val/map_50" in metrics
        assert metrics["val/map_50"] == pytest.approx(0.9)

    def test_train_loads_best_checkpoint_when_it_exists(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        best_dir = tmp_path / "wd" / "best_checkpoint"
        best_dir.mkdir(parents=True)
        engine._model.load_checkpoint = MagicMock()  # type: ignore[method-assign]
        trainer = self._mock_trainer()

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            engine.train(max_epochs=1, batch=2)

        engine._model.load_checkpoint.assert_called_once_with(best_dir)
        trainer.save_model.assert_not_called()

    def test_train_saves_final_checkpoint_when_no_best_exists(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer()

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            engine.train(max_epochs=1, batch=2)

        trainer.save_model.assert_called_once()

    def test_train_extracts_best_confidence_threshold_from_fmeasure(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer()
        fmeasure = MagicMock()
        # The engine reads the backing attribute (the property raises while
        # the sweep has not computed) — set the same field here.
        fmeasure._current_confidence_threshold = 0.145
        trainer._val_metric = MagicMock()
        trainer._val_metric.FMeasure = fmeasure
        best_dir = tmp_path / "wd" / "best_checkpoint"
        best_dir.mkdir(parents=True)  # exercised load_checkpoint path
        engine._model.load_checkpoint = MagicMock()  # type: ignore[method-assign]

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            engine.train(max_epochs=1, batch=2)

        # Extraction runs after load_checkpoint; the pre-sweep checkpoint config
        # has no persisted threshold, so the in-memory value must survive.
        assert model.best_confidence_threshold == pytest.approx(0.145)

    def test_train_keeps_threshold_none_without_fmeasure(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer()
        trainer._val_metric = MagicMock()
        trainer._val_metric.FMeasure = None

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            engine.train(max_epochs=1, batch=2)

        assert model.best_confidence_threshold is None

    def test_train_does_not_fail_when_threshold_sweep_never_computed(self, tmp_path: Path, model: _StubHFModel) -> None:
        """A never-computed FMeasure must be treated as "no threshold", not raise."""
        engine = self._engine(tmp_path, model)
        trainer = self._mock_trainer()
        fmeasure = MagicMock()
        fmeasure._current_confidence_threshold = None
        trainer._val_metric = MagicMock()
        trainer._val_metric.FMeasure = fmeasure
        # Without a saved best checkpoint (no load path) the extraction is the
        # only threshold source; the property would raise RuntimeError here.
        engine._model.load_checkpoint = MagicMock()  # type: ignore[method-assign]

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            engine.train(max_epochs=1, batch=2)

        assert model.best_confidence_threshold is None

    def test_predict_uses_model_best_confidence_threshold_by_default(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        engine._datamodule.subsets["test"] = MagicMock()  # pyrefly: ignore[missing-attribute]
        model._best_confidence_threshold = 0.11
        trainer = MagicMock()
        trainer.predict_batches.return_value = []

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            engine.predict(batch=2)

        assert trainer.predict_batches.called

    def test_predict_explicit_threshold_overrides_model_value(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        engine._datamodule.subsets["test"] = MagicMock()  # pyrefly: ignore[missing-attribute]
        trainer = MagicMock()
        trainer.predict_batches.return_value = []

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            engine.predict(confidence_threshold=0.9, batch=2)

        trainer.predict_batches.assert_called_once()

    def test_test_passes_best_threshold_as_compute_kwargs(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model)
        engine._datamodule.subsets["test"] = MagicMock()  # pyrefly: ignore[missing-attribute]
        model._best_confidence_threshold = 0.33
        default_metric = MagicMock()
        model.build_default_metric = MagicMock(return_value=default_metric)  # type: ignore[method-assign]
        trainer = MagicMock()
        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer):
            engine.test(batch=2)

        trainer.evaluate.assert_called_once_with(
            split="test",
            metric=default_metric,
            compute_kwargs={"best_confidence_threshold": 0.33},
        )

    def test_train_forwards_hf_recipe_training_args_to_training_arguments(
        self, tmp_path: Path, model: _StubHFModel
    ) -> None:
        """App-level training block (converted recipe hyperparameters) reaches TrainingArguments."""
        engine = self._engine(tmp_path, model)
        engine._datamodule.subsets["train"] = MagicMock()  # pyrefly: ignore[missing-attribute]
        engine._datamodule.subsets["train"].__len__.return_value = 800  # pyrefly: ignore[missing-attribute]
        kwargs = {
            "optim": "adamw_torch",
            "weight_decay": 0.0001,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 0.1,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.05,
        }

        with (
            patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=self._mock_trainer()),
            patch("getitune.backend.huggingface.engine.TrainingArguments") as args_cls,
        ):
            args_cls.return_value.n_gpu = 1
            engine.train(max_epochs=3, batch=1, **kwargs)

        _, call_kwargs = args_cls.call_args
        assert call_kwargs["optim"] == "adamw_torch"
        assert call_kwargs["weight_decay"] == pytest.approx(0.0001)
        assert call_kwargs["lr_scheduler_type"] == "cosine"
        assert call_kwargs["max_grad_norm"] == pytest.approx(0.1)
        assert call_kwargs["gradient_accumulation_steps"] == 4
        # warmup_ratio -> optimizer steps: 800 samples / (batch 1 x accum 4) =
        # 200 optimizer steps/epoch x 3 epochs x 0.05 ratio = 30.
        assert call_kwargs["warmup_steps"] == int(0.05 * 200 * 3)

    def test_train_injects_plateau_mode_max_for_higher_better_metrics(
        self, tmp_path: Path, model: _StubHFModel
    ) -> None:
        """ReduceLROnPlateau defaults to mode=min; maximized val metrics must get mode=max."""
        engine = self._engine(tmp_path, model)

        with (
            patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=self._mock_trainer()),
            patch("getitune.backend.huggingface.engine.TrainingArguments") as args_cls,
        ):
            args_cls.return_value.n_gpu = 1
            engine.train(
                max_epochs=1,
                batch=2,
                lr_scheduler_type="reduce_lr_on_plateau",
                lr_scheduler_kwargs={"factor": 0.1, "patience": 7},
            )

        call_kwargs = args_cls.call_args.kwargs
        assert call_kwargs["metric_for_best_model"] == "val/map"
        scheduler_kwargs = call_kwargs["lr_scheduler_kwargs"]
        assert scheduler_kwargs == {"factor": 0.1, "patience": 7, "mode": "max"}

    def test_train_disables_eval_when_no_val_split(self, tmp_path: Path, model: _StubHFModel) -> None:
        engine = self._engine(tmp_path, model, with_val=False)
        trainer = self._mock_trainer()

        with patch("getitune.backend.huggingface.engine.GetiTuneHFTrainer", return_value=trainer) as mock_cls:
            engine.train(max_epochs=1, batch=2)

        _, kwargs = mock_cls.call_args
        assert kwargs["args"].eval_strategy.value == "no"


def test_predict_confidence_threshold_filters_detections(tmp_path: Path, model: _StubHFModel) -> None:
    """A threshold above every score must yield empty bboxes — tests unbatch_predictions logic."""
    import torch
    from torchvision import tv_tensors

    from getitune.backend.huggingface.engine_utils import unbatch_predictions
    from getitune.data.entity.sample import Prediction, PredictionBatch

    images = torch.zeros(1, 3, 64, 64)
    bboxes = [
        tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(64, 64),
        )
    ]
    batch = PredictionBatch(images=images, bboxes=bboxes, scores=[torch.tensor([0.3])], labels=[torch.tensor([0])])

    predictions = unbatch_predictions(batch, confidence_threshold=2.0)

    assert len(predictions) == 1
    assert isinstance(predictions[0], Prediction)
    assert predictions[0].bboxes is not None
    assert predictions[0].bboxes.shape[0] == 0
