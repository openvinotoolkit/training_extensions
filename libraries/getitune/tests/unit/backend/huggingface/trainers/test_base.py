# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``GetiTuneHFTrainer``."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers as tf
from torch.utils.data import DataLoader, TensorDataset
from torchvision import tv_tensors
from transformers import TrainingArguments

from getitune.backend.huggingface.models import HFDetectionModel, HFMulticlassClsModel
from getitune.backend.huggingface.trainers.base import GetiTuneHFTrainer
from getitune.config.data import SubsetConfig
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import SampleBatch
from getitune.types import PathLike
from getitune.types.label import LabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(
        label_names=["cat", "dog", "bird"], label_ids=["0", "1", "2"], label_groups=[["cat", "dog", "bird"]]
    )


def _detection_model() -> HFDetectionModel:
    return HFDetectionModel(tf.RTDetrV2Config(num_queries=10, decoder_layers=2), _label_info())


def _multiclass_model() -> HFMulticlassClsModel:
    config = tf.ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    return HFMulticlassClsModel(config, _label_info())


def _mock_datamodule(*, with_gpu_augmentations: bool = False) -> MagicMock:
    datamodule = MagicMock()
    datamodule.train_subset = SubsetConfig(
        batch_size=2,
        num_workers=0,
        augmentations_gpu=(
            [{"class_path": "kornia.augmentation.RandomHorizontalFlip", "init_args": {"p": 1.0}}]
            if with_gpu_augmentations
            else []
        ),
    )
    datamodule.val_subset = SubsetConfig(batch_size=2, num_workers=0, augmentations_gpu=[])
    datamodule.test_subset = SubsetConfig(batch_size=2, num_workers=0, augmentations_gpu=[])
    return datamodule


def _detection_batch(device: torch.device | None = None) -> SampleBatch:
    device = device or torch.device("cpu")
    images = torch.rand(2, 3, 64, 64, dtype=torch.float32, device=device)
    bboxes = [
        tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.tensor([[10.0, 10.0, 30.0, 30.0]], device=device),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(64, 64),
        ),
        tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.zeros((0, 4), device=device), format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(64, 64)
        ),
    ]
    labels = [torch.tensor([1], device=device), torch.zeros(0, dtype=torch.long, device=device)]
    imgs_info = [
        ImageInfo(img_idx=0, img_shape=(64, 64), ori_shape=(64, 64)),  # pyrefly: ignore[no-matching-overload]
        ImageInfo(img_idx=1, img_shape=(64, 64), ori_shape=(64, 64)),  # pyrefly: ignore[no-matching-overload]
    ]
    return SampleBatch(images=images, bboxes=bboxes, labels=labels, imgs_info=imgs_info)


def _build_trainer(model_wrapper, datamodule, **args_kwargs) -> GetiTuneHFTrainer:
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            output_dir=tmp_dir,
            report_to=[],
            remove_unused_columns=False,
            label_names=list(model_wrapper.label_keys),
            use_cpu=True,
            **args_kwargs,
        )
        return GetiTuneHFTrainer(
            model_wrapper,
            datamodule,
            model=model_wrapper.hf_model,
            args=args,
            train_dataset=[0],
            eval_dataset=[0],
        )


def _collate_indexed_samples(samples: list[tuple[torch.Tensor]]) -> SampleBatch:
    indices = [int(sample[0]) for sample in samples]
    return SampleBatch(
        images=torch.tensor(indices, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 3, 2, 2),
        labels=[torch.full((index % 3 + 1,), index, dtype=torch.long) for index in indices],
    )


def _ddp_train_dataloader_worker(rank: int, init_file: str, result_dir: str) -> None:
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT="29571"
    )
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        datamodule = _mock_datamodule()
        datamodule.train_dataloader.return_value = DataLoader(
            TensorDataset(torch.arange(8)), batch_size=2, collate_fn=_collate_indexed_samples
        )
        trainer = _build_trainer(_multiclass_model(), datamodule)
        batches = [
            {"images": batch.images[:, 0, 0, 0].tolist(), "labels": [labels.tolist() for labels in batch.labels]}
            for batch in trainer.get_train_dataloader()
        ]
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(batches))

        def save_rank_marker(checkpoint: PathLike) -> None:
            checkpoint_dir = Path(checkpoint)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / f"rank-{rank}").touch()

        trainer.model_wrapper.save_pretrained = save_rank_marker
        trainer.save_model(str(Path(result_dir, "checkpoint")))
    finally:
        dist.destroy_process_group()


def test_train_dataloader_is_sharded_across_distributed_processes(tmp_path: Path) -> None:
    init_file = str(tmp_path / "distributed-init")
    result_dir = str(tmp_path)

    mp.spawn(_ddp_train_dataloader_worker, args=(init_file, result_dir), nprocs=2, join=True)

    rank_batches = [json.loads(Path(result_dir, f"rank-{rank}.json").read_text()) for rank in range(2)]
    rank_indices = [[index for batch in batches for index in batch["images"]] for batches in rank_batches]
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])
    assert sorted(rank_indices[0] + rank_indices[1]) == list(range(8))
    for batches in rank_batches:
        for batch in batches:
            for image_id, labels in zip(batch["images"], batch["labels"], strict=True):
                sample_index = int(image_id)
                assert labels == [sample_index] * (sample_index % 3 + 1)
    assert [path.name for path in (tmp_path / "checkpoint").iterdir()] == ["rank-0"]


class TestGpuPipelineConstruction:
    def test_no_pipeline_when_subset_has_no_gpu_augmentations(self) -> None:
        model = _multiclass_model()
        trainer = _build_trainer(model, _mock_datamodule(with_gpu_augmentations=False))
        assert trainer._train_gpu_pipeline is None
        assert trainer._eval_gpu_pipeline is None

    def test_pipeline_built_when_subset_has_gpu_augmentations(self) -> None:
        model = _multiclass_model()
        trainer = _build_trainer(model, _mock_datamodule(with_gpu_augmentations=True))
        assert trainer._train_gpu_pipeline is not None
        # val subset in the mock has no augmentations_gpu configured
        assert trainer._eval_gpu_pipeline is None


class TestMoveBatchToDevice:
    def test_moves_images_and_ragged_fields(self) -> None:
        batch = _detection_batch()
        moved = GetiTuneHFTrainer._move_batch_to_device(batch, torch.device("cpu"))
        assert isinstance(moved.images, torch.Tensor)
        assert moved.images.device.type == "cpu"
        assert moved.bboxes is not None
        assert moved.labels is not None
        assert all(b.device.type == "cpu" for b in moved.bboxes)
        assert all(t.device.type == "cpu" for t in moved.labels)

    def test_handles_none_optional_fields(self) -> None:
        batch = SampleBatch(images=torch.rand(1, 3, 8, 8))
        moved = GetiTuneHFTrainer._move_batch_to_device(batch, torch.device("cpu"))
        assert moved.bboxes is None
        assert moved.masks is None
        assert moved.keypoints is None


class TestApplyGpuAugmentation:
    def test_augments_images_and_rewraps_bboxes_as_tv_tensors(self) -> None:
        model = _detection_model()
        trainer = _build_trainer(model, _mock_datamodule(with_gpu_augmentations=True))
        batch = _detection_batch()
        pipeline = trainer._train_gpu_pipeline
        assert pipeline is not None

        augmented = trainer._apply_gpu_augmentation(pipeline, batch)

        assert isinstance(augmented.images, torch.Tensor)
        assert isinstance(batch.images, torch.Tensor)
        assert augmented.images.shape == batch.images.shape
        assert augmented.bboxes is not None
        assert all(isinstance(b, tv_tensors.BoundingBoxes) for b in augmented.bboxes)


class TestDictShapeOverrides:
    """SampleBatch isn't a Mapping; Trainer's base implementations assume one."""

    def test_get_num_items_in_batch_returns_none(self) -> None:
        model = _multiclass_model()
        trainer = _build_trainer(model, _mock_datamodule())
        assert trainer._get_num_items_in_batch([_detection_batch()], torch.device("cpu")) is None

    def test_prepare_inputs_is_a_no_op(self) -> None:
        model = _multiclass_model()
        trainer = _build_trainer(model, _mock_datamodule())
        batch = _detection_batch()
        assert trainer._prepare_inputs(batch) is batch

    def test_floating_point_ops_returns_zero(self) -> None:
        model = _multiclass_model()
        trainer = _build_trainer(model, _mock_datamodule())
        assert trainer.floating_point_ops(_detection_batch()) == 0


class TestComputeLoss:
    def test_computes_a_finite_loss_via_build_targets(self) -> None:
        model = _detection_model()
        trainer = _build_trainer(model, _mock_datamodule())
        batch = _detection_batch()

        loss = trainer.compute_loss(model.hf_model, batch)

        assert torch.isfinite(loss)

    def test_return_outputs_also_returns_the_raw_model_output(self) -> None:
        model = _detection_model()
        trainer = _build_trainer(model, _mock_datamodule())
        batch = _detection_batch()

        loss, outputs = trainer.compute_loss(model.hf_model, batch, return_outputs=True)

        assert torch.isfinite(loss)
        assert outputs.loss is loss

    def test_applies_gpu_augmentation_during_training(self) -> None:
        """model.training selects the train pipeline; augmenting must not break the forward pass."""
        model = _detection_model()
        trainer = _build_trainer(model, _mock_datamodule(with_gpu_augmentations=True))
        model.hf_model.train()

        loss = trainer.compute_loss(model.hf_model, _detection_batch())

        assert torch.isfinite(loss)

    def test_prediction_step_returns_loss_only(self) -> None:
        model = _detection_model()
        trainer = _build_trainer(model, _mock_datamodule())
        model.hf_model.eval()

        loss, logits, labels = trainer.prediction_step(model.hf_model, _detection_batch(), prediction_loss_only=True)

        assert loss is not None
        assert torch.isfinite(loss)
        assert logits is None
        assert labels is None


class TestEvaluateOnTestSplitAndPredictBatches:
    """Mocked tests for evaluate(split='test') and predict_batches."""

    def _build_trainer(self, tmp_path: Path) -> GetiTuneHFTrainer:
        model = _detection_model()
        datamodule = _mock_datamodule()
        args = TrainingArguments(
            output_dir=str(tmp_path / "train"),
            report_to=[],
            remove_unused_columns=False,
            label_names=list(model.label_keys),
            use_cpu=True,
        )
        trainer = GetiTuneHFTrainer(
            model,
            datamodule,
            model=model.hf_model,
            args=args,
            train_dataset=[0],
            eval_dataset=[0],
        )
        # mock the test dataloader to return one synthetic batch
        batch = _detection_batch()
        from getitune.data.entity.sample import PredictionBatch

        trainer.get_test_dataloader = MagicMock(return_value=[batch])
        model.to_metric_inputs = MagicMock(return_value={})  # type: ignore[method-assign]
        model.postprocess = MagicMock(return_value=PredictionBatch(images=batch.images))  # type: ignore[method-assign]
        return trainer

    def test_evaluate_on_test_split_returns_prefixed_metrics(self, tmp_path: Path) -> None:
        trainer = self._build_trainer(tmp_path)
        fake_metric = _FakeMetric(0.42)

        result = trainer.evaluate(split="test", metric=fake_metric)  # pyrefly: ignore[bad-argument-type]

        assert "test/map" in result
        assert result["test/map"] == pytest.approx(0.42)
        assert fake_metric.update_calls == 1

    def test_predict_batches_returns_one_batch_per_dataloader_batch(self, tmp_path: Path) -> None:
        trainer = self._build_trainer(tmp_path)

        batches = trainer.predict_batches()

        assert len(batches) == 1  # one batch from mocked dataloader


class _FakeMetric:
    """Stand-in torchmetrics-like object for exercising GetiTuneHFTrainer.evaluate()."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.update_calls = 0

    def reset(self) -> None:
        self.update_calls = 0

    def to(self, device: torch.device) -> _FakeMetric:
        return self

    def update(self, **_kwargs: object) -> None:
        self.update_calls += 1

    def compute(self) -> dict[str, torch.Tensor]:
        return {"map": torch.tensor(self.value), "map_50": torch.tensor(self.value + 0.1)}


class TestTrainerLoopExercisesOverrides:
    """Mocked tests verifying SampleBatch-compatible overrides wire correctly.

    These overrides exist because ``transformers.Trainer`` assumes a dict-like
    batch everywhere (``labels in batch``, ``batch["labels"]`` etc.). We verify
    each override in isolation with a synthetic batch, without running the full
    Trainer training loop.
    """

    def _trainer(self, tmp_path: Path) -> GetiTuneHFTrainer:
        model = _detection_model()
        datamodule = _mock_datamodule()
        args = TrainingArguments(
            output_dir=str(tmp_path / "train"),
            report_to=[],
            remove_unused_columns=False,
            label_names=list(model.label_keys),
            use_cpu=True,
        )
        return GetiTuneHFTrainer(
            model,
            datamodule,
            model=model.hf_model,
            args=args,
            train_dataset=[0],
            eval_dataset=[0],
        )

    def test_prepare_inputs_is_a_no_op(self, tmp_path: Path) -> None:
        trainer = self._trainer(tmp_path)
        batch = _detection_batch()
        assert trainer._prepare_inputs(batch) is batch

    def test_get_num_items_in_batch_returns_none(self, tmp_path: Path) -> None:
        trainer = self._trainer(tmp_path)
        assert trainer._get_num_items_in_batch([_detection_batch()], torch.device("cpu")) is None

    def test_floating_point_ops_returns_zero(self, tmp_path: Path) -> None:
        trainer = self._trainer(tmp_path)
        assert trainer.floating_point_ops(_detection_batch()) == 0

    def test_training_log_includes_timing(self, tmp_path: Path) -> None:
        trainer = self._trainer(tmp_path)
        trainer._train_data_time = 0.1
        trainer._train_iter_time = 0.4

        trainer.log({"loss": 0.5})

        entry = trainer.state.log_history[-1]
        assert entry["train/data_time"] == pytest.approx(0.1)
        assert entry["train/iter_time"] == pytest.approx(0.4)

    def test_evaluate_logs_val_map_keys(self, tmp_path: Path) -> None:
        """evaluate() logs val/* keys when FakeMetric is configured."""
        model = _detection_model()
        datamodule = _mock_datamodule()
        args = TrainingArguments(
            output_dir=str(tmp_path / "train"),
            report_to=[],
            remove_unused_columns=False,
            label_names=list(model.label_keys),
            use_cpu=True,
        )
        trainer = GetiTuneHFTrainer(
            model,
            datamodule,
            metric=lambda _li: _FakeMetric(0.5),  # pyrefly: ignore[bad-argument-type]
            model=model.hf_model,
            args=args,
            train_dataset=[0],
            eval_dataset=[0],
        )
        batch = _detection_batch()
        trainer.get_eval_dataloader = MagicMock(return_value=[batch])
        model.to_metric_inputs = MagicMock(return_value={})  # type: ignore[method-assign]

        metrics = trainer.evaluate()

        assert "val/map" in metrics
        assert "validation/data_time" in metrics
        assert "validation/iter_time" in metrics
        assert any("val/map" in e for e in trainer.state.log_history)


class TestEvaluateOverride:
    """Unit tests for GetiTuneHFTrainer.evaluate() metric + best-checkpoint behavior."""

    def _build_trainer(
        self, tmp_path: Path, metric_value: float = 0.5, val_check_interval: int = 1
    ) -> GetiTuneHFTrainer:
        model = _detection_model()
        datamodule = _mock_datamodule()
        args = TrainingArguments(
            output_dir=str(tmp_path / "train"),
            report_to=[],
            remove_unused_columns=False,
            label_names=list(model.label_keys),
            use_cpu=True,
            metric_for_best_model="val/map",
            greater_is_better=True,
        )
        trainer = GetiTuneHFTrainer(
            model,
            datamodule,
            metric=lambda _label_info: _FakeMetric(metric_value),  # pyrefly: ignore[bad-argument-type]
            val_check_interval=val_check_interval,
            model=model.hf_model,
            args=args,
            train_dataset=[0],
            eval_dataset=[0],
        )
        trainer.get_eval_dataloader = MagicMock(return_value=[_detection_batch()])
        model.to_metric_inputs = MagicMock(return_value={})  # type: ignore[method-assign]
        return trainer

    def test_evaluate_logs_prefixed_scalar_metrics(self, tmp_path: Path) -> None:
        trainer = self._build_trainer(tmp_path, metric_value=0.42)

        metrics = trainer.evaluate()

        assert "val/map" in metrics
        assert "val/map_50" in metrics
        assert metrics["val/map"] == pytest.approx(0.42)
        assert any("val/map" in entry for entry in trainer.state.log_history)

    def test_evaluate_saves_best_checkpoint_only_on_improvement(self, tmp_path: Path) -> None:
        trainer = self._build_trainer(tmp_path, metric_value=0.5)
        save_model = MagicMock()
        trainer.save_model = save_model

        trainer.evaluate()
        assert save_model.call_count == 1

        # Worse metric: no new save
        trainer._val_metric.value = 0.4  # pyrefly: ignore[missing-attribute, bad-argument-type]
        trainer.evaluate()
        assert save_model.call_count == 1

        # Better metric: save again
        trainer._val_metric.value = 0.6  # pyrefly: ignore[missing-attribute, bad-argument-type]
        trainer.evaluate()
        assert save_model.call_count == 2

    def test_evaluate_skips_non_eval_epochs(self, tmp_path: Path) -> None:
        trainer = self._build_trainer(tmp_path, val_check_interval=2)
        trainer.state.epoch = 1.0

        metrics = trainer.evaluate()

        assert metrics == {}
        assert trainer._val_metric.update_calls == 0  # pyrefly: ignore[missing-attribute]

    def test_evaluate_runs_on_eval_epochs_when_interval_is_two(self, tmp_path: Path) -> None:
        trainer = self._build_trainer(tmp_path, val_check_interval=2)
        trainer.state.epoch = 2.0

        metrics = trainer.evaluate()

        assert "val/map" in metrics
        assert trainer._val_metric.update_calls == 1  # pyrefly: ignore[missing-attribute]


class TestPlateauWarmup:
    """The trainer ramps LR itself for plateau schedules (transformers ignores warmup_steps)."""

    def _trainer(self, tmp_path: Path, scheduler_type: str, warmup_steps: int) -> GetiTuneHFTrainer:
        model = _detection_model()
        args = TrainingArguments(
            output_dir=str(tmp_path / "train"),
            report_to=[],
            remove_unused_columns=False,
            label_names=list(model.label_keys),
            use_cpu=True,
            learning_rate=1e-4,
            lr_scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            eval_strategy="epoch" if scheduler_type == "reduce_lr_on_plateau" else "no",
            metric_for_best_model="val/map" if scheduler_type == "reduce_lr_on_plateau" else None,
        )
        return GetiTuneHFTrainer(
            model, _mock_datamodule(), model=model.hf_model, args=args, train_dataset=[0], eval_dataset=[0]
        )

    def test_create_optimizer_arms_warmup_only_for_plateau(self, tmp_path: Path) -> None:
        trainer = self._trainer(tmp_path, "reduce_lr_on_plateau", warmup_steps=4)
        optimizer = trainer.create_optimizer()
        assert trainer._plateau_warmup_steps == 4
        assert trainer._plateau_warmup_base_lr == pytest.approx(1e-4)

        cosine = self._trainer(tmp_path, "cosine", warmup_steps=4)
        cosine.create_optimizer()
        assert cosine._plateau_warmup_steps == 0
        assert optimizer is not None

    def test_apply_plateau_warmup_ramps_then_releases(self, tmp_path: Path) -> None:
        trainer = self._trainer(tmp_path, "reduce_lr_on_plateau", warmup_steps=4)
        trainer.create_optimizer()
        for group in trainer.optimizer.param_groups:  # pyrefly: ignore[missing-attribute]
            group["lr"] = 1e-4

        trainer.state.global_step = 0
        trainer._apply_plateau_warmup()
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 / 4)  # pyrefly: ignore[missing-attribute]

        trainer.state.global_step = 3
        trainer._apply_plateau_warmup()
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)  # pyrefly: ignore[missing-attribute]

        # First step past warmup disarms the manual ramp without re-writing LR:
        # ReduceLROnPlateau may already have reduced the LR during the first
        # post-warmup evaluation; rewriting base_lr would undo that drop.
        trainer.state.global_step = 5
        trainer.optimizer.param_groups[0]["lr"] = 1e-6  # pyrefly: ignore[missing-attribute]
        trainer._apply_plateau_warmup()
        assert trainer._plateau_warmup_steps == 0
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)  # pyrefly: ignore[missing-attribute]
