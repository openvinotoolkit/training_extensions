# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The ``transformers.Trainer`` bridge for the Hugging Face backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torchvision import tv_tensors
from transformers import Trainer

from getitune.data.augmentation import GPUAugmentationPipeline
from getitune.data.augmentation.task_keys import DATA_KEYS_BY_TASK
from getitune.metrics import MetricCallable

from .utils import resolve_greater_is_better

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.utils.data import DataLoader
    from torchmetrics import Metric, MetricCollection

    from getitune.backend.huggingface.models.base import HFModel
    from getitune.config.data import SubsetConfig
    from getitune.data.entity.sample import PredictionBatch, SampleBatch
    from getitune.data.module import DataModule

__all__ = ["GetiTuneHFTrainer"]


class GetiTuneHFTrainer(Trainer):
    """Binds ``transformers.Trainer`` to a Geti ``DataModule`` and ``HFModel``.

    Args:
        model_wrapper: The ``HFModel`` being trained. Supplies
            ``build_targets`` and the task used to pick the GPU augmentation
            pipeline's ``data_keys``.
        datamodule: The Geti ``DataModule`` supplying the train/eval splits.
        metric: Optional metric callable that receives the model's
            ``label_info`` and returns a ``torchmetrics.Metric`` (or
            ``MetricCollection``). If ``None``, ``model_wrapper``'s default
            task metric is used.
        val_check_interval: Evaluate and checkpoint on every N-th epoch.
            ``1`` means every epoch; ``None`` defaults to ``1``.
        *args: Forwarded to ``transformers.Trainer`` (typically ``model`` and
            ``args``).
        **kwargs: Forwarded to ``transformers.Trainer``.
    """

    def __init__(
        self,
        model_wrapper: HFModel,
        datamodule: DataModule,
        metric: MetricCallable | None = None,
        val_check_interval: int | None = None,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        self.model_wrapper = model_wrapper
        self.datamodule = datamodule
        self._train_gpu_pipeline = self._build_gpu_pipeline(datamodule.train_subset, sanitize=True)
        self._eval_gpu_pipeline = self._build_gpu_pipeline(datamodule.val_subset, sanitize=False)
        self._test_gpu_pipeline = self._build_gpu_pipeline(datamodule.test_subset, sanitize=False)
        self._val_check_interval = max(1, val_check_interval or 1)
        self._val_metric: Metric | MetricCollection | None = (
            metric(model_wrapper.label_info) if metric is not None else model_wrapper.build_default_metric()
        )
        self._best_metric_value: float | None = None
        self._best_eval_metrics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _build_gpu_pipeline(
        self, subset_config: SubsetConfig | None, *, sanitize: bool
    ) -> GPUAugmentationPipeline | None:
        """Build the GPU augmentation pipeline for one subset, or ``None`` if unconfigured.

        Skipped entirely when the subset has no ``augmentations_gpu`` — most
        commonly meaning normalization happens on the CPU stage instead, in
        which case there is nothing for this pipeline to do.
        """
        if subset_config is None or not subset_config.augmentations_gpu:
            return None
        data_keys = ["input", *DATA_KEYS_BY_TASK.get(self.model_wrapper.task, ())]
        return GPUAugmentationPipeline.from_config(subset_config, data_keys=data_keys, sanitize_annotations=sanitize)

    def get_train_dataloader(self) -> DataLoader:
        """Return the DataModule's training dataloader."""
        return self.datamodule.train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Any = None) -> DataLoader:  # noqa: ANN401
        """Return the DataModule's validation dataloader."""
        return self.datamodule.val_dataloader()

    def get_test_dataloader(self, test_dataset: Any = None) -> DataLoader:  # noqa: ANN401
        """Return the DataModule's test dataloader."""
        return self.datamodule.test_dataloader()

    def _get_num_items_in_batch(self, batch_samples: list[Any], device: torch.device) -> int | None:
        """Disable per-batch item counting for gradient-accumulation loss scaling.

        ``Trainer.training_step`` calls this to learn how many examples are in a
        batch so it can rescale the loss under gradient accumulation. The base
        implementation assumes a dict-like batch: it tests ``"labels" in
        batch_samples[0]`` and then indexes ``batch_samples[0]["labels"]``. A
        ``SampleBatch`` dataclass is neither subscriptable nor a ``Mapping``, so
        those operators don't exist on it. Returning ``None`` tells ``Trainer``
        not to rescale — which is correct here, because each of our vision
        models already returns a batch-mean loss, not a per-token loss that
        would need normalizing the way a causal LM's does.
        """
        return None

    def _prepare_inputs(self, inputs: SampleBatch) -> SampleBatch:  # pyrefly: ignore[bad-override]
        """Pass ``SampleBatch`` through unchanged.

        ``Trainer.training_step`` calls ``self._prepare_inputs(inputs)`` before
        the forward pass. The base version only knows how to recurse into a
        ``Mapping``/``list``/``tuple`` and otherwise treats the argument as a
        single tensor to move to the target device; a ``SampleBatch`` dataclass
        matches none of those branches, so the base would either no-op or try to
        move the whole dataclass as a tensor. The real device transfer (and any
        GPU augmentation) happens explicitly in ``compute_loss``/``_prepare_batch``,
        so this override is a deliberate identity function.
        """
        return inputs

    def floating_point_ops(self, inputs: SampleBatch) -> int:  # pyrefly: ignore[bad-override]
        """Skip FLOPs accounting (returns 0).

        ``Trainer.training_step`` adds ``self.floating_point_ops(inputs)`` to the
        running ``total_flos`` it logs. The base implementation estimates FLOPs
        by indexing the batch with ``self.main_input_name`` (``"input_ids"`` by
        default) and reading ``inputs[main_input_name].shape`` — again a
        dict-style access that doesn't apply to a ``SampleBatch``. We return 0:
        a meaningful per-call FLOP count for arbitrary Hugging Face vision models
        would require model-specific analysis, and no Geti consumer reads
        ``total_flos``, so the effort isn't warranted.
        """
        return 0

    def prediction_step(  # pyrefly: ignore[bad-override]
        self,
        model: torch.nn.Module,
        inputs: SampleBatch,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, None, None]:
        """Compute the evaluation loss; return ``(loss, None, None)``.

        ``Trainer.predict()`` calls this once per batch. The base implementation
        expects a dict-like batch: it inspects ``inputs`` with ``inputs.get(k)``
        to decide whether labels are present, then builds
        ``(loss, logits, labels)`` for ``Trainer``'s ``compute_metrics``. Two
        reasons we don't do that here:

        * ``SampleBatch`` is a dataclass, not a ``Mapping``, so the base's
          dict probing would fail; and
        * Geti's own metric callables run through ``HFEngine.test()`` /
          ``GetiTuneHFTrainer.evaluate(split="test")``, not through
          ``Trainer.compute_metrics``.

        Only the loss is returned because nothing downstream consumes logits or
        labels here. ``GetiTuneHFTrainer.evaluate()`` bypasses this method
        entirely and computes real task metrics on the validation split.
        """
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss, None, None

    def compute_loss(  # pyrefly: ignore[bad-override]
        self,
        model: torch.nn.Module,
        inputs: SampleBatch,
        return_outputs: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Move the batch to the model's device, augment, convert, and forward.

        *inputs* is the raw ``SampleBatch`` our own collate function
        returns — ``Trainer._prepare_inputs`` only recurses into
        ``Mapping``/``list``/``tuple``/``Tensor``, so it leaves a dataclass
        like ``SampleBatch`` untouched. The device move that would normally
        happen there happens explicitly here instead.
        """
        pipeline = self._train_gpu_pipeline if model.training else self._eval_gpu_pipeline
        batch = self._prepare_batch(inputs, pipeline)
        targets = self.model_wrapper.build_targets(batch)
        outputs = model(**targets)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _determine_best_metric(  # pyrefly: ignore[bad-override]
        self,
        metrics: dict[str, float],
        trial: Any,  # noqa: ANN401
    ) -> bool:
        """Look up ``metric_for_best_model``.

        ``Trainer._determine_best_metric`` unconditionally prepends ``eval_``
        to ``metric_for_best_model`` if it doesn't already start with that
        prefix. Our ``evaluate()`` logs directly as ``val/map``, ``val/Dice``
        etc., so we skip that mangling and look up the key as-is.
        """
        monitor = self.args.metric_for_best_model
        if not monitor or monitor not in metrics:
            return False

        metric_value = metrics[monitor]
        operator = np.greater if self.args.greater_is_better else np.less

        if self.state.best_metric is None:
            self.state.best_metric = float("-inf") if self.args.greater_is_better else float("inf")

        if operator(metric_value, self.state.best_metric):
            self.state.best_metric = metric_value
            return True
        return False

    def evaluate(  # pyrefly: ignore[bad-override]
        self,
        eval_dataset: Any = None,  # noqa: ANN401
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
        *,
        split: str = "val",
        metric: Metric | MetricCollection | None = None,
    ) -> dict[str, float]:
        """Run a Geti metric on *split* and return scalar ``val/`` or ``test/`` metrics.

        This replaces ``transformers.Trainer.evaluate()`` during training so
        that validation reports real task metrics (e.g. ``val/map``,
        ``val/f1-score``, ``val/Dice``) instead of ``eval_loss``. It is also
        used for the test split.

        Args:
            eval_dataset: Ignored; the dataloader is taken from
                ``self.datamodule``.
            ignore_keys: Ignored; kept for signature compatibility with the
                base ``Trainer.evaluate()``.
            metric_key_prefix: Ignored; kept for signature compatibility.
            split: Either ``"val"`` or ``"test"``. Determines which dataloader
                is used and the metric key prefix.
            metric: Optional metric or metric collection to use. If ``None``,
                the metric passed to ``__init__`` (the task's default or an
                override) is used.

        Returns:
            A dictionary of scalar metrics prefixed with ``val/`` or ``test/``.
        """
        if split not in ("val", "test"):
            msg = f"split must be 'val' or 'test', got {split!r}"
            raise ValueError(msg)

        metric_obj = metric if metric is not None else self._val_metric
        if metric_obj is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if split == "val":
            dataloader = self.get_eval_dataloader(eval_dataset)
        else:
            dataloader = self.get_test_dataloader(eval_dataset)

        epoch = int(getattr(self.state, "epoch", 0))
        if split == "val" and epoch % self._val_check_interval != 0:
            return {}

        self.model_wrapper.ensure_predict_ready()
        metric_obj.reset()

        pipeline = self._eval_gpu_pipeline if split == "val" else self._test_gpu_pipeline
        model = self.model
        if model is None:
            msg = "Trainer model is not set; cannot run evaluation."
            raise RuntimeError(msg)
        with torch.no_grad():
            for inputs in dataloader:
                batch = self._prepare_batch(inputs, pipeline)
                outputs = model(**self.model_wrapper.build_eval_inputs(batch))
                metric_inputs = self.model_wrapper.to_metric_inputs(outputs, batch)
                # HF postprocess returns CPU tensors while the batch lives on the
                # accelerator, so move every tensor in the metric inputs to CPU once
                # here instead of scattering .cpu() through each to_metric_inputs.
                metric_inputs = apply_to_collection(metric_inputs, torch.Tensor, lambda t: t.cpu())
                metric_obj.update(**metric_inputs)

        computed = metric_obj.compute()
        metrics = self._format_metrics(computed, f"{split}/")
        if not metrics:
            return {}

        if split == "val":
            metrics["epoch"] = epoch
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self._maybe_save_best_checkpoint(metrics)
            model.train()

        return metrics

    @staticmethod
    def _format_metrics(computed: dict[str, Any], prefix: str) -> dict[str, float]:
        """Flatten a metric result, keep only scalar values, and prefix keys."""
        metrics: dict[str, float] = {}
        for key, value in computed.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    metrics[f"{prefix}{key}"] = value.item()
            elif isinstance(value, (int, float)):
                metrics[f"{prefix}{key}"] = float(value)
        return metrics

    def _maybe_save_best_checkpoint(self, metrics: dict[str, float]) -> None:
        """Save ``best_checkpoint/`` if ``metric_for_best_model`` improved."""
        monitor = self.args.metric_for_best_model
        if not monitor:
            return

        current = metrics.get(monitor)
        if current is None or not isinstance(current, (int, float)):
            return

        greater_is_better = self.args.greater_is_better
        if greater_is_better is None:
            greater_is_better = resolve_greater_is_better(monitor)

        best = self._best_metric_value
        if best is None:
            improved = True
        elif greater_is_better:
            improved = current > best
        else:
            improved = current < best

        if improved:
            self._best_metric_value = float(current)
            self._best_eval_metrics = metrics.copy()
            output_dir = self.args.output_dir
            if output_dir is None:
                msg = "TrainingArguments.output_dir is not set; cannot save best checkpoint."
                raise RuntimeError(msg)
            best_dir = Path(output_dir).parent / "best_checkpoint"
            best_dir.mkdir(parents=True, exist_ok=True)
            self.save_model(str(best_dir))

    def _prepare_batch(self, batch: SampleBatch, pipeline: GPUAugmentationPipeline | None) -> SampleBatch:
        """Move a batch to the model's device and apply GPU augmentation, if configured."""
        batch = self._move_batch_to_device(batch, self.args.device)
        if pipeline is not None:
            batch = self._apply_gpu_augmentation(pipeline, batch)
        return batch

    def predict_batches(self) -> list[PredictionBatch]:
        """Run the test split through the model and postprocess every batch.

        This iterates the test dataloader directly instead of going through
        ``Trainer.predict()``, whose ``EvalLoopOutput``/``compute_metrics``
        contract assumes fixed-shape per-sample outputs. Confidence
        thresholding is deliberately not applied here — it belongs to
        whichever caller decides what to do with these predictions
        (``HFEngine.predict()``).

        Returns:
            One :class:`~getitune.data.entity.sample.PredictionBatch` per
            batch in the test split, in dataloader order.
        """
        model = self.model_wrapper.hf_model
        self.model_wrapper.ensure_predict_ready()
        batches: list[PredictionBatch] = []
        with torch.no_grad():
            for inputs in self.get_test_dataloader():
                batch = self._prepare_batch(inputs, self._test_gpu_pipeline)
                outputs = model(**self.model_wrapper.build_eval_inputs(batch))
                batches.append(self.model_wrapper.postprocess(outputs, batch))
        return batches

    @staticmethod
    def _move_batch_to_device(batch: SampleBatch, device: torch.device) -> SampleBatch:
        if not isinstance(batch.images, torch.Tensor):
            msg = f"Expected a stacked image tensor, got {type(batch.images)}"
            raise TypeError(msg)

        def move(items: Sequence[torch.Tensor] | None) -> list[torch.Tensor] | None:
            return None if items is None else [item.to(device) for item in items]

        return batch.wrap(
            images=batch.images.to(device),
            labels=move(batch.labels),
            bboxes=move(batch.bboxes),
            masks=move(batch.masks),
            keypoints=move(batch.keypoints),
        )

    def _apply_gpu_augmentation(self, pipeline: GPUAugmentationPipeline, batch: SampleBatch) -> SampleBatch:
        """Run the Kornia pipeline and re-wrap its output as Geti tv_tensors.

        A trimmed version of ``GPUAugmentationCallback._apply_pipeline``: the
        same re-wrapping logic, minus tiling and keypoints, since neither
        applies to the five tasks this backend supports.
        """
        result = pipeline(batch.images, labels=batch.labels, bboxes=batch.bboxes, masks=batch.masks)

        bboxes = batch.bboxes
        if result.get("bboxes") is not None and batch.bboxes is not None:
            bboxes = [
                box
                if isinstance(box, tv_tensors.BoundingBoxes)
                else tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    box, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=batch.bboxes[i].canvas_size
                )
                for i, box in enumerate(result["bboxes"])
            ]

        masks = batch.masks
        if result.get("masks") is not None:
            masks = [mask if isinstance(mask, tv_tensors.Mask) else tv_tensors.Mask(mask) for mask in result["masks"]]

        return batch.wrap(
            images=result["images"],
            labels=result["labels"] if result.get("labels") is not None else batch.labels,
            bboxes=bboxes,
            masks=masks,
        )
