"""OTX Padim model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anomalib.callbacks.normalization.min_max_normalization import _MinMaxNormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.models.image import Padim as AnomalibPadim

from otx.core.model.anomaly import OTXAnomaly
from otx.core.types.label import AnomalyLabelInfo
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torch.optim.optimizer import Optimizer

    from otx.core.model.anomaly import AnomalyModelInputs, AnomalyModelOutputs
    from otx.core.types.label import LabelInfoTypes


class Padim(OTXAnomaly, AnomalibPadim):
    """OTX Padim model.

    Args:
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        layers (list[str], optional): Feature extractor layers. Defaults to ["layer1", "layer2", "layer3"].
        pre_trained (bool, optional): Pretrained backbone. Defaults to True.
        n_features (int | None, optional): Number of features. Defaults to None.
        task (Literal[
                OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION
            ], optional): Task type of Anomaly Task. Defaults to OTXTaskType.ANOMALY_CLASSIFICATION.
        input_size (tuple[int, int], optional):
            Model input size in the order of height and width. Defaults to (256, 256)
    """

    def __init__(
        self,
        label_info: LabelInfoTypes = AnomalyLabelInfo(),
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
        task: Literal[
            OTXTaskType.ANOMALY_CLASSIFICATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_SEGMENTATION,
        ] = OTXTaskType.ANOMALY_CLASSIFICATION,
        input_size: tuple[int, int] = (256, 256),
    ) -> None:
        OTXAnomaly.__init__(self, label_info=label_info, input_size=input_size)
        AnomalibPadim.__init__(
            self,
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
        )
        self.task = task
        self.input_size = input_size

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Optimizer]] | None:
        """PADIM doesn't require optimization, therefore returns no optimizers."""
        return

    def configure_metric(self) -> None:
        """This does not follow OTX metric configuration."""
        return

    def on_train_epoch_end(self) -> None:
        """Callback triggered when the training epoch ends."""
        return AnomalibPadim.on_train_epoch_end(self)

    def on_validation_start(self) -> None:
        """Callback triggered when the validation starts."""
        return AnomalibPadim.on_validation_start(self)

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        AnomalibPadim.on_validation_epoch_start(self)

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        AnomalibPadim.on_test_epoch_start(self)

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        AnomalibPadim.on_validation_epoch_end(self)

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        AnomalibPadim.on_test_epoch_end(self)

    def training_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call training step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return AnomalibPadim.training_step(self, inputs, batch_idx)  # type: ignore[misc]

    def validation_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call validation step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return AnomalibPadim.validation_step(self, inputs, batch_idx)  # type: ignore[misc]

    def test_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return AnomalibPadim.test_step(self, inputs, batch_idx, **kwargs)  # type: ignore[misc]

    def predict_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return AnomalibPadim.predict_step(self, inputs, batch_idx, **kwargs)  # type: ignore[misc]

    def forward(
        self,
        inputs: AnomalyModelInputs,
    ) -> AnomalyModelOutputs:
        """Wrap forward method of the Anomalib model."""
        outputs = self.validation_step(inputs)
        # TODO(Ashwin): update forward implementation to comply with other OTX models
        _PostProcessorCallback._post_process(outputs)  # noqa: SLF001
        _PostProcessorCallback._compute_scores_and_labels(self, outputs)  # noqa: SLF001
        _MinMaxNormalizationCallback._normalize_batch(outputs, self)  # noqa: SLF001

        return self._customize_outputs(outputs=outputs, inputs=inputs)
