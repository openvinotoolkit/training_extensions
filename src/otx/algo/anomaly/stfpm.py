"""OTX STFPM model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from anomalib.models.image.stfpm import Stfpm as AnomalibStfpm

from otx.core.model.anomaly import OTXAnomaly
from otx.core.model.base import OTXModel
from otx.core.types.label import AnomalyLabelInfo
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torch.optim.optimizer import Optimizer

    from otx.core.model.anomaly import AnomalyModelInputs


class Stfpm(OTXAnomaly, OTXModel, AnomalibStfpm):
    """OTX STFPM model.

    Args:
        layers (Sequence[str]): Feature extractor layers.
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        task (Literal[
                OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION
            ], optional): Task type of Anomaly Task. Defaults to OTXTaskType.ANOMALY_CLASSIFICATION.
    """

    def __init__(
        self,
        layers: Sequence[str] = ["layer1", "layer2", "layer3"],
        backbone: str = "resnet18",
        task: Literal[
            OTXTaskType.ANOMALY_CLASSIFICATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_SEGMENTATION,
        ] = OTXTaskType.ANOMALY_CLASSIFICATION,
        **kwargs,
    ) -> None:
        OTXAnomaly.__init__(self)
        OTXModel.__init__(self, label_info=AnomalyLabelInfo())
        AnomalibStfpm.__init__(
            self,
            backbone=backbone,
            layers=layers,
        )
        self.task = task

    @property
    def trainable_model(self) -> str:
        """Used by configure optimizer."""
        return "student_model"

    def configure_metric(self) -> None:
        """This does not follow OTX metric configuration."""
        return

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Optimizer]] | None:
        """STFPM does not follow OTX optimizer configuration."""
        return AnomalibStfpm.configure_optimizers(self)

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        AnomalibStfpm.on_validation_epoch_start(self)

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        AnomalibStfpm.on_test_epoch_start(self)

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        AnomalibStfpm.on_validation_epoch_end(self)

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        AnomalibStfpm.on_test_epoch_end(self)

    def training_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call training step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return AnomalibStfpm.training_step(self, inputs, batch_idx)  # type: ignore[misc]

    def validation_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call validation step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return AnomalibStfpm.validation_step(self, inputs, batch_idx)  # type: ignore[misc]

    def test_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return AnomalibStfpm.test_step(self, inputs, batch_idx, **kwargs)  # type: ignore[misc]

    def predict_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return AnomalibStfpm.predict_step(self, inputs, batch_idx, **kwargs)  # type: ignore[misc]
