"""OTX Padim model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anomalib.models.image import Padim as AnomalibPadim

from otx.core.model.anomaly import OTXAnomaly
from otx.core.model.base import OTXModel
from otx.core.types.label import AnomalyLabelInfo
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torch.optim.optimizer import Optimizer

    from otx.core.model.anomaly import AnomalyModelInputs


class Padim(OTXAnomaly, OTXModel, AnomalibPadim):
    """OTX Padim model.

    Args:
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        layers (list[str], optional): Feature extractor layers. Defaults to ["layer1", "layer2", "layer3"].
        pre_trained (bool, optional): Pretrained backbone. Defaults to True.
        n_features (int | None, optional): Number of features. Defaults to None.
        task (Literal[
                OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION
            ], optional): Task type of Anomaly Task. Defaults to OTXTaskType.ANOMALY_CLASSIFICATION.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
        task: Literal[
            OTXTaskType.ANOMALY_CLASSIFICATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_SEGMENTATION,
        ] = OTXTaskType.ANOMALY_CLASSIFICATION,
    ) -> None:
        OTXAnomaly.__init__(self)
        OTXModel.__init__(self, label_info=AnomalyLabelInfo())
        AnomalibPadim.__init__(
            self,
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
        )
        self.task = task

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
