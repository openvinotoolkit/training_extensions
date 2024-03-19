"""OTX Draem model."""
# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from anomalib.models.image import Draem as AnomalibDraem

from otx.core.model.anomaly import OTXAnomaly
from otx.core.model.base import OTXModel

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torch.optim.optimizer import Optimizer

    from otx.core.model.anomaly import AnomalyModelInputs


class Draem(OTXAnomaly, OTXModel, AnomalibDraem):
    """OTX Draem model.

    Args:
        enable_sspcab (bool): Enable SSPCAB training. Defaults to ``False``.
        sspcab_lambda (float): SSPCAB loss weight. Defaults to ``0.1``.
        anomaly_source_path (str | None): Path to folder that contains the anomaly source images. Random noise will
            be used if left empty. Defaults to ``None``.
        beta (float | tuple[float, float]): Parameter that determines the opacity of the noise mask.
            Defaults to ``(0.1, 1.0)``.
    """

    def __init__(
        self,
        enable_sspcab: bool = False,
        sspcab_lambda: float = 0.1,
        anomaly_source_path: str | None = None,
        beta: float | tuple[float, float] = (0.1, 1.0),
        num_classes: int = 2,
    ) -> None:
        OTXAnomaly.__init__(self)
        OTXModel.__init__(self, num_classes=num_classes)
        AnomalibDraem.__init__(
            self,
            enable_sspcab=enable_sspcab,
            sspcab_lambda=sspcab_lambda,
            anomaly_source_path=anomaly_source_path,
            beta=beta,
        )

    def configure_metric(self) -> None:
        """This does not follow OTX metric configuration."""
        return

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Optimizer]] | None:
        """STFPM does not follow OTX optimizer configuration."""
        return Draem.configure_optimizers(self)

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        Draem.on_validation_epoch_start(self)

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        Draem.on_test_epoch_start(self)

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        Draem.on_validation_epoch_end(self)

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        Draem.on_test_epoch_end(self)

    def training_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call training step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return Draem.training_step(self, inputs, batch_idx)  # type: ignore[misc]

    def validation_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Call validation step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return Draem.validation_step(self, inputs, batch_idx)  # type: ignore[misc]

    def test_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return Draem.test_step(self, inputs, batch_idx, **kwargs)  # type: ignore[misc]

    def predict_step(
        self,
        inputs: AnomalyModelInputs,
        batch_idx: int = 0,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Call test step of the anomalib model."""
        if not isinstance(inputs, dict):
            inputs = self._customize_inputs(inputs)
        return Draem.predict_step(self, inputs, batch_idx, **kwargs)  # type: ignore[misc]
