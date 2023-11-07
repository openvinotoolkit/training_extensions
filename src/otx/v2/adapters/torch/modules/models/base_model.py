"""Abstract Class of OTX Base Model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path



class BaseOTXModel(ABC):
    """Abstract base class for OTX models.

    This class defines the interface for OTX models, including the callbacks to be used during training
    and the ability to export the model to a specified format (ONNX & OPENVINO).

    Attributes:
        callbacks (list[Callback]): A list of callbacks to be used during training.
    """

    @abstractproperty
    def callbacks(self) -> list:
        """Returns a list of callbacks to be used during training.

        Returns:
            list: A list of callbacks to be used during training.
        """

    @abstractmethod
    def export(
        self,
        export_dir: str | Path,
        export_type: str = "OPENVINO",
        precision: str | int | None = None,
    ) -> dict:
        """Export the model to a specified format.

        Args:
            export_dir (str | Path): The directory to export the model to.
            export_type (str, optional): The type of export to perform. Defaults to "OPENVINO".
            precision (str | int | None, optional): The precision to use for the export. Defaults to None.

        Returns:
            dict: A dictionary containing information about the exported model.
        """
