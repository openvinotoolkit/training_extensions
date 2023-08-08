"""Interface for inferencer."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from typing import Any, Tuple, Union

from openvino.model_api.models import Model

import numpy as np

from otx.api.entities.annotation import AnnotationSceneEntity

__all__ = [
    "IInferencer",
]


class IInferencer(metaclass=abc.ABCMeta):
    """Base interface class for the inference task.

    This class could be used by both the analyse method in the task, and the exportable code inference.

    """

    model: Model

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> Union[AnnotationSceneEntity, Tuple[Any, ...]]:
        """This method performs a prediction."""
        raise NotImplementedError
