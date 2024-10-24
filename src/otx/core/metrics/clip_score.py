# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLIP Score Metric for CLIP fine-tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from otx.core.metrics.types import MetricCallable

if TYPE_CHECKING:
    from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

    from otx.core.types.label import LabelInfo


# Copy from torchmetrics.functional.multimodal.clip_score.py
# Modified to be more generally available for fine-tuning models.
# Reduced unnecessary feature pulls and modified to count with features already obtained
class CLIPScore(Metric):
    r"""Calculates `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption for
    an image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    As output of `forward` and `compute` the metric returns the following output

    - ``clip_score`` (:class:`~torch.Tensor`): float scalar tensor with mean CLIP score over samples

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound = 100.0

    score: Tensor
    n_samples: Tensor
    feature_network: str = "model"

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, img_features: torch.Tensor, txt_features: torch.Tensor) -> None:
        """Update CLIP score on a batch of images and text.

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match

        """
        # Copy from torchmetrics.functional.multimodal.clip_score.py::_clip_score_update
        # cosine similarity between feature vectors
        score = 100 * (img_features * txt_features).sum(axis=-1)
        self.score += score.sum(0)
        self.n_samples += len(txt_features)

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))

    def plot(self, val: Tensor | Sequence[Tensor] | None = None, ax: _AX_TYPE | None = None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.multimodal.clip_score import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> metric.update(torch.randint(255, (3, 224, 224)), "a photo of a cat")
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.multimodal.clip_score import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(255, (3, 224, 224)), "a photo of a cat"))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


def _clip_score_callable(label_info: LabelInfo) -> MetricCollection:  # noqa: ARG001
    return MetricCollection(
        {"clip_score": CLIPScore()},
    )


CLIPScoreCallable: MetricCallable = _clip_score_callable
