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

    def update(self, image_features: torch.Tensor, text_features: torch.Tensor) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            image_features: Tensor of shape (N, D) with image embeddings
            text_features: Tensor of shape (M, D) with text embeddings
        """
        # Calculate cosine similarity between feature vectors
        cosine_sim = torch.nn.functional.cosine_similarity(image_features, text_features, dim=-1)
        score = 100 * cosine_sim
        self.score += score.sum(0)
        self.n_samples += len(text_features)

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
        return torch.clamp(self.score / self.n_samples, min=0.0)

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
        """
        return self._plot(val, ax)


class ImageTextMeanAveragePrecision(Metric):
    """Computes the mean average precision for image-text retrieval."""

    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound = 1.0

    def __init__(self, k: int = 10, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("average_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, image_features: Tensor, text_features: Tensor) -> None:
        """Update function to calculate average precision for image-text retrieval.

        Args:
            image_features: Tensor of shape (N, D) with image embeddings
            text_features: Tensor of shape (M, D) with text embeddings
        """
        # Calculate cosine similarities between image and text features
        cosine_similarities = torch.matmul(image_features, text_features.T)

        # Calculate average precision for each image
        for i in range(cosine_similarities.shape[0]):
            similarities = cosine_similarities[i]

            # Sort by predicted scores
            sorted_indices = torch.argsort(similarities, descending=True)

            # Assume ground truth is the diagonal (i.e., i-th image matches i-th text)
            true_sorted = torch.zeros_like(similarities)
            true_sorted[sorted_indices == i] = 1

            # Calculate precision at each rank
            precisions = []
            num_hits = 0
            for j in range(min(self.k, len(true_sorted))):
                if true_sorted[j] == 1:
                    num_hits += 1
                    precisions.append(num_hits / (j + 1))

            if precisions:
                self.average_precision += torch.tensor(precisions).mean()
                self.total += 1

    def compute(self) -> Tensor:
        """Computes the clip score.

        Returns:
            Tensor: The average precision divided by the total if the total is greater than 0,
                    otherwise returns a tensor with value 0.0.
        """
        return self.average_precision / self.total if self.total > 0 else torch.tensor(0.0)

    def reset(self) -> None:
        """Resets the metrics for average precision and total count.

        This method initializes `self.average_precision` to a tensor with a value of 0.0
        and `self.total` to a tensor with a value of 0.
        """
        self.average_precision = torch.tensor(0.0)
        self.total = torch.tensor(0)


def _clip_metric_callable(label_info: LabelInfo) -> MetricCollection:  # noqa: ARG001
    return MetricCollection(
        {
            "clip_score": CLIPScore(),
            "mAP": ImageTextMeanAveragePrecision(),
        },
    )


CLIPMetricCallable: MetricCallable = _clip_metric_callable
