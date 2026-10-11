# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample classes for getitune data entities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Sequence, cast

import polars as pl
import torch
from datumaro.experimental import register_sample
from datumaro.experimental.dataset import Sample as _DatumSample
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import (
    Subset,
    bbox_field,
    image_field,
    image_info_field,
    instance_mask_field,
    keypoints_field,
    label_field,
    mask_field,
    subset_field,
)
from torchvision import tv_tensors

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.utils import (
    register_pytree_node,
)
from getitune.data.entity.validation import (
    validate_bboxes,
    validate_feature_vectors,
    validate_images,
    validate_imgs_info,
    validate_keypoints,
    validate_labels,
    validate_masks,
    validate_scores,
)

if TYPE_CHECKING:
    from torchvision.tv_tensors import BoundingBoxes, Mask


@register_pytree_node
class BaseSample(_DatumSample):
    """Base class for getitune data samples."""

    image: torch.Tensor | tv_tensors.Image
    subset: Subset = subset_field()

    @property
    def img_info(self) -> ImageInfo:
        """Get image information for the sample."""
        if self._img_info is None:
            err_msg = "img_info is not set."
            raise ValueError(err_msg)
        return self._img_info

    @img_info.setter
    def img_info(self, value: ImageInfo) -> None:
        self._img_info = value


@register_pytree_node
@register_sample
class ClassificationSample(BaseSample):
    """ClassificationSample is a base class for getitune classification items."""

    subset: Subset = subset_field()

    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8())
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
@register_sample
class ClassificationMultiLabelSample(BaseSample):
    """ClassificationMultiLabelSample is a base class for getitune multi label classification items."""

    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8(), multi_label=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
@register_sample
class DetectionSample(BaseSample):
    """DetectionSample is a base class for getitune detection items."""

    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), format="RGB", channels_first=True)
    label: torch.Tensor | None = label_field(pl.UInt8(), is_list=True)
    # Use Union type to allow torch.Tensor from Polars (since tv_tensors.BoundingBoxes
    # conversion is not supported in Datumaro), then convert in __post_init__
    bboxes: tv_tensors.BoundingBoxes | torch.Tensor = bbox_field(dtype=pl.Float32())
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        # Handle None fields for images with no annotations
        if self.label is None:
            self.label = torch.zeros(0, dtype=torch.uint8)

        # Ensure bboxes are tv_tensors.BoundingBoxes
        if not isinstance(self.bboxes, tv_tensors.BoundingBoxes):
            bboxes = self.bboxes if self.bboxes is not None else torch.zeros((0, 4), dtype=torch.float32)
            # Round-tripping through storage can yield a degenerate (1, 0) empty tensor instead of (0, 4).
            bboxes = bboxes.reshape(-1, 4)
            self.bboxes = tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=shape,
                dtype=torch.float32,
            )

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
@register_sample
class SegmentationSample(BaseSample):
    """BaseSample for segmentation tasks."""

    subset: Subset = subset_field()
    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=False)
    masks: tv_tensors.Mask = mask_field(dtype=pl.UInt8(), channels_first=True, has_channels_dim=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)
        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
@register_sample
class InstanceSegmentationSample(BaseSample):
    """BaseSample for instance segmentation tasks."""

    subset: Subset = subset_field()
    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    # Use Union type to allow torch.Tensor from Polars (since tv_tensors.BoundingBoxes
    # conversion is not supported in Datumaro), then convert in __post_init__
    bboxes: tv_tensors.BoundingBoxes | torch.Tensor = bbox_field(dtype=pl.Float32())
    masks: tv_tensors.Mask | None = instance_mask_field(dtype=pl.UInt8())
    label: torch.Tensor | None = label_field(dtype=pl.UInt8(), is_list=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        # Handle None fields for images with no annotations
        if self.label is None:
            self.label = torch.zeros(0, dtype=torch.uint8)
        if self.masks is None:
            self.masks = tv_tensors.Mask(torch.zeros((0, *shape), dtype=torch.uint8))

        # Ensure bboxes are tv_tensors.BoundingBoxes
        if not isinstance(self.bboxes, tv_tensors.BoundingBoxes):
            bboxes = self.bboxes if self.bboxes is not None else torch.zeros((0, 4), dtype=torch.float32)
            # Round-tripping through storage can yield a degenerate (1, 0) empty tensor instead of (0, 4).
            bboxes = bboxes.reshape(-1, 4)
            self.bboxes = tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=shape,
                dtype=torch.float32,
            )

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
@register_sample
class KeypointSample(BaseSample):
    """KeypointSample is a base class for getitune keypoint detection items."""

    subset: Subset = subset_field()
    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(dtype=pl.UInt8(), is_list=True)
    keypoints: torch.Tensor = keypoints_field()
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@dataclass
class SampleBatch:
    """getitune sample batch implementation.

    Attributes:
        images: The batch of images as a BCHW tensor.
        labels: List of label tensors, optional.
        masks: List of masks, optional.
        bboxes: List of bounding boxes, optional.
        keypoints: List of keypoint tensors, optional.
        imgs_info: Sequence of image information, optional.
    """

    images: torch.Tensor | tv_tensors.Image | list[torch.Tensor] | list[tv_tensors.Image]
    labels: list[torch.Tensor] | None = None
    masks: list[Mask] | None = None
    bboxes: list[BoundingBoxes] | None = None
    keypoints: list[torch.Tensor] | None = None
    imgs_info: Sequence[ImageInfo | None] | None = None

    @property
    def batch_size(self) -> int:
        """Get the number of samples in the batch."""
        if isinstance(self.images, list):
            return len(self.images)
        return self.images.shape[0]

    def __post_init__(self) -> None:
        """Validate the batch after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate the batch fields."""
        validate_images(cast("torch.Tensor | list[torch.Tensor]", self.images))
        if self.labels is not None:
            validate_labels(self.labels)
        if self.bboxes is not None:
            validate_bboxes(self.bboxes)
        if self.keypoints is not None:
            validate_keypoints(self.keypoints)
        if self.masks is not None:
            validate_masks(self.masks)
        if self.imgs_info is not None:
            validate_imgs_info(self.imgs_info)

    def pin_memory(self) -> SampleBatch:
        """Pin memory for member tensor variables."""
        # https://github.com/pytorch/pytorch/issues/116403

        kwargs = {}

        def maybe_pin(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, torch.Tensor):
                return x.pin_memory()
            return x

        def maybe_wrap_tv(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, tv_tensors.TVTensor):
                return tv_tensors.wrap(x.pin_memory(), like=x)
            return maybe_pin(x)

        # Handle images separately because of tv_tensors wrapping
        if self.images is not None:
            kwargs["images"] = maybe_wrap_tv(self.images)

        # Generic handler for all other fields
        for field in ["labels", "bboxes", "keypoints", "masks"]:
            value = getattr(self, field)
            if value is not None:
                kwargs[field] = [maybe_wrap_tv(v) if v is not None else None for v in value]

        return self.wrap(**kwargs)

    def wrap(self, **kwargs) -> SampleBatch:
        """Wrap this dataclass with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be overwritten on top of this dataclass

        Returns:
            Updated dataclass
        """
        updated = object.__new__(self.__class__)
        updated.__dict__.update(asdict(self))
        updated.__dict__.update(kwargs)
        return updated


@dataclass
class PredictionBatch(SampleBatch):
    """getitune prediction batch implementation.

    Extends SampleBatch with prediction-specific fields.

    Attributes:
        scores: List of score tensors, optional.
        feature_vector: List of feature vector tensors, optional.
        saliency_map: List of saliency map tensors, optional.
    """

    scores: list[torch.Tensor] | None = None
    feature_vector: list[torch.Tensor] | None = None
    saliency_map: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        """Validate the prediction batch after initialization."""
        super().__post_init__()
        if self.scores is not None:
            validate_scores(self.scores)
        if self.feature_vector is not None:
            validate_feature_vectors(self.feature_vector)


@dataclass
class Prediction:
    """getitune prediction data entity for a single sample.

    This is used for storing individual prediction results, e.g., after tile merging.

    Attributes:
        image: The image tensor.
        img_info: Image metadata information.
        label: The predicted label tensor, optional.
        masks: The predicted masks, optional.
        bboxes: The predicted bounding boxes, optional.
        keypoints: The predicted keypoints, optional.
        scores: The prediction scores, optional.
        feature_vector: The feature vector for XAI, optional.
        saliency_map: The saliency map for XAI, optional.
    """

    image: torch.Tensor | tv_tensors.Image
    img_info: ImageInfo | None = None
    label: torch.Tensor | None = None
    masks: Mask | None = None
    bboxes: BoundingBoxes | None = None
    keypoints: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    feature_vector: torch.Tensor | None = None
    saliency_map: torch.Tensor | None = None


def __getattr__(name: str) -> type:
    """PEP 562 hook: recreate dynamic sample-dtype classes for pickle.

    ``with_image_dtype()`` creates subclasses like ``ClassificationSample_uint16``
    at runtime.  When a DataLoader worker (spawned process) unpickles the
    dataset, Python looks up these classes by name on this module.  In a fresh
    process they don't exist yet, so this hook recreates them on demand.
    """
    for suffix in ("_uint16", "_int16", "_float32"):
        if name.endswith(suffix):
            base_cls = globals().get(name[: -len(suffix)])
            if base_cls is not None and isinstance(base_cls, type):
                from getitune.data.entity.utils import with_image_dtype

                return with_image_dtype(base_cls, suffix[1:])  # strip leading '_'
    raise AttributeError(name)
