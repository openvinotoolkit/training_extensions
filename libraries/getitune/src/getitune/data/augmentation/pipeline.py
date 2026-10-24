# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU and GPU augmentation pipelines for getitune.

This module provides the core augmentation pipeline classes:
- CPUAugmentationPipeline: Runs in Dataset workers using torchvision.transforms.v2
- GPUAugmentationPipeline: Runs on GPU using Kornia (to be implemented)
"""

from __future__ import annotations

import ast
import operator
import typing
from copy import copy, deepcopy
from inspect import isclass
from typing import TYPE_CHECKING, Any, Callable

import kornia.augmentation as K  # noqa: N812
import torch
import torchvision.transforms.v2 as tvt_v2
import typeguard
from kornia.augmentation.container import ops
from lightning.pytorch.cli import instantiate_class
from omegaconf import DictConfig
from torch import nn

from getitune.config.data import IntensityConfig
from getitune.data.augmentation.cache import CacheableMixin
from getitune.data.augmentation.intensity import build_intensity_transform
from getitune.data.entity.sample import BaseSample
from getitune.data.utils import import_object_from_module

if TYPE_CHECKING:
    from getitune.config.data import SubsetConfig
    from getitune.data.dataset.base import VisionDataset


_KORNIA_PATCHED = False
_original_transform_list = ops.MaskSequentialOps.transform_list


def _fixed_transform_list(cls, input, module, param, extra_args=None):  # noqa: ANN001, ANN202, A002
    """Fixed version that slices transform_matrix for each list element."""
    if extra_args is None:
        extra_args = {}
    if isinstance(module, (K.GeometricAugmentationBase2D,)):
        tfm_input = []
        params = cls.get_instance_module_param(param)
        params_i = deepcopy(params)
        for i, inp in enumerate(input):
            params_i["batch_prob"] = params["batch_prob"][i : i + 1]
            transform_i = module.transform_matrix[i : i + 1] if module.transform_matrix is not None else None
            tfm_inp = module.transform_masks(
                inp, params=params_i, flags=module.flags, transform=transform_i, **extra_args
            )
            tfm_input.append(tfm_inp)
        return tfm_input
    return _original_transform_list.__func__(cls, input, module, param, extra_args)  # type: ignore[attr-defined]


def _ensure_kornia_patched() -> None:
    """Apply the Kornia MaskSequentialOps monkey-patch on first use."""
    global _KORNIA_PATCHED  # noqa: PLW0603
    if _KORNIA_PATCHED:
        return
    ops.MaskSequentialOps.transform_list = classmethod(_fixed_transform_list)  # type: ignore[assignment]
    _KORNIA_PATCHED = True


# Mapping from storage_dtype string to bit depth.
_DTYPE_TO_BIT_DEPTH: dict[str, int] = {
    "uint8": 8,
    "uint16": 16,
    "int16": 16,
    "float32": 32,
}


def _eval_input_size_str(str_to_eval: str) -> tuple[int, ...] | int:
    """Safely evaluate an arithmetic expression involving ``$(input_size)``.

    Only multiplication and division are supported.  Operands may be
    constants or tuples.  The result is rounded to ``int``.

    Args:
        str_to_eval: String expression to evaluate.

    Returns:
        Evaluated result as int or tuple of ints.
    """
    bin_ops: dict[type, Callable[[Any, Any], Any]] = {
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    un_ops: dict[type, Callable[[Any], Any]] = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    available_ops = tuple(bin_ops) + tuple(un_ops) + (ast.BinOp, ast.UnaryOp)

    tree = ast.parse(str_to_eval, mode="eval")

    def _eval(node: Any) -> Any:  # noqa: ANN401
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return torch.tensor([_eval(val) for val in node.elts])
        if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
            left = _eval(node.left)
            right = _eval(node.right)
            return bin_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in un_ops:
            operand = _eval(node.operand) if isinstance(node.operand, available_ops) else node.operand.value
            return un_ops[type(node.op)](operand)
        msg = f"Bad syntax, {type(node)}. Available operations for calculating input size are {available_ops}"
        raise SyntaxError(msg)

    ret = _eval(tree)
    if isinstance(ret, torch.Tensor):
        return tuple(ret.round().int().tolist())
    return round(ret)


def _configure_input_size(
    cfg: dict[str, Any],
    input_size: int | tuple[int, int] | None,
) -> dict[str, Any]:
    """Replace ``$(input_size)`` placeholders in augmentation config ``init_args``.

    Input size should be specified as ``$(input_size)``
    (e.g. ``$(input_size) * 0.5``).  Only simple multiplication or division
    evaluation is supported.  The function decides whether to pass a ``tuple``
    or ``int`` based on the type-hint of the target argument.  Floating-point
    values are rounded to ``int``.

    Args:
        cfg: Augmentation config dict with ``class_path`` and ``init_args``.
        input_size: Target input size ``(H, W)`` or single ``int``.

    Returns:
        Config with placeholders replaced by actual values.
    """
    init_args = cfg.get("init_args", {})
    if not init_args:
        return cfg

    _input_size: tuple[int, int] | None = None
    if input_size is not None:
        _input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)  # type: ignore[assignment]

    def check_type(value: Any, expected_type: Any) -> bool:  # noqa: ANN401
        try:
            typeguard.check_type(value, expected_type)
        except typeguard.TypeCheckError:
            return False
        return True

    model_cls = None
    for key, val in init_args.items():
        if not (isinstance(val, str) and "$(input_size)" in val):
            continue

        if input_size is None:
            msg = (
                f"{cfg['class_path'].split('.')[-1]} initial argument has `$(input_size)`, "
                "but input_size is set to None."
            )
            raise RuntimeError(msg)

        if model_cls is None:
            model_cls = import_object_from_module(cfg["class_path"])

        available_types = typing.get_type_hints(model_cls.__init__).get(key)

        if available_types is None or check_type(_input_size, available_types):
            # Pass tuple[int, int]
            init_args[key] = _eval_input_size_str(val.replace("$(input_size)", str(_input_size)))
        elif check_type(_input_size[0], available_types):  # type: ignore[index]
            # Pass int
            init_args[key] = _eval_input_size_str(val.replace("$(input_size)", str(_input_size[0])))  # type: ignore[index]
        else:
            msg = f"{key} argument should be able to get int or tuple[int, int], but it can get {available_types}"
            raise RuntimeError(msg)

    return cfg


class _IntensityAdapter(nn.Module):
    """Wrap an intensity transform and stamp ``img_info.bit_depth``.

    Unlike :class:`_SampleImageAdapter`, this also records the original
    bit-depth of the image (derived from ``storage_dtype``) on the sample's
    :class:`~getitune.data.entity.base.ImageInfo`.  Downstream code (e.g. YOLOX)
    can use ``img_info.bit_depth`` to reject unsupported high-bit-depth inputs.
    """

    def __init__(self, transform: nn.Module, storage_dtype: str = "uint8") -> None:
        super().__init__()
        self.transform = transform
        self.bit_depth = _DTYPE_TO_BIT_DEPTH.get(storage_dtype, 8)

    def forward(self, sample: BaseSample) -> BaseSample:
        """Apply intensity transform and set ``img_info.bit_depth``."""
        sample.image = self.transform(sample.image)
        if hasattr(sample, "img_info") and sample.img_info is not None:
            sample.img_info.bit_depth = self.bit_depth
        return sample


class CPUAugmentationPipeline(nn.Module):
    """CPU-stage augmentation pipeline using torchvision.transforms.v2.

    This pipeline runs in Dataset workers (before collate) and handles:
    - Intensity mapping (uint16 → float32 for medical images)
    - Size-dependent geometric augmentations (Resize, RandomResizedCrop)
    - Augmentations applied to image, bboxes, masks, keypoints, etc.

    All outputs are fixed-size tensors suitable for batch stacking.

    The pipeline supports two types of transforms:
    1. getitune-style transforms: Have a `forward(*DataItem)` signature and handle
       all data types internally (image, bboxes, masks, etc.)
    2. Native torchvision.v2 transforms: Applied to all tv_tensors extracted
       from the DataItem (image, bboxes, masks, keypoints)

    Args:
        augmentations: List of torchvision.transforms.v2 transforms or getitune transforms.

    Example:
        >>> pipeline = CPUAugmentationPipeline([
        ...     v2.RandomResizedCrop(size=(224, 224)),
        ...     v2.RandomHorizontalFlip(p=0.5),
        ...     v2.ToDtype(torch.float32, scale=True),
        ... ])
        >>> item = pipeline(item)  # DataItem with fixed-size image
    """

    def __init__(self, augmentations: list[nn.Module] | None = None) -> None:
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations or [])
        self._mean, self._std = self._extract_normalization_params(list(self.augmentations))

    @staticmethod
    def _extract_normalization_params(
        augmentations: list[nn.Module],
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        """Extract mean and std from the first torchvision Normalize transform found.

        Args:
            augmentations: List of augmentation modules.

        Returns:
            Tuple of (mean, std) extracted from first Normalize found, or (None, None).
        """
        for transform in augmentations:
            if isinstance(transform, tvt_v2.Normalize):
                mean: tuple[float, float, float] = tuple(float(v) for v in transform.mean)  # type: ignore[assignment]
                std: tuple[float, float, float] = tuple(float(v) for v in transform.std)  # type: ignore[assignment]
                return mean, std
        return None, None

    @property
    def mean(self) -> tuple[float, float, float] | None:
        """Get normalization mean."""
        return self._mean

    @property
    def std(self) -> tuple[float, float, float] | None:
        """Get normalization std."""
        return self._std

    def prepare(self, dataset: VisionDataset) -> None:
        """Pre-populate caches in augmentations that support it.

        Delegates to ``pre_cache(dataset)`` on each augmentation that
        implements this method (e.g. CachedMosaic, CachedMixUp).
        Call before creating a multi-worker DataLoader so all workers
        inherit full, diverse caches.

        Args:
            dataset: The training dataset (must support ``len()`` and ``[]``).
        """
        for aug in self.augmentations:
            if isinstance(aug, CacheableMixin):
                aug.pre_cache(dataset)

    @classmethod
    def list_available_transforms(cls) -> list[type[tvt_v2.Transform]]:
        """List available TorchVision transform (only V2) classes."""
        return [
            obj
            for name in dir(tvt_v2)
            if (obj := getattr(tvt_v2, name)) and isclass(obj) and issubclass(obj, tvt_v2.Transform)
        ]

    @classmethod
    def from_config(cls, config: SubsetConfig) -> CPUAugmentationPipeline:
        """Build CPU augmentation pipeline from SubsetConfig.

        This function handles:
        - Intensity mapping (prepended automatically from ``IntensityConfig``)
        - New ``augmentations_cpu`` field (preferred)
        - Input size placeholder replacement ``$(input_size)``

        The intensity transform is always the **first** operation in the
        pipeline.  For uint8 with ``mode="scale_to_unit"`` this is equivalent
        to the old ``to_dtype(float32, scale=True)``; for uint16 / thermal /
        medical inputs it applies the correct domain-specific mapping.

        Args:
            config: SubsetConfig with augmentations_cpu and intensity.

        Returns:
            CPUAugmentationPipeline ready for use in Dataset.
        """
        input_size = getattr(config, "input_size", None)
        aug_configs = config.augmentations_cpu
        intensity_config = getattr(config, "intensity", None)
        if isinstance(intensity_config, dict):
            intensity_config = IntensityConfig(**intensity_config)

        augmentations: list[nn.Module] = []

        # --- 1. Prepend intensity mapping transform ---------------------------
        if intensity_config is not None:
            intensity_transform = build_intensity_transform(intensity_config)
            augmentations.append(_IntensityAdapter(intensity_transform, intensity_config.storage_dtype))

        # --- 2. User-configured augmentations ---------------------------------
        if aug_configs:
            for aug_config in aug_configs:
                cfg = copy(aug_config)
                if isinstance(cfg, (dict, DictConfig)):
                    # Handle input_size placeholder
                    cfg = _configure_input_size(dict(cfg), input_size)

                    # Instantiate the transform
                    transform = cls._dispatch_transform(cfg)
                elif isinstance(cfg, nn.Module):
                    transform = cfg
                else:
                    msg = f"Unsupported augmentation config type: {type(cfg)}"
                    raise TypeError(msg)

                augmentations.append(transform)

        return cls(augmentations)

    @classmethod
    def _dispatch_transform(cls, cfg_transform: DictConfig | dict | nn.Module) -> nn.Module:
        """Dispatch and instantiate a transform from config or return as-is.

        Args:
            cfg_transform: Transform config dict or already instantiated transform.

        Returns:
            Instantiated transform.
        """
        if isinstance(cfg_transform, (DictConfig, dict)):
            return instantiate_class(args=(), init=dict(cfg_transform))
        if isinstance(cfg_transform, nn.Module):
            # Already instantiated transform, return as-is
            return cfg_transform

        msg = f"CPUAugmentationPipeline accepts only DictConfig | dict | nn.Module, got {type(cfg_transform)}."
        raise TypeError(msg)

    def _is_native_torchvision_transform(self, transform: nn.Module) -> bool:
        """Return True if the transform should be applied via ``_apply_native_transform``.

        Rules:
        - Pure torchvision transforms (module starts with ``torchvision.``) → native.
        - getitune subclasses of ``tvt_v2.Transform`` that define their own ``forward()``
          (e.g. ``Resize``, ``CachedMosaic``) handle ``BaseSample`` themselves → NOT native.
        - getitune wrappers that only add ``__call__`` probability gating without a custom
          ``forward()`` (e.g. ``RandomIoUCrop``) delegate to the parent torchvision
          ``forward()`` and must go through ``_apply_native_transform`` → native.
        """
        module = type(transform).__module__
        if module.startswith("torchvision."):
            return True
        # getitune class that is a tvt_v2.Transform subclass: treat as native only when it
        # does NOT define its own forward() (i.e. it relies on the parent's forward).
        if isinstance(transform, tvt_v2.Transform):
            return "forward" not in type(transform).__dict__
        return False

    def _apply_native_transform(self, transform: nn.Module, inputs: BaseSample) -> BaseSample:  # type: ignore[return-value]
        """Apply native torchvision transform only to image-related fields.

        TorchVision v2 expects standard field names like `boxes`/`labels`; we
        map to those before calling the transform and map back afterward.
        We also keep `img_info` in sync when the image size changes.
        """
        # Build a dict of transformable fields with torchvision-friendly keys.
        transformable: dict[str, Any] = {}
        if (image := getattr(inputs, "image", None)) is not None:
            transformable["image"] = image
        if (img_info := getattr(inputs, "img_info", None)) is not None:
            transformable["img_info"] = img_info
        if (masks := getattr(inputs, "masks", None)) is not None:
            transformable["masks"] = masks
        if (bboxes := getattr(inputs, "bboxes", None)) is not None:
            transformable["boxes"] = bboxes
        if (label := getattr(inputs, "label", None)) is not None:
            transformable["labels"] = label

        if not transformable:
            return inputs

        # Apply transform to transformable fields
        # If there's only an image, pass it directly; otherwise pass as dict
        if len(transformable) == 1 and "image" in transformable:
            result = transform(transformable["image"])
            inputs.image = result
        else:
            # Reverse mapping: torchvision key → BaseSample attribute name
            tv_to_getitune = {"boxes": "bboxes", "labels": "label"}

            result = transform(transformable)
            if isinstance(result, dict):
                for key, value in result.items():
                    attr = tv_to_getitune.get(key, key)
                    setattr(inputs, attr, value)
            else:
                # Single result, assume it's the image
                inputs.image = result
        return inputs

    def forward(self, *inputs: BaseSample) -> BaseSample | None:
        """Forward with skipping None."""
        needs_unpacking = len(inputs) > 1
        outputs: BaseSample = inputs[0]  # type: ignore[assignment]
        for transform in self.augmentations:
            if self._is_native_torchvision_transform(transform):
                # Apply native transforms only to image-related fields
                outputs = self._apply_native_transform(transform, inputs[0])
            else:
                outputs = transform(*inputs)
            if outputs is None:
                return outputs
            inputs = outputs if needs_unpacking else (outputs,)  # type: ignore[assignment]

        if isinstance(outputs.image, torch.Tensor):
            outputs.image = outputs.image.clamp(0.0, 1.0)

        return outputs

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        aug_strs = [f"  {aug}" for aug in self.augmentations]
        return "CPUAugmentationPipeline(\n" + "\n".join(aug_strs) + "\n)"


class GPUAugmentationPipeline(nn.Module):
    """GPU-stage augmentation pipeline using Kornia AugmentationSequential.

    This pipeline runs on GPU after batch transfer via Lightning Callback.
    It uses Kornia's AugmentationSequential for efficient batch-level processing
    with support for multiple data types (images, bboxes, masks, keypoints).

    Key features:
    - Uses Kornia AugmentationSequential for optimized batch processing
    - Supports data_keys for transforming bboxes, masks, keypoints along with images
    - Extracts normalization parameters for model export

    The pipeline expects batched tensors in BCHW format with values in [0, 1].

    Args:
        augmentations: List of Kornia augmentation modules.
        data_keys: List of data keys to transform (e.g., ["input", "bbox", "mask"]).
            Defaults to ["input"] for image-only augmentation.
        sanitize_annotations: Whether to clip and filter bboxes/keypoints after
            geometric transforms.  Set to ``False`` for validation/test pipelines
            where ground-truth coordinates may be in original image space to prevent
            clipping them to the smaller network input dimensions.

    Example:
        >>> import kornia.augmentation as K
        >>> pipeline = GPUAugmentationPipeline(
        ...     augmentations=[
        ...         K.RandomHorizontalFlip(p=0.5),
        ...         K.ColorJiggle(0.1, 0.1, 0.1, 0.1),
        ...         K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ...     ],
        ...     data_keys=["input"],
        ... )
        >>> augmented_images = pipeline(batch_images)
    """

    def __init__(
        self,
        augmentations: list[nn.Module] | None = None,
        data_keys: list[str] | None = None,
        sanitize_annotations: bool = True,
    ) -> None:
        super().__init__()
        self._augmentations_list = augmentations or []
        self._data_keys = data_keys or ["input"]
        self._mean, self._std = self._extract_normalization_params(self._augmentations_list)
        self._sanitize_annotations_enabled = sanitize_annotations

        # Check if pipeline contains geometric augmentations that can move/resize objects.
        # If only intensity transforms (Normalize, ColorJiggle, etc.) are present,
        # we skip _sanitize_annotations to avoid clipping bboxes that may be in a
        # different coordinate space (e.g., original image coords with resize_targets=False).
        self._has_geometric_augs = any(
            isinstance(aug, K.GeometricAugmentationBase2D) for aug in self._augmentations_list
        )

        # Build Kornia AugmentationSequential for efficient batch processing
        self.aug_sequential: K.AugmentationSequential | None = None
        if self._augmentations_list:
            _ensure_kornia_patched()
            # Cast to Any because Kornia stubs restrict to _AugmentationBase but
            # any nn.Module works at runtime.
            _augs: Any = self._augmentations_list
            self.aug_sequential = K.AugmentationSequential(
                *_augs,
                data_keys=self._data_keys,
                same_on_batch=False,
            )

    @staticmethod
    def _extract_normalization_params(
        augmentations: list[nn.Module],
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        """Extract mean and std from Normalize augmentation in the list.

        Args:
            augmentations: List of augmentation modules.

        Returns:
            Tuple of (mean, std) extracted from first Normalize found, or (None, None).
        """
        mean: tuple[float, float, float] | None = None
        std: tuple[float, float, float] | None = None

        for aug in augmentations:
            # Check if this is a Normalize augmentation (Kornia stores in flags dict)
            if isinstance(aug, K.Normalize):
                mean = typing.cast("tuple[float, float, float]", tuple(aug.flags["mean"].tolist()))
                std = typing.cast("tuple[float, float, float]", tuple(aug.flags["std"].tolist()))
                # Stop after finding the first Normalize
                break

        return mean, std

    @property
    def mean(self) -> tuple[float, float, float] | None:
        """Get normalization mean."""
        return self._mean

    @property
    def std(self) -> tuple[float, float, float] | None:
        """Get normalization std."""
        return self._std

    @property
    def data_keys(self) -> list[str]:
        """Get data keys used by the pipeline."""
        return self._data_keys

    def apply_image_only(self, images: torch.Tensor) -> torch.Tensor:
        """Apply only image-level (non-geometric) augmentations to a batch of images.

        Unlike :meth:`forward`, this does not require (or transform) annotations and
        therefore does not depend on the Kornia ``AugmentationSequential`` data-keys.
        It is used for tile batches during validation/testing, where the tiles carry
        no per-tile annotations (predictions are merged back to the original image
        afterwards) and only intensity transforms such as ``Normalize`` are relevant.

        Design contract for tiling: geometric augmentations are applied in the CPU
        pipeline (per-sample, with annotations) so that the GPU pipeline used for
        tiles contains only intensity transforms. As a result ``sanitize_annotations``
        is a no-op here (see ``_has_geometric_augs`` gating in :meth:`forward`). Any
        geometric augmentation that still ends up on the GPU pipeline is skipped
        defensively, since applying it without annotation tracking would desynchronise
        boxes/masks.

        Args:
            images: Batched images tensor in BCHW format, values in [0, 1].

        Returns:
            The transformed images tensor.
        """
        if not self._augmentations_list:
            return images
        out = images
        for aug in self._augmentations_list:
            if isinstance(aug, K.GeometricAugmentationBase2D):
                continue
            out = aug(out)
        return out

    @classmethod
    def list_available_transforms(cls) -> list[type]:
        """List available Kornia augmentation classes."""
        return [
            obj
            for name in dir(K)
            if (obj := getattr(K, name))
            and isclass(obj)
            and issubclass(obj, (K.AugmentationBase2D, K.IntensityAugmentationBase2D))
        ]

    @classmethod
    def from_config(
        cls,
        config: SubsetConfig,
        data_keys: list[str] | None = None,
        sanitize_annotations: bool = True,
    ) -> GPUAugmentationPipeline:
        """Build GPU augmentation pipeline from SubsetConfig.

        This function handles:
        - `augmentations_gpu` field with Kornia augmentations
        - Extraction of normalization parameters for model update
        - Input size placeholder replacement $(input_size)
        - data_keys for Kornia AugmentationSequential

        Args:
            config: SubsetConfig with augmentations_gpu.
            data_keys: List of data keys for AugmentationSequential.
                Defaults to ["input"] for image-only augmentation.
            sanitize_annotations: Forwarded to ``GPUAugmentationPipeline.__init__``.
                Pass ``False`` for val/test pipelines.

        Returns:
            GPUAugmentationPipeline ready for use in Callback.
        """
        input_size = getattr(config, "input_size", None)
        aug_configs = config.augmentations_gpu

        if not aug_configs:
            return cls([], data_keys=data_keys, sanitize_annotations=sanitize_annotations)

        augmentations = []

        for aug_config in aug_configs:
            cfg = copy(aug_config)
            if isinstance(cfg, (dict, DictConfig)):
                # Handle input_size placeholder
                cfg = _configure_input_size(dict(cfg), input_size)

                # Instantiate the transform
                transform = cls._dispatch_transform(cfg)
            elif isinstance(cfg, nn.Module):
                transform = cfg
            else:
                msg = f"Unsupported augmentation config type: {type(cfg)}"
                raise TypeError(msg)

            augmentations.append(transform)

        return cls(augmentations, data_keys=data_keys, sanitize_annotations=sanitize_annotations)

    @classmethod
    def _dispatch_transform(cls, cfg_transform: DictConfig | dict | nn.Module) -> nn.Module:
        """Dispatch and instantiate a transform from config or return as-is.

        Args:
            cfg_transform: Transform config dict or already instantiated transform.

        Returns:
            Instantiated transform.
        """
        if isinstance(cfg_transform, (DictConfig, dict)):
            return instantiate_class(args=(), init=dict(cfg_transform))
        if isinstance(cfg_transform, nn.Module):
            # Already instantiated transform, return as-is
            return cfg_transform
        msg = f"GPUAugmentationPipeline accepts only DictConfig | dict | nn.Module, got {type(cfg_transform)}."
        raise TypeError(msg)

    def forward(
        self,
        images: torch.Tensor,
        labels: list[torch.Tensor] | None = None,
        bboxes: list[torch.Tensor] | None = None,
        masks: list[torch.Tensor] | None = None,
        keypoints: list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Apply GPU augmentations to batched data using Kornia AugmentationSequential.

        Args:
            images: Batched images tensor in BCHW format, values in [0, 1].
            labels: List of labels per image (optional).
            bboxes: List of bounding boxes per image (optional).
            masks: List of masks per image (optional). Can be:
                - Semantic segmentation: (C, H, W) format
                - Instance segmentation: (N_instances, H, W) format
            keypoints: List of keypoints per image (optional).

        Returns:
            Dict with augmented data:
            {"images": tensor, "labels": list, "bboxes": list, "masks": list, "keypoints": list}
        """
        if self.aug_sequential is None:
            return {"images": images, "labels": labels, "bboxes": bboxes, "masks": masks, "keypoints": keypoints}

        # Handle instance segmentation masks: Kornia expects (N, C, H, W) but instance
        # masks are (N_instances, H, W). We add a channel dim before and squeeze after.
        # This allows Kornia to properly transform instance masks along with images.
        if masks is not None and "mask" in self._data_keys:
            # Instance seg masks: (N_instances, H, W) - 3D per sample
            # Semantic seg masks: (C, H, W) where C is often 1 or num_classes
            # We add channel dim to all masks for consistency with Kornia
            masks = [m.unsqueeze(0) for m in masks]  # (N, H, W) -> (N, 1, H, W)

        # Kornia expects keypoints as a single (B, N, 2) tensor, not a list.
        if keypoints is not None and "keypoints" in self._data_keys:
            keypoints = torch.stack(keypoints)  # type: ignore[assignment]  # (B, N, 2)

        # Map data key names to actual data
        data_map = {
            "input": images,
            "label": labels,
            "bbox_xyxy": bboxes,
            "mask": masks,
            "keypoints": keypoints,
        }

        # Build input list in the SAME ORDER as self._data_keys
        # This is critical because Kornia uses the order to match data to keys
        inputs = []
        provided_keys = []
        for key in self._data_keys:
            data = data_map.get(key)
            if data is not None:
                inputs.append(data)
                provided_keys.append(key)

        # Apply augmentation to all inputs
        results = self.aug_sequential(*inputs)

        # Kornia returns a plain tensor when only one data key is provided,
        # but a list when multiple keys are used. Normalise to always be a list.
        if not isinstance(results, (list, tuple)):
            results = [results]

        # Parse results back
        output = {"images": None, "labels": labels, "bboxes": bboxes, "masks": masks, "keypoints": keypoints}
        for i, key in enumerate(provided_keys):
            if key == "input":
                output["images"] = results[i]
            elif key == "label":
                output["labels"] = results[i]
            elif key == "bbox_xyxy":
                output["bboxes"] = results[i]
            elif key == "mask":
                # Remove channel
                mask_results = results[i]
                mask_results = [m.squeeze(0) for m in mask_results]  # (1, N, H, W) -> (N, H, W)
                output["masks"] = mask_results
            elif key == "keypoints":
                # Kornia returns (B, N, 2) tensor; split back to list of per-sample tensors
                kp_result = results[i]
                if isinstance(kp_result, torch.Tensor) and kp_result.dim() == 3:
                    output["keypoints"] = list(kp_result.unbind(0))
                else:
                    output["keypoints"] = kp_result

        # Sanitize geometric annotations after Kornia transforms.
        if self._sanitize_annotations_enabled and self._has_geometric_augs and output["images"] is not None:
            s_bboxes, s_labels, s_masks, s_keypoints = self._sanitize_annotations(
                output["images"],
                output["bboxes"],
                output["labels"],
                output["masks"],
                output["keypoints"],
            )
            output["bboxes"] = s_bboxes
            output["labels"] = s_labels
            output["masks"] = s_masks
            output["keypoints"] = s_keypoints

        return output

    def _sanitize_annotations(
        self,
        images: torch.Tensor,
        bboxes: list[torch.Tensor] | None,
        labels: list[torch.Tensor] | None,
        masks: list[torch.Tensor] | None,
        keypoints: list[torch.Tensor] | None,
        min_size: float = 4.0,
        min_area: float = 16.0,
    ) -> tuple[
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        """Sanitize transformed annotations.

        - Clip bboxes to image bounds
        - Remove invalid bboxes (non-finite, x2<=x1, y2<=y1, too small)
        - Filter aligned labels/masks/keypoints using the same valid indices
        """
        if bboxes is None:
            return bboxes, labels, masks, keypoints

        batch_size, _, h, w = images.shape
        if len(bboxes) != batch_size:
            msg = f"GPU sanitize: bboxes batch mismatch, got {len(bboxes)} vs {batch_size}"
            raise RuntimeError(msg)

        if labels is not None and len(labels) != batch_size:
            msg = f"GPU sanitize: labels batch mismatch, got {len(labels)} vs {batch_size}"
            raise RuntimeError(msg)
        if masks is not None and len(masks) != batch_size:
            msg = f"GPU sanitize: masks batch mismatch, got {len(masks)} vs {batch_size}"
            raise RuntimeError(msg)
        if keypoints is not None and len(keypoints) != batch_size:
            msg = f"GPU sanitize: keypoints batch mismatch, got {len(keypoints)} vs {batch_size}"
            raise RuntimeError(msg)

        out_bboxes: list[torch.Tensor] = []
        out_labels: list[torch.Tensor] | None = [] if labels is not None else None
        out_masks: list[torch.Tensor] | None = [] if masks is not None else None
        out_keypoints: list[torch.Tensor] | None = [] if keypoints is not None else None

        for i in range(batch_size):
            boxes = bboxes[i]
            if not (boxes.ndim == 2 and boxes.shape[-1] == 4):
                msg = f"GPU sanitize: bboxes[{i}] must be [N,4], got {tuple(boxes.shape)}"
                raise RuntimeError(msg)

            if boxes.numel() == 0:
                clipped = boxes
                valid = torch.zeros((0,), dtype=torch.bool, device=boxes.device)
            else:
                clipped = boxes
                clipped[:, 0::2].clamp_(0, w)
                clipped[:, 1::2].clamp_(0, h)

                x1, y1, x2, y2 = clipped[:, 0], clipped[:, 1], clipped[:, 2], clipped[:, 3]
                widths = x2 - x1
                heights = y2 - y1
                areas = widths * heights
                valid = (
                    torch.isfinite(clipped).all(dim=1)
                    & (widths > min_size)
                    & (heights > min_size)
                    & (areas >= min_area)
                )

            out_bboxes.append(clipped[valid])

            if out_labels is not None and labels is not None:
                lbl = labels[i]
                if not (lbl.ndim >= 1 and lbl.shape[0] == valid.shape[0]):
                    msg = f"GPU sanitize: labels[{i}] size mismatch with bboxes ({lbl.shape[0]} vs {valid.shape[0]})"
                    raise RuntimeError(msg)
                out_labels.append(lbl[valid])

            if out_masks is not None and masks is not None:
                m = masks[i]
                # For instance masks, first dimension corresponds to object index.
                if m.ndim >= 3 and m.shape[0] == valid.shape[0]:
                    out_masks.append(m[valid])
                elif m.ndim == 2 and valid.shape[0] == 1:
                    out_masks.append(m.unsqueeze(0)[valid])
                else:
                    out_masks.append(m)

            if out_keypoints is not None and keypoints is not None:
                kp = keypoints[i]
                if kp.numel() > 0:
                    kp[..., 0].clamp_(0, w)
                    kp[..., 1].clamp_(0, h)
                if kp.ndim >= 2 and kp.shape[0] == valid.shape[0]:
                    out_keypoints.append(kp[valid])
                else:
                    out_keypoints.append(kp)

        return out_bboxes, out_labels, out_masks, out_keypoints

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        aug_str = str(self.aug_sequential) if self.aug_sequential is not None else "  (empty)"
        info = f"  mean={self._mean}, std={self._std}" if self._mean or self._std else ""
        return f"GPUAugmentationPipeline(\n{aug_str}\n  data_keys={self._data_keys}{info}\n)"
