# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for getitune data entities."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import torch
import torch.utils._pytree as pytree
from datumaro.experimental.dataset import Sample  # noqa: TC002
from datumaro.experimental.fields import image_field
from datumaro.experimental.fields.images import ImageField, ImagePathField
from datumaro.experimental.fields.videos import MediaPathField

if TYPE_CHECKING:
    from collections.abc import Iterable

    from datumaro.experimental.dataset import Dataset

    from getitune.data.entity.base import ImageInfo

#: Map from IntensityConfig.storage_dtype strings to Polars dtype instances.
STORAGE_DTYPE_MAP: dict[str, pl.DataType] = {
    "uint8": pl.UInt8(),
    "uint16": pl.UInt16(),
    "int16": pl.Int16(),
    "float32": pl.Float32(),
}

#: PIL modes that indicate high-bit-depth (>8-bit) images.
_PIL_16BIT_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N"})
_PIL_FLOAT_MODES = frozenset({"F"})

#: Magic bytes for image format detection.
_PNG_SIGNATURE = b"\x89PNG"
_JPEG_SIGNATURE = b"\xff\xd8"
_TIFF_LE = b"II"
_TIFF_BE = b"MM"


def detect_storage_dtype(dataset: Dataset) -> tuple[str, int]:
    """Detect image storage dtype and channel count from the dataset.

    Reads the first sample's media file header to determine bit depth
    and number of colour channels without decoding the entire image.

    Must be called on a raw dataset (before ``convert_to_schema``).

    Args:
        dataset: A ``datumaro.experimental.Dataset`` instance.

    Returns:
        A ``(storage_dtype, num_channels)`` tuple where
        *storage_dtype* is one of ``"uint8"``, ``"uint16"``, ``"int16"``,
        ``"float32"`` and *num_channels* is the image channel count
        (e.g.  1 = grayscale, 3 = RGB, 4 = RGBA).

    Raises:
        ValueError: If no image field is found in the dataset schema.
    """
    for name, attr in dataset.schema.attributes.items():
        if isinstance(attr.field, (ImagePathField, MediaPathField)):
            return _detect_dtype_from_file(Path(dataset.df[name][0]))
        if isinstance(attr.field, ImageField):
            dtype = str(attr.field.dtype).lower()
            fmt = getattr(attr.field, "format", "RGB")
            num_channels = {"L": 1, "RGB": 3, "RGBA": 4}.get(fmt, 3)
            return dtype, num_channels

    msg = (
        f"Cannot detect image storage dtype: no image field "
        f"found in dataset schema. Available fields: {list(dataset.schema.attributes.keys())}"
    )
    raise ValueError(msg)


def _detect_dtype_from_file(path: Path) -> tuple[str, int]:
    """Detect image storage dtype and channel count from a file header.

    Reads the minimum bytes needed to determine bit depth and colour
    channels without decoding the full image.
    """
    with path.open("rb") as f:
        sig = f.read(8)

    # PNG: bit_depth at byte 24, color_type at byte 25
    if sig[:4] == _PNG_SIGNATURE:
        with path.open("rb") as f:
            f.seek(24)  # signature(8) + length(4) + "IHDR"(4) + width(4) + height(4)
            bit_depth = struct.unpack("B", f.read(1))[0]
            color_type = struct.unpack("B", f.read(1))[0]
        dtype = "uint16" if bit_depth == 16 else "uint8"
        # color_type: 0=Grayscale, 2=RGB, 4=Gray+Alpha, 6=RGBA
        num_channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type, 1)
        return dtype, num_channels

    # JPEG is always 8-bit; check mode via PIL for channel count
    if sig[:2] == _JPEG_SIGNATURE:
        from PIL import Image

        with Image.open(path) as img:
            num_channels = 1 if img.mode == "L" else 3
        return "uint8", num_channels

    # TIFF: check dtype and mode via PIL
    if sig[:2] in (_TIFF_LE, _TIFF_BE):
        from PIL import Image

        with Image.open(path) as img:
            if img.mode in _PIL_16BIT_MODES:
                dtype = "uint16"
            elif img.mode in _PIL_FLOAT_MODES:
                dtype = "float32"
            else:
                tag_v2 = getattr(img, "tag_v2", None)
                if tag_v2 and 258 in tag_v2:
                    bits = tag_v2[258]
                    if isinstance(bits, tuple):
                        bits = bits[0]
                    dtype = "uint16" if bits == 16 else "uint8"
                else:
                    dtype = "uint8"
            num_channels = 1 if img.mode in ("L", "I", "I;16", "F") else 3
        return dtype, num_channels

    # Fallback: use PIL mode
    from PIL import Image

    with Image.open(path) as img:
        mode = img.mode
    if mode in _PIL_16BIT_MODES:
        dtype = "uint16"
    elif mode in _PIL_FLOAT_MODES:
        dtype = "float32"
    else:
        dtype = "uint8"
    num_channels = 1 if mode in ("L", "I", "I;16", "F") else 3
    return dtype, num_channels


#: Cache for dynamically created sample classes to avoid re-creation.
_SAMPLE_DTYPE_CACHE: dict[tuple[type, str], type[Sample]] = {}


def with_image_dtype(
    sample_cls: type[Sample],
    storage_dtype: str,
) -> type[Sample]:
    """Create a variant of *sample_cls* whose ``image`` field uses *storage_dtype*.

    When ``storage_dtype == "uint8"`` (the default) the original class is
    returned unchanged — zero overhead for the common case.

    For other dtypes a thin **dynamic subclass** is created that overrides the
    ``image`` class-variable with the requested Polars dtype.  The subclass is
    cached so repeated calls with the same arguments return the same class
    object (important for Datumaro schema identity comparisons).

    The dynamic class is used only during dataset construction
    (``dm_subset.convert_to_schema(sample_type)``) and is **not** stored on
    the dataset instance, so it never needs to survive pickle across
    DataLoader worker boundaries.

    Args:
        sample_cls: One of the concrete sample classes (e.g.
            :class:`ClassificationSample`, :class:`DetectionSample`).
        storage_dtype: A key from :data:`STORAGE_DTYPE_MAP` — ``"uint8"``,
            ``"uint16"``, ``"int16"``, or ``"float32"``.

    Returns:
        Either *sample_cls* itself (uint8) or a dynamically created subclass
        with the overridden ``image`` field.
    """
    if storage_dtype == "uint8":
        return sample_cls

    pl_dtype = STORAGE_DTYPE_MAP.get(storage_dtype)
    if pl_dtype is None:
        msg = f"Unsupported storage_dtype={storage_dtype!r}. Supported values: {list(STORAGE_DTYPE_MAP)}"
        raise ValueError(msg)

    cache_key = (sample_cls, storage_dtype)
    if cache_key in _SAMPLE_DTYPE_CACHE:
        return _SAMPLE_DTYPE_CACHE[cache_key]

    orig_image = getattr(sample_cls, "image", None)
    channels_first = getattr(orig_image, "channels_first", True)
    fmt = getattr(orig_image, "format", "RGB")

    new_image_default = image_field(dtype=pl_dtype, channels_first=channels_first, format=fmt)

    new_cls_name = f"{sample_cls.__name__}_{storage_dtype}"
    new_cls: type[Sample] = type(  # type: ignore[assignment]
        new_cls_name,
        (sample_cls,),
        {"image": new_image_default},
    )
    new_cls.__module__ = sample_cls.__module__
    new_cls.__qualname__ = new_cls_name

    # Make the class discoverable by pickle in the current process.
    # In spawned workers the module-level __getattr__ in
    # getitune.data.entity.sample handles the lookup instead.
    parent_module = sys.modules.get(sample_cls.__module__)
    if parent_module is not None:
        setattr(parent_module, new_cls_name, new_cls)

    # Register with pytree so torchvision v2 transforms work
    register_pytree_node(new_cls)

    _SAMPLE_DTYPE_CACHE[cache_key] = new_cls
    return new_cls


def register_pytree_node(cls: type[Sample]) -> type[Sample]:
    """Decorator to register a getitune data entity with PyTorch's PyTree.

    This decorator should be applied to every getitune data entity, as TorchVision V2 transforms
    use the PyTree to flatten and unflatten the data entity during runtime.

    Example:
        `MulticlassClsDataEntity` example ::

            @register_pytree_node
            @dataclass
            class MulticlassClsDataEntity(DataEntity):
                ...
    """

    def flatten_fn(obj: object) -> tuple[list[Any], list[str]]:
        obj_dict = dict(obj.__dict__)

        missing_keys = set(obj.__class__.__annotations__.keys()) - set(obj_dict.keys())
        for key in missing_keys:
            obj_dict[key] = getattr(obj, key)

        return (list(obj_dict.values()), list(obj_dict.keys()))

    def unflatten_fn(values: Iterable[Any], context: Any) -> object:  # noqa: ANN401
        kwargs = dict(zip(context, values))
        # Extract _img_info to set after construction (since __post_init__ would overwrite it)
        img_info = kwargs.pop("_img_info", None)
        obj = cls(**kwargs)
        # Restore _img_info if it was present (preserves transformed img_info)
        if img_info is not None:
            object.__setattr__(obj, "_img_info", img_info)
        return obj

    pytree.register_pytree_node(
        cls,
        flatten_fn=flatten_fn,
        unflatten_fn=unflatten_fn,
    )
    return cls


def stack_batch(
    tensor_list: list[torch.Tensor],
    img_info_list: list[ImageInfo],
    pad_size_divisor: int = 1,
    pad_value: int | float = 0,
) -> tuple[torch.Tensor, list[ImageInfo]]:
    """Stack multiple tensors to form a batch.

    Pad the tensor to the max shape use the right bottom padding mode in these images.
    If ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        img_info_list (List[Tensor]): A list of img_info to be updated.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
        (tuple[torch.Tensor, list[ImageInfo]]): The n dim tensor and updated a list of ImageInfo.
    """
    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list), img_info_list
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    batch_info = []
    for idx, (tensor, info) in enumerate(zip(tensor_list, img_info_list)):
        padded_img = torch.nn.functional.pad(tensor, tuple(pad[idx].tolist()), value=pad_value)
        # update img_info.img_shape
        info.img_shape = padded_img.shape[1:]
        # update img_info.padding
        left, top, right, bottom = info.padding
        info.padding = (left + pad[idx, 0], top + pad[idx, 2], right + pad[idx, 1], bottom + pad[idx, 3])
        # append padded img
        batch_tensor.append(padded_img)
        batch_info.append(info)

    stacked_images = torch.stack(batch_tensor)

    return stacked_images, batch_info
