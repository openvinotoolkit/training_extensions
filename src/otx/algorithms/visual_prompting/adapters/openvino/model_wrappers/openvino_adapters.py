"""Openvino Adapter Wrappers of OTX Visual Prompting."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import partial
from typing import Tuple

import numpy as np
import openvino.runtime as ov
from openvino.model_api.adapters import OpenvinoAdapter
from openvino.preprocess import ColorFormat, PrePostProcessor
from openvino.runtime import Output, Type
from openvino.runtime import opset10 as opset
from openvino.runtime.utils.decorators import custom_preprocess_function


def resize_image_with_aspect_pad(input: Output, size, keep_aspect_ratio=True, interpolation="linear", pad_value=0):
    """Resize and pad image."""
    h_axis = 1
    w_axis = 2
    w, h = size

    target_size = list(size)
    target_size.reverse()

    image_shape = opset.shape_of(input, name="shape")
    iw = opset.convert(
        opset.gather(image_shape, opset.constant(w_axis), axis=0),
        destination_type="f32",
    )
    ih = opset.convert(
        opset.gather(image_shape, opset.constant(h_axis), axis=0),
        destination_type="f32",
    )
    w_ratio = opset.divide(np.float32(w), iw)
    h_ratio = opset.divide(np.float32(h), ih)
    scale = opset.minimum(w_ratio, h_ratio)

    nw = opset.convert(
        opset.round(opset.multiply(iw, scale), "half_to_even"), destination_type="i32"
    )
    nh = opset.convert(
        opset.round(opset.multiply(ih, scale), "half_to_even"), destination_type="i32"
    )
    new_size = opset.concat([opset.unsqueeze(nh, 0), opset.unsqueeze(nw, 0)], axis=-1)
    image = opset.interpolate(
        input,
        new_size,
        scales=np.array([0.0, 0.0], dtype=np.float32),
        axes=[h_axis, w_axis],
        mode=interpolation,
        shape_calculation_mode="sizes",
    )

    dx = opset.subtract(opset.constant(w, dtype=np.int32), nw)
    dy = opset.subtract(opset.constant(h, dtype=np.int32), nh)
    pads_begin = opset.concat(
        [
            opset.constant([0], dtype=np.int32),
            opset.constant([0], dtype=np.int32),
            opset.constant([0], dtype=np.int32),
            opset.constant([0], dtype=np.int32),
        ],
        axis=0,
    )
    pads_end = opset.concat(
        [
            opset.constant([0], dtype=np.int32),
            opset.unsqueeze(dy, 0),
            opset.unsqueeze(dx, 0),
            opset.constant([0], dtype=np.int32),
        ],
        axis=0
    )

    return opset.pad(
        image,
        pads_begin,
        pads_end,
        "constant",
        opset.constant(pad_value, dtype=np.uint8),
    )


def resize_image_with_aspect(size, interpolation, pad_value):
    return custom_preprocess_function(
        partial(
            resize_image_with_aspect_pad,
            size=size,
            keep_aspect_ratio=True,
            interpolation=interpolation,
        )
    )


class VisualPromptingOpenvinoAdapter(OpenvinoAdapter):
    """Openvino Adapter Wrappers of OTX Visual Prompting.
    
    This class is just to use customized `resize_image_with_aspect_pad` function instead of other resize functions.
    """
    def embed_preprocessing(
        self,
        layout,
        resize_mode: str,
        interpolation_mode,
        target_shape: Tuple[int],
        pad_value,
        dtype=type(int),
        brg2rgb=False,
        mean=None,
        scale=None,
        input_idx=0,
    ):
        ppp = PrePostProcessor(self.model)

        # Change the input type to the 8-bit image
        if dtype == type(int):
            ppp.input(input_idx).tensor().set_element_type(Type.u8)

        ppp.input(input_idx).tensor().set_layout(ov.Layout("NHWC")).set_color_format(
            ColorFormat.BGR
        )

        INTERPOLATION_MODE_MAP = {
            "LINEAR": "linear",
            "CUBIC": "cubic",
            "NEAREST": "nearest",
        }

        RESIZE_MODE_MAP = {"fit_to_window": resize_image_with_aspect}

        # Handle resize
        # Change to dynamic shape to handle various image size
        # TODO: check the number of input channels and rank of input shape
        if resize_mode and target_shape:
            if resize_mode in RESIZE_MODE_MAP:
                input_shape = [1, -1, -1, 3]
                ppp.input(input_idx).tensor().set_shape(input_shape)
                ppp.input(input_idx).preprocess().custom(
                    RESIZE_MODE_MAP[resize_mode](
                        target_shape,
                        INTERPOLATION_MODE_MAP[interpolation_mode],
                        pad_value,
                    )
                )

            else:
                raise ValueError(
                    f"Upsupported resize type in model preprocessing: {resize_mode}"
                )

        # Handle layout
        ppp.input(input_idx).model().set_layout(ov.Layout(layout))

        # Handle color format
        if brg2rgb:
            ppp.input(input_idx).preprocess().convert_color(ColorFormat.RGB)

        ppp.input(input_idx).preprocess().convert_element_type(Type.f32)

        if mean:
            ppp.input(input_idx).preprocess().mean(mean)
        if scale:
            ppp.input(input_idx).preprocess().scale(scale)

        self.model = ppp.build()
        self.load_model()
