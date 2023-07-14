# type: ignore
# TODO: Need to remove line 1 (ignore mypy) and fix mypy issues
"""Utils for otx.core.ov."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import errno
import os
from typing import Optional

from openvino.runtime import Core, Model, Node

from .omz_wrapper import AVAILABLE_OMZ_MODELS, get_omz_model

# pylint: disable=too-many-locals


def to_dynamic_model(ov_model: Model) -> Model:
    """Convert ov_model to dynamic Model."""
    assert isinstance(ov_model, Model)

    shapes = {}
    target_layouts = {}
    for input_node in ov_model.inputs:
        target_layout = {
            "batch": ["N", None, None],
            "height": ["H", None, None],
            "width": ["W", None, None],
        }

        any_name = input_node.any_name
        parameter_node = input_node.get_node()
        layout = parameter_node.get_layout()
        if layout.empty:
            continue
        layout = layout.to_string()[1:-1].split(",")
        shape = [str(i) for i in input_node.get_partial_shape()]
        for i, (layout_name, shape_) in enumerate(zip(layout, shape)):
            try:
                shape_ = int(shape_)
            except ValueError:
                shape_ = -1

            for target_layout_ in target_layout.values():
                target_layout_name = target_layout_[0]
                if layout_name == target_layout_name:
                    target_layout_[1] = i
                    target_layout_[2] = shape_
                    shape_ = -1
                    break
            shape[i] = shape_

        shapes[any_name] = shape
        target_layouts[any_name] = target_layout

    def reshape_model(ov_model, shapes):
        try:
            ov_model.reshape(shapes)
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            return False

    pop_targets = [["height", "width"], ["batch"]]
    pop_targets = pop_targets[::-1]
    while not reshape_model(ov_model, shapes):
        for key, shape in shapes.items():
            target_layout = target_layouts[key]

            targets = pop_targets.pop()
            for target in targets:
                target_idx, target_origin = target_layout[target][1:]
                if target_idx is not None:
                    shape[target_idx] = target_origin

        if len(pop_targets) == 0:
            reshape_model(ov_model, shapes)
            break

    return ov_model


def load_ov_model(model_path: str, weight_path: Optional[str] = None, convert_dynamic: bool = False) -> Model:
    """Load ov_model from model_path."""
    model_path = str(model_path)
    if model_path.startswith("omz://"):
        model_path = model_path.replace("omz://", "")
        assert model_path in AVAILABLE_OMZ_MODELS
        ov_ir_path = get_omz_model(model_path)
        model_path = ov_ir_path["model_path"]
        weight_path = ov_ir_path["weight_path"]

    if weight_path is None:
        weight_path = os.path.splitext(model_path)[0] + ".bin"

    if not os.path.exists(model_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weight_path)

    ie_core = Core()
    ov_model = ie_core.read_model(model=model_path, weights=weight_path)

    if convert_dynamic:
        ov_model = to_dynamic_model(ov_model)

    return ov_model


def normalize_name(name: str) -> str:
    """Normalize name string."""
    # ModuleDict does not allow '.' in module name string
    name = name.replace(".", "#")
    return f"{name}"


def unnormalize_name(name: str) -> str:
    """Unnormalize name string."""
    name = name.replace("#", ".")
    return name


def get_op_name(op_node: Node) -> str:
    """Get op name string."""
    op_name = op_node.get_friendly_name()
    op_name = normalize_name(op_name)
    return op_name
