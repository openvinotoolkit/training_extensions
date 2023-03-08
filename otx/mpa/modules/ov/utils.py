# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import errno
import os
from typing import List, Optional

from openvino.pyopenvino import Model, Node
from openvino.runtime import Core

from otx.mpa.utils.logger import get_logger

from .omz_wrapper import AVAILABLE_OMZ_MODELS, get_omz_model

logger = get_logger()


def to_dynamic_model(ov_model: Model) -> Model:
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
        except Exception:
            return False

    pop_targets = [["height", "width"], ["batch"]]
    pop_targets = pop_targets[::-1]
    while not reshape_model(ov_model, shapes):
        for key in shapes.keys():
            shape = shapes[key]
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

    ie = Core()
    ov_model = ie.read_model(model=model_path, weights=weight_path)

    if convert_dynamic:
        ov_model = to_dynamic_model(ov_model)

    return ov_model


def normalize_name(name: str) -> str:
    # ModuleDict does not allow '.' in module name string
    name = name.replace(".", "#")
    return name


def unnormalize_name(name: str) -> str:
    name = name.replace("#", ".")
    return name


def get_op_name(op: Node) -> str:
    op_name = op.get_friendly_name()
    op_name = normalize_name(op_name)
    return op_name


def convert_op_to_torch(op: Node):

    from .ops import OPS

    op_type = op.get_type_name()
    op_version = op.get_version()

    try:
        torch_module = OPS.get_by_type_version(op_type, op_version).from_ov(op)
    except Exception as e:
        logger.error(e)
        logger.error(op_type)
        logger.error(op_version)
        logger.error(op.get_attributes())
        raise e

    return torch_module


def convert_op_to_torch_module(target_op: Node):
    from .ops.modules import OperationModule

    dependent_modules = []
    for in_port in target_op.inputs():
        out_port = in_port.get_source_output()
        parent = out_port.get_node()

        parent_type = parent.get_type_name()
        if parent_type == "Constant":
            dependent_modules.append(convert_op_to_torch(parent))
        else:
            dependent_modules.append(None)
    module = convert_op_to_torch(target_op)
    module = OperationModule(module, dependent_modules)
    return module
