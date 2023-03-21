# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import onnx
import torch

from otx.algorithms.common.adapters.mmdeploy.apis import NaiveExporter
from otx.algorithms.common.adapters.mmdeploy.utils.onnx import (
    prepare_onnx_for_openvino,
    remove_nodes_by_op_type,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmdeploy.test_helpers import create_model


@e2e_pytest_unit
def test_remove_nodes_by_op_type():
    model = create_model("mmcls")

    with tempfile.TemporaryDirectory() as tempdir:
        onnx_path = NaiveExporter.torch2onnx(
            tempdir,
            model,
            {"img": [torch.zeros((1, 50, 50, 3))], "img_metas": []},
        )
        assert os.path.exists(onnx_path)

        onnx_model = onnx.load(onnx_path)
        onnx_model = remove_nodes_by_op_type(onnx_model, "Gemm")
        nodes = []
        for node in onnx_model.graph.node:
            if node.op_type == "Gemm":
                nodes.append(node)
        assert not nodes

        onnx_model = onnx.load(onnx_path)
        onnx_model = remove_nodes_by_op_type(onnx_model, "Conv")
        nodes = []
        for node in onnx_model.graph.node:
            if node.op_type == "Conv":
                nodes.append(node)
        assert not nodes


@e2e_pytest_unit
def test_prepare_onnx_for_openvino():

    model = create_model("mmcls")

    with tempfile.TemporaryDirectory() as tempdir:
        onnx_path = NaiveExporter.torch2onnx(
            tempdir,
            model,
            {"img": [torch.zeros((1, 50, 50, 3))], "img_metas": []},
        )
        assert os.path.exists(onnx_path)

        prepare_onnx_for_openvino(onnx_path, onnx_path)
        onnx_model = onnx.load(onnx_path)
        nodes = []
        for node in onnx_model.graph.node:
            if node.op_type == "Mark":
                nodes.append(node)
        assert not nodes
