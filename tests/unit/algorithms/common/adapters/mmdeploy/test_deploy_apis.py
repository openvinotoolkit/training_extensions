# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import torch
from mmcv.utils import Config

from otx.algorithms.common.adapters.mmdeploy.apis import NaiveExporter
from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmdeploy.test_helpers import (
    create_config,
    create_model,
)


class TestNaiveExporter:
    @e2e_pytest_unit
    def test_sub_component(self):
        model = create_model("mmcls")

        with tempfile.TemporaryDirectory() as tempdir:
            onnx_path = NaiveExporter.torch2onnx(
                tempdir,
                model,
                {"img": [torch.zeros((1, 50, 50, 3))], "img_metas": []},
            )
            assert os.path.exists(onnx_path)

            openvino_paths = NaiveExporter.onnx2openvino(
                tempdir,
                onnx_path,
            )
            for openvino_path in openvino_paths:
                assert os.path.exists(openvino_path)

    @e2e_pytest_unit
    def test_export2backend(self):
        from otx.algorithms.classification.adapters.mmcls.utils.builder import (
            build_classifier,
        )

        config = create_config()
        create_model("mmcls")

        with tempfile.TemporaryDirectory() as tempdir:
            NaiveExporter.export2backend(
                tempdir,
                build_classifier,
                config,
                {"img": [torch.zeros((50, 50, 3))], "img_metas": []},
            )
            assert [f for f in os.listdir(tempdir) if f.endswith(".xml")]
            assert [f for f in os.listdir(tempdir) if f.endswith(".bin")]


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER, mark

    from otx.algorithms.common.adapters.mmdeploy.apis import MMdeployExporter

    class TestMMdeployExporter:
        @e2e_pytest_unit
        def test_sub_component(self):
            config = create_config()
            deploy_config = Config(
                {
                    "ir_config": {
                        "type": "onnx",
                        "input_names": ["input"],
                        "output_names": ["output"],
                    },
                    "codebase_config": {
                        "type": "mmcls",
                        "task": "Classification",
                    },
                    "backend_config": {
                        "type": "openvino",
                        "model_inputs": [
                            {
                                "opt_shapes": {
                                    "input": [1, 3, 50, 50],
                                }
                            }
                        ],
                    },
                }
            )
            create_model("mmcls")

            with tempfile.TemporaryDirectory() as tempdir:
                onnx_path = MMdeployExporter.torch2onnx(
                    tempdir,
                    np.zeros((50, 50, 3), dtype=np.float32),
                    config,
                    deploy_config,
                )
                assert isinstance(onnx_path, str)
                assert os.path.exists(onnx_path)

                openvino_paths = MMdeployExporter.onnx2openvino(
                    tempdir,
                    onnx_path,
                    deploy_config,
                )
                for openvino_path in openvino_paths:
                    assert os.path.exists(openvino_path)

        @e2e_pytest_unit
        def test_export2backend(self):
            from otx.algorithms.classification.adapters.mmcls.utils.builder import (
                build_classifier,
            )

            config = create_config()
            deploy_config = Config(
                {
                    "ir_config": {
                        "type": "onnx",
                        "input_names": ["input"],
                        "output_names": ["output"],
                    },
                    "codebase_config": {
                        "type": "mmcls",
                        "task": "Classification",
                    },
                    "backend_config": {
                        "type": "openvino",
                        "model_inputs": [
                            {
                                "opt_shapes": {
                                    "input": [1, 3, 50, 50],
                                }
                            }
                        ],
                    },
                }
            )
            create_model("mmcls")

            with tempfile.TemporaryDirectory() as tempdir:
                MMdeployExporter.export2backend(
                    tempdir,
                    build_classifier,
                    config,
                    deploy_config,
                    str(ExportType.OPENVINO),
                )
                assert [f for f in os.listdir(tempdir) if f.endswith(".xml")]
                assert [f for f in os.listdir(tempdir) if f.endswith(".bin")]

        @e2e_pytest_unit
        def test_partition(self):
            from otx.algorithms.classification.adapters.mmcls.utils.builder import (
                build_classifier,
            )

            config = create_config()
            deploy_config = Config(
                {
                    "ir_config": {
                        "type": "onnx",
                        "input_names": ["input"],
                        "output_names": ["output"],
                    },
                    "codebase_config": {
                        "type": "mmcls",
                        "task": "Classification",
                    },
                    "backend_config": {
                        "type": "openvino",
                        "model_inputs": [
                            {
                                "opt_shapes": {
                                    "input": [1, 3, 50, 50],
                                }
                            }
                        ],
                    },
                    "partition_config": {
                        "apply_marks": True,
                        "partition_cfg": [
                            {
                                "save_file": "partition.onnx",
                                "start": ["test:input"],
                                "end": ["test:output"],
                                "output_names": ["output"],
                                "mo_options": {
                                    "_delete_": True,
                                    "args": {},
                                    "flags": [],
                                },
                            }
                        ],
                    },
                }
            )
            create_model("mmcls")

            @FUNCTION_REWRITER.register_rewriter(
                "tests.unit.algorithms.common.adapters.mmdeploy.test_helpers.MockModel.forward"
            )
            @mark("test", inputs=["input"], outputs=["output"])
            def forward(ctx, self, *args, **kwargs):
                return ctx.origin_func(self, *args, **kwargs)

            with tempfile.TemporaryDirectory() as tempdir:
                MMdeployExporter.export2backend(tempdir, build_classifier, config, deploy_config, "OPENVINO")
                files = os.listdir(tempdir)
                assert "model.onnx" in files
                assert "model.xml" in files
                assert "model.bin" in files
                assert "partition.onnx" in files
                assert "partition.xml" in files
                assert "partition.bin" in files
