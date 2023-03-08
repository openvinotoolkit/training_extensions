# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import torch
from nncf.torch.nncf_network import NNCFNetwork

from otx.algorithms.common.adapters.nncf.compression import NNCFMetaState
from otx.algorithms.segmentation.adapters.mmseg.nncf.builder import build_nncf_segmentor
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmcv.nncf.test_helpers import (
    create_config,
    create_dataset,
    create_model,
)


@e2e_pytest_unit
def test_build_nncf_segmentor():
    mock_config = create_config(lib="mmseg")
    model = create_model(lib="mmseg")
    create_dataset(lib="mmseg")

    with tempfile.TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, "model.bin")
        state_to_build = model.state_dict()
        torch.save(state_to_build, model_path)
        mock_config.load_from = model_path
        ctrl, model = build_nncf_segmentor(mock_config)
        assert isinstance(model, NNCFNetwork)
        assert len([hook for hook in mock_config.custom_hooks if hook.type == "CompressionHook"]) == 1
        mock_config.pop("custom_hooks")

        torch.save(
            {
                "meta": {
                    "nncf_enable_compression": True,
                    "nncf_meta": NNCFMetaState(
                        data_to_build=np.zeros((50, 50, 3)),
                        compression_ctrl=ctrl.get_compression_state(),
                        state_to_build=state_to_build,
                    ),
                },
                "state_dict": model.state_dict(),
            },
            model_path,
        )
        ctrl, model = build_nncf_segmentor(mock_config, model_path)
        assert isinstance(model, NNCFNetwork)
        assert len([hook for hook in mock_config.custom_hooks if hook.type == "CompressionHook"]) == 1
