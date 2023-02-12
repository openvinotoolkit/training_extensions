# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile
from functools import partial

import numpy as np
import pytest
import torch
from mmcls.datasets.pipelines import Compose
from nncf.torch.nncf_network import NNCFNetwork

from otx.algorithms.common.adapters.mmcv.nncf.utils import (
    get_fake_input,
    model_eval,
    wrap_nncf_model,
)
from otx.algorithms.common.adapters.nncf.compression import NNCFMetaState
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmcv.nncf.test_helpers import (
    create_config,
    create_dataloader,
    create_eval_fn,
    create_model,
)


@e2e_pytest_unit
def test_get_fake_input():
    pipeline = Compose([{"type": "Resize", "size": (50, 50)}, {"type": "Collect", "keys": ["img"]}])

    output = get_fake_input(pipeline)
    assert torch.equal(output["img"][0], torch.zeros((1, 50, 50, 3), dtype=torch.uint8))

    if torch.cuda.is_available():
        output = get_fake_input(pipeline, device="cuda")
        assert torch.equal(
            output["img"][0],
            torch.zeros((1, 50, 50, 3), dtype=torch.uint8, device="cuda"),
        )

    output = get_fake_input(pipeline, np.zeros((128, 128, 3), dtype=np.uint8))
    assert torch.equal(output["img"][0], torch.zeros((1, 50, 50, 3), dtype=torch.uint8))


@e2e_pytest_unit
def test_model_eval():
    mock_model = create_model()
    mock_config = create_config()
    mock_eval_fn = create_eval_fn()
    dataloader = create_dataloader()

    if torch.cuda.is_available():
        model_eval(
            mock_model,
            config=mock_config,
            val_dataloader=dataloader,
            evaluate_fn=mock_eval_fn,
            distributed=False,
        )

        with pytest.raises(RuntimeError):
            model_eval(
                mock_model,
                config=mock_config,
                val_dataloader=None,
                evaluate_fn=mock_eval_fn,
                distributed=False,
            )

        with pytest.raises(RuntimeError):
            mock_config["nncf_config"]["target_metric_name"] = "failed"
            model_eval(
                mock_model,
                config=mock_config,
                val_dataloader=dataloader,
                evaluate_fn=mock_eval_fn,
                distributed=False,
            )


@e2e_pytest_unit
def test_wrap_nncf_model():
    mock_model = create_model()
    mock_config = create_config()
    mock_eval_fn = create_eval_fn()
    dataloader = create_dataloader()

    pipeline = Compose(
        [
            {"type": "Resize", "size": (50, 50)},
            {"type": "Normalize", "mean": [0, 0, 0], "std": [1, 1, 1]},
            {"type": "Collect", "keys": ["img"]},
        ]
    )
    get_fake_input_fn = partial(get_fake_input, pipeline)

    model_eval_fn = partial(
        model_eval,
        config=mock_config,
        val_dataloader=dataloader,
        evaluate_fn=mock_eval_fn,
        distributed=False,
    )

    ctrl, model = wrap_nncf_model(
        mock_config,
        mock_model,
        model_eval_fn=model_eval_fn,
        get_fake_input_fn=get_fake_input_fn,
        dataloader_for_init=dataloader,
        is_accuracy_aware=True,
    )
    assert isinstance(model, NNCFNetwork)

    mock_model = create_model()
    mock_config.nncf_config["input_info"] = {"sample_size": (1, 3, 128, 128)}
    ctrl, model = wrap_nncf_model(
        mock_config,
        mock_model,
        model_eval_fn=model_eval_fn,
        get_fake_input_fn=get_fake_input_fn,
        dataloader_for_init=dataloader,
        is_accuracy_aware=True,
    )
    assert isinstance(model, NNCFNetwork)
    mock_config.nncf_config.pop("input_info")

    with tempfile.TemporaryDirectory() as tempdir:
        mock_model = create_model()
        model_path = os.path.join(tempdir, "model.bin")
        torch.save(mock_model.state_dict(), model_path)
        mock_config.load_from = model_path
        mock_config.nncf_config.log_dir = tempdir
        ctrl, model = wrap_nncf_model(
            mock_config,
            mock_model,
            model_eval_fn=model_eval_fn,
            get_fake_input_fn=get_fake_input_fn,
            dataloader_for_init=dataloader,
            is_accuracy_aware=True,
        )
        assert isinstance(model, NNCFNetwork)
        mock_config.pop("load_from")
        mock_config.nncf_config.pop("log_dir")

        init_state_dict = {
            "meta": {
                "nncf_enable_compression": True,
                "nncf_meta": NNCFMetaState(compression_ctrl=ctrl.get_compression_state()),
            },
            "state_dict": model.state_dict(),
        }
        mock_model = create_model()
        ctrl, model = wrap_nncf_model(
            mock_config,
            mock_model,
            model_eval_fn=model_eval_fn,
            get_fake_input_fn=get_fake_input_fn,
            dataloader_for_init=dataloader,
            init_state_dict=init_state_dict,
            is_accuracy_aware=True,
        )
