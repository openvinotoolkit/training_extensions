"""NNCF wrapped mmcls models builder."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import partial
from typing import Optional, Union

import torch
from mmcv.parallel import DataContainer
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config, ConfigDict

from otx.algorithms.classification.adapters.mmcls.utils import build_classifier
from otx.algorithms.common.adapters.mmcv.nncf.runners import NNCF_META_KEY
from otx.algorithms.common.adapters.mmcv.utils import (
    get_configs_by_pairs,
    remove_from_configs_by_type,
)
from otx.algorithms.common.adapters.nncf import is_accuracy_aware_training_set
from otx.algorithms.common.adapters.nncf.compression import NNCFMetaState
from otx.utils.logger import get_logger

logger = get_logger()


def build_nncf_classifier(  # pylint: disable=too-many-locals,too-many-statements
    config: Config,
    checkpoint: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    cfg_options: Optional[Union[Config, ConfigDict]] = None,
    distributed: bool = False,
):
    """A function to build NNCF wrapped mmcls model."""

    from mmcls.apis import multi_gpu_test, single_gpu_test
    from mmcls.datasets import build_dataloader as mmcls_build_dataloader
    from mmcls.datasets import build_dataset as mmcls_build_dataset
    from mmcls.datasets.pipelines import Compose

    from otx.algorithms.common.adapters.mmcv.nncf.utils import (
        get_fake_input,
        model_eval,
        wrap_nncf_model,
    )
    from otx.algorithms.common.adapters.mmcv.utils.builder import (
        build_dataloader,
        build_dataset,
    )

    if checkpoint is None:
        # load model in this function not in runner
        checkpoint = config.get("load_from")
    assert checkpoint is not None, "checkpoint is not given. NNCF model must be initialized with pretrained model"

    model = build_classifier(config, cfg_options=cfg_options, from_scratch=True)
    model = model.to(device)

    state_dict = CheckpointLoader.load_checkpoint(checkpoint, map_location=device)

    is_acc_aware = is_accuracy_aware_training_set(config.get("nncf_config"))

    init_dataloader = None
    model_eval_fn = None
    if "meta" in state_dict and NNCF_META_KEY in state_dict["meta"]:
        # NNCF ckpt
        nncf_meta_state = state_dict["meta"][NNCF_META_KEY]
        data_to_build_nncf = nncf_meta_state.data_to_build
        state_to_build_nncf = nncf_meta_state.state_to_build
    else:
        # pytorch ckpt
        state_to_build_nncf = state_dict
        if "state_dict" in state_dict:
            state_to_build_nncf = state_dict["state_dict"]

        init_dataloader = build_dataloader(
            build_dataset(
                config,
                subset="train",
                dataset_builder=mmcls_build_dataset,
            ),
            config,
            subset="val",
            dataloader_builder=mmcls_build_dataloader,
            distributed=distributed,
            persistent_workers=False,
        )

        # This data and state dict will be used to build NNCF graph later
        # when loading NNCF model
        # because some models run their subcomponents based on intermediate outputs
        # resulting differently and partially traced NNCF graph
        data_to_build_nncf = next(iter(init_dataloader))["img"]
        if isinstance(data_to_build_nncf, DataContainer):
            data_to_build_nncf = data_to_build_nncf.data[0]
        data_to_build_nncf = data_to_build_nncf.cpu().numpy()
        if len(data_to_build_nncf.shape) == 4:
            data_to_build_nncf = data_to_build_nncf[0]
        if data_to_build_nncf.shape[0] == 3:
            data_to_build_nncf = data_to_build_nncf.transpose(1, 2, 0)

        val_dataloader = None
        if is_acc_aware:
            val_dataloader = build_dataloader(
                build_dataset(
                    config,
                    subset="val",
                    dataset_builder=mmcls_build_dataset,
                ),
                config,
                subset="val",
                dataloader_builder=mmcls_build_dataloader,
                distributed=distributed,
                persistent_workers=False,
            )

        model_eval_fn = partial(
            model_eval,
            config=config,
            val_dataloader=val_dataloader,
            evaluate_fn=multi_gpu_test if distributed else single_gpu_test,
            distributed=distributed,
        )
        state_dict = None

    if config.data.test.pipeline[0]["type"] == "LoadImageFromOTXDataset":
        config.data.test.pipeline = config.data.test.pipeline[1:]
    test_pipeline = Compose(config.data.test.pipeline)
    get_fake_input_fn = partial(
        get_fake_input,
        preprocessor=test_pipeline,
        data=data_to_build_nncf,
    )

    compression_ctrl, model = wrap_nncf_model(
        config,
        model,
        model_eval_fn=model_eval_fn,
        get_fake_input_fn=get_fake_input_fn,
        dataloader_for_init=init_dataloader,
        is_accuracy_aware=is_acc_aware,
    )

    # update runner to save metadata
    config.runner.nncf_meta = NNCFMetaState(
        state_to_build=state_to_build_nncf,
        data_to_build=data_to_build_nncf,
    )

    # update custom hooks
    custom_hooks = config.get("custom_hooks", [])
    custom_hooks.append(ConfigDict({"type": "CancelTrainingHook"}))
    custom_hooks.append(
        ConfigDict(
            type="CompressionHook",
            compression_ctrl=compression_ctrl,
        )
    )
    # TODO: move this to OTX task
    remove_from_configs_by_type(custom_hooks, "CancelInterfaceHook")
    remove_from_configs_by_type(custom_hooks, "TaskAdaptHook")
    remove_from_configs_by_type(custom_hooks, "LazyEarlyStoppingHook")
    remove_from_configs_by_type(custom_hooks, "EarlyStoppingHook")
    config.custom_hooks = custom_hooks

    for hook in get_configs_by_pairs(custom_hooks, dict(type="OTXProgressHook")):
        time_monitor = hook.get("time_monitor", None)
        if time_monitor and getattr(time_monitor, "on_initialization_end", None) is not None:
            time_monitor.on_initialization_end()

    return compression_ctrl, model
