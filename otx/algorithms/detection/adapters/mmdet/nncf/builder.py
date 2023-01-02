# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from functools import partial
from typing import Optional, Union

import torch
from mmcv.runner import CheckpointLoader, load_state_dict
from mmcv.utils import Config, ConfigDict
from mmdet.utils import get_root_logger

from otx.algorithms.common.adapters.mmcv.utils import (
    get_configs_by_dict,
    get_configs_by_keys,
    remove_from_configs_by_type,
)
from otx.algorithms.common.adapters.nncf import is_accuracy_aware_training_set
from otx.algorithms.common.adapters.nncf.compression import (
    DATA_TO_BUILD_NAME,
    NNCF_STATE_NAME,
    STATE_TO_BUILD_NAME,
)
from otx.algorithms.common.adapters.nncf.utils import no_nncf_trace
from otx.algorithms.detection.adapters.mmdet.utils import build_detector


logger = get_root_logger()


def build_nncf_detector(
    config: Config,
    train_cfg: Optional[Union[Config, ConfigDict]] = None,
    test_cfg: Optional[Union[Config, ConfigDict]] = None,
    checkpoint: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    cfg_options: Optional[Union[Config, ConfigDict]] = None,
    distributed=False,
):
    from mmdet.apis import multi_gpu_test, single_gpu_test
    from mmdet.apis.inference import LoadImage
    from mmdet.datasets import build_dataloader as mmdet_build_dataloader
    from mmdet.datasets import build_dataset
    from mmdet.datasets.pipelines import Compose
    from nncf.torch.dynamic_graph.io_handling import nncf_model_input

    from otx.algorithms.common.adapters.mmcv.nncf import (
        build_dataloader,
        get_fake_input,
        model_eval,
        wrap_nncf_model,
    )

    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if checkpoint is None:
        # load model in this function not in runner
        checkpoint = config.load_from
        config.load_from = None
    assert checkpoint is not None

    model = build_detector(
        config, train_cfg=train_cfg, test_cfg=test_cfg, from_scratch=True
    )
    model = model.to(device)

    state_dict = CheckpointLoader.load_checkpoint(checkpoint, map_location=device)

    is_acc_aware = is_accuracy_aware_training_set(config.get("nncf_config"))

    init_dataloader = None
    model_eval_fn = None
    if NNCF_STATE_NAME in state_dict:
        # NNCF ckpt
        data_to_build_nncf = state_dict[DATA_TO_BUILD_NAME]
        state_to_build_nncf = state_dict[STATE_TO_BUILD_NAME]
    else:
        # pytorch ckpt
        state_to_build_nncf = state_dict

        # This data and state dict will be used to build NNCF graph later
        # when loading NNCF model
        # because some models run their subcomponents based on intermediate outputs
        # resulting differently and partially traced NNCF graph
        datasets = get_configs_by_keys(config.data.train, "otx_dataset")
        data_to_build_nncf = datasets[0][0].numpy

        init_dataloader = build_dataloader(
            config,
            subset="train",
            distributed=distributed,
            dataloader_builder=mmdet_build_dataloader,
            dataset_builder=build_dataset,
        )

        val_dataloader = None
        if is_acc_aware:
            val_dataloader = build_dataloader(
                config,
                subset="val",
                distributed=distributed,
                dataloader_builder=mmdet_build_dataloader,
                dataset_builder=build_dataset,
            )

        model_eval_fn = partial(
            model_eval,
            config=config,
            val_dataloader=val_dataloader,
            evaluate_fn=multi_gpu_test if distributed else single_gpu_test,
            distributed=distributed,
        )
        state_dict = None
    load_state_dict(model, state_to_build_nncf)

    test_pipeline = [LoadImage()] + config.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    get_fake_input_fn = partial(
        get_fake_input,
        preprocessor=test_pipeline,
        data=data_to_build_nncf,
    )

    if "nncf_compress_postprocessing" in config:
        # NB: This parameter is used to choose if we should try to make NNCF compression
        #     for a whole model graph including postprocessing (`nncf_compress_postprocessing=True`),
        #     or make NNCF compression of the part of the model without postprocessing
        #     (`nncf_compress_postprocessing=False`).
        #     Our primary goal is to make NNCF compression of such big part of the model as
        #     possible, so `nncf_compress_postprocessing=True` is our primary choice, whereas
        #     `nncf_compress_postprocessing=False` is our fallback decision.
        #     When we manage to enable NNCF compression for sufficiently many models,
        #     we should keep one choice only.
        nncf_compress_postprocessing = config.get("nncf_compress_postprocessing")
        logger.debug(
            "set should_compress_postprocessing=" f"{nncf_compress_postprocessing}"
        )
    else:
        # TODO: Do we have to keep this configuration?
        # This configuration is not enabled in forked mmdetection library in the first place
        nncf_compress_postprocessing = True

    def dummy_forward_fn(model):
        def _get_fake_data_for_forward(nncf_config):
            input_size = nncf_config.get("input_info").get("sample_size")
            assert len(input_size) == 4 and input_size[0] == 1
            H, W, C = input_size[2], input_size[3], input_size[1]
            device = next(model.parameters()).device
            with no_nncf_trace():
                return get_fake_input_fn(shape=tuple([H, W, C]), device=device)

        fake_data = _get_fake_data_for_forward(config.nncf_config)
        img, img_metas = fake_data["img"], fake_data["img_metas"]
        ctx = model.nncf_trace_context(img_metas, nncf_compress_postprocessing)
        with ctx:
            # The device where model is could be changed under this context
            img = [i.to(next(model.parameters()).device) for i in img]
            # Marking data as NNCF network input must be after device movement
            img = [nncf_model_input(i) for i in img]
            if nncf_compress_postprocessing:
                logger.debug(
                    "NNCF will try to compress a postprocessing part of the model"
                )
            else:
                logger.debug(
                    "NNCF will NOT compress a postprocessing part of the model"
                )
                img = img[0]
            model(img)

    compression_ctrl, model = wrap_nncf_model(
        config,
        model,
        init_state_dict=state_dict,
        model_eval_fn=model_eval_fn,
        dummy_forward_fn=dummy_forward_fn,
        dataloader_for_init=init_dataloader,
        is_accuracy_aware=is_acc_aware,
    )

    # update custom hooks
    custom_hooks = config.get("custom_hooks", [])
    custom_hooks.append(
        ConfigDict(type="CompressionHook", compression_ctrl=compression_ctrl)
    )
    custom_hooks.append(ConfigDict({"type": "CancelTrainingHook"}))
    custom_hooks.append(
        ConfigDict(
            type="CheckpointHookBeforeTraining",
            save_optimizer=True,
            meta={
                DATA_TO_BUILD_NAME: data_to_build_nncf,
                STATE_TO_BUILD_NAME: state_to_build_nncf,
            },
        )
    )
    remove_from_configs_by_type(custom_hooks, "CancelInterfaceHook")
    remove_from_configs_by_type(custom_hooks, "TaskAdaptHook")
    remove_from_configs_by_type(custom_hooks, "EMAHook")
    remove_from_configs_by_type(custom_hooks, "CustomModelEMAHook")
    remove_from_configs_by_type(custom_hooks, "AdaptiveTrainSchedulingHook")

    for hook in get_configs_by_dict(custom_hooks, dict(type="OTXProgressHook")):
        time_monitor = hook.get("time_monitor", None)
        if (
            time_monitor
            and getattr(time_monitor, "on_initialization_end", None) is not None
        ):
            time_monitor.on_initialization_end()

    return compression_ctrl, model
