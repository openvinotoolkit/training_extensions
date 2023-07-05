"""NNCF wrapped mmdet models builder."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import partial
from typing import Optional, Union

import torch
from mmcv.parallel import DataContainer
from mmcv.runner import CheckpointLoader
from mmcv.utils import Config, ConfigDict
from mmdet.utils import get_root_logger

from otx.algorithms.common.adapters.mmcv.nncf.runners import NNCF_META_KEY
from otx.algorithms.common.adapters.mmcv.utils import (
    get_configs_by_pairs,
    remove_from_configs_by_type,
)
from otx.algorithms.common.adapters.nncf import is_accuracy_aware_training_set
from otx.algorithms.common.adapters.nncf.compression import NNCFMetaState
from otx.algorithms.common.adapters.nncf.utils import no_nncf_trace
from otx.algorithms.detection.adapters.mmdet.utils import build_detector

logger = get_root_logger()


def build_nncf_detector(  # pylint: disable=too-many-locals,too-many-statements
    config: Config,
    train_cfg: Optional[Union[Config, ConfigDict]] = None,
    test_cfg: Optional[Union[Config, ConfigDict]] = None,
    checkpoint: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    cfg_options: Optional[Union[Config, ConfigDict]] = None,
    distributed=False,
):
    """A function to build NNCF wrapped mmdet model."""

    from mmdet.apis import multi_gpu_test, single_gpu_test
    from mmdet.apis.inference import LoadImage
    from mmdet.datasets import build_dataloader as mmdet_build_dataloader
    from mmdet.datasets import build_dataset as mmdet_build_dataset
    from mmdet.datasets.pipelines import Compose
    from nncf.torch.dynamic_graph.io_handling import nncf_model_input

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

    model = build_detector(
        config,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        cfg_options=cfg_options,
        from_scratch=True,
    )
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
                dataset_builder=mmdet_build_dataset,
            ),
            config,
            subset="val",
            dataloader_builder=mmdet_build_dataloader,
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
                    dataset_builder=mmdet_build_dataset,
                ),
                config,
                subset="val",
                dataloader_builder=mmdet_build_dataloader,
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

    test_pipeline = [LoadImage()]
    for pipeline in config.data.test.pipeline:
        if not pipeline.type.startswith("LoadImage"):
            test_pipeline.append(pipeline)
        if pipeline.get("transforms", None):
            transforms = pipeline.transforms
            for transform in transforms:
                if transform.type == "Collect":
                    for collect_key in transform["keys"]:
                        if collect_key != "img":
                            transform["keys"].remove(collect_key)

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
        logger.debug(f"set should_compress_postprocessing={nncf_compress_postprocessing}")
    else:
        # TODO: Do we have to keep this configuration?
        # This configuration is not enabled in forked mmdetection library in the first place
        nncf_compress_postprocessing = True

    def dummy_forward_fn(model):
        def _get_fake_data_for_forward(nncf_config):
            device = next(model.parameters()).device

            if nncf_config.get("input_info", None) and nncf_config.get("input_info").get("sample_size", None):
                input_size = nncf_config.get("input_info").get("sample_size")
                assert len(input_size) == 4 and input_size[0] == 1
                H, W, C = input_size[2], input_size[3], input_size[1]  # pylint: disable=invalid-name
                shape = tuple([H, W, C])
            else:
                shape = (128, 128, 3)

            with no_nncf_trace():
                return get_fake_input_fn(shape=shape, device=device)

        fake_data = _get_fake_data_for_forward(config.nncf_config)
        img, img_metas = fake_data["img"], fake_data["img_metas"]
        ctx = model.nncf_trace_context(img_metas, nncf_compress_postprocessing)
        with ctx:
            # The device where model is could be changed under this context
            img = [i.to(next(model.parameters()).device) for i in img]
            # Marking data as NNCF network input must be after device movement
            img = [nncf_model_input(i) for i in img]
            if nncf_compress_postprocessing:
                logger.debug("NNCF will try to compress a postprocessing part of the model")
            else:
                logger.debug("NNCF will NOT compress a postprocessing part of the model")
                img = img[0]
            model(img)

    compression_ctrl, model = wrap_nncf_model(
        config,
        model,
        model_eval_fn=model_eval_fn,
        dummy_forward_fn=dummy_forward_fn,
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
    # TODO: move this to OTX task when MPA is absorbed into OTX
    remove_from_configs_by_type(custom_hooks, "CancelInterfaceHook")
    remove_from_configs_by_type(custom_hooks, "TaskAdaptHook")
    remove_from_configs_by_type(custom_hooks, "LazyEarlyStoppingHook")
    remove_from_configs_by_type(custom_hooks, "EarlyStoppingHook")
    remove_from_configs_by_type(custom_hooks, "EMAHook")
    remove_from_configs_by_type(custom_hooks, "CustomModelEMAHook")
    config.custom_hooks = custom_hooks

    for hook in get_configs_by_pairs(custom_hooks, dict(type="OTXProgressHook")):
        time_monitor = hook.get("time_monitor", None)
        if time_monitor and getattr(time_monitor, "on_initialization_end", None) is not None:
            time_monitor.on_initialization_end()

    return compression_ctrl, model
