# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from functools import partial

from mmcv.runner import CheckpointLoader, load_state_dict
from mmcv.utils import Config, ConfigDict
from mmseg.utils import get_root_logger

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
from otx.algorithms.common.adapters.nncf.utils import load_checkpoint


logger = get_root_logger()


def build_nncf_model(config, distributed=False):
    from mmseg.apis import multi_gpu_test, single_gpu_test
    from mmseg.apis.inference import LoadImage
    from mmseg.datasets import build_dataloader as mmseg_build_dataloader
    from mmseg.datasets import build_dataset
    from mmseg.datasets.pipelines import Compose
    from mmseg.models import build_segmentor

    from otx.algorithms.common.adapters.mmcv.nncf import (
        build_dataloader,
        get_fake_input,
        model_eval,
        wrap_nncf_model,
    )

    def build_model(config: Config, from_scratch: bool = False):
        """Creates a model, based on the configuration in config.

        :param config: mmsegmentation configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: ModelEntity in training mode
        """

        model_cfg = deepcopy(config.model)

        init_from = None if from_scratch else config.get("load_from", None)
        logger.warning(f"Init from: {init_from}")

        if init_from is not None:
            # No need to initialize backbone separately, if all weights are provided.
            model_cfg.pretrained = None
            logger.warning("build segmentor")
            model = build_segmentor(model_cfg)

            # Load all weights.
            logger.warning("load checkpoint")
            load_checkpoint(model, init_from, map_location="cpu")
        else:
            logger.warning("build segmentor")
            model = build_segmentor(model_cfg)

        return model

    state_dict = CheckpointLoader.load_checkpoint(config.load_from, map_location="cpu")
    # load model in this function not in runner
    config.load_from = None

    model = build_model(config, from_scratch=True)

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
            dataloader_builder=mmseg_build_dataloader,
            dataset_builder=build_dataset,
        )

        val_dataloader = None
        if is_acc_aware:
            val_dataloader = build_dataloader(
                config,
                subset="val",
                distributed=distributed,
                dataloader_builder=mmseg_build_dataloader,
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

    compression_ctrl, model = wrap_nncf_model(
        config,
        model,
        init_state_dict=state_dict,
        model_eval_fn=model_eval_fn,
        get_fake_input_fn=get_fake_input_fn,
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
    remove_from_configs_by_type(custom_hooks, "AdaptiveTrainSchedulingHook")

    for hook in get_configs_by_dict(custom_hooks, dict(type="OTXProgressHook")):
        time_monitor = hook.get("time_monitor", None)
        if (
            time_monitor
            and getattr(time_monitor, "on_initialization_end", None) is not None
        ):
            time_monitor.on_initialization_end()

    return compression_ctrl, model
