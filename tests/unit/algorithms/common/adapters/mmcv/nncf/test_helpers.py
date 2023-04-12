# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import partial

import numpy as np
import torch
from mmcls.core import EvalHook
from mmcls.datasets import build_dataloader
from mmcls.datasets.pipelines import Compose
from mmcv.utils import Config, ConfigDict, get_logger
from torch.optim import SGD

import otx.algorithms.common.adapters.mmcv.nncf.patches  # noqa: F401
from otx.algorithms.common.adapters.mmcv.nncf.hooks import CompressionHook
from otx.algorithms.common.adapters.mmcv.nncf.runners import AccuracyAwareRunner
from otx.algorithms.common.adapters.mmcv.nncf.utils import (
    get_fake_input,
    model_eval,
    wrap_nncf_model,
)
from otx.algorithms.common.adapters.mmcv.utils import build_data_parallel
from otx.algorithms.common.adapters.nncf.patches import nncf_trace_context


def create_model(lib="mmcls"):
    class MockModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, img, img_metas=None, **kwargs):
            if isinstance(img, list):
                img = img[0]
            if img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)

            x = self.conv1(img)
            x = self.conv2(img)
            x = torch.mean(x, dim=(2, 3))
            x = self.linear(x)
            return x

        def forward_dummy(self, img, img_metas=None, **kwargs):
            if isinstance(img, list):
                img = img[0]
            if img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)

            x = self.conv1(img)
            x = self.conv2(img)
            x = torch.mean(x, dim=(2, 3))
            x = self.linear(x)
            return x

        def train_step(self, *args, **kwargs):
            return dict()

        def init_weights(self, *args, **kwargs):
            pass

        def set_step_params(self, *args, **kwargs):
            pass

    MockModel.nncf_trace_context = nncf_trace_context

    if lib == "mmcls":
        from mmcls.models import CLASSIFIERS

        CLASSIFIERS.register_module(MockModel, force=True)
    elif lib == "mmdet":
        from mmdet.models import DETECTORS

        DETECTORS.register_module(MockModel, force=True)
    elif lib == "mmseg":
        from mmseg.models import SEGMENTORS

        SEGMENTORS.register_module(MockModel, force=True)
    else:
        raise ValueError()

    return MockModel()


def create_dataset(pipeline=None, lib="mmcls"):
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, pipeline, *args, **kwargs):
            super().__init__()
            self.dataset = np.zeros((10, 128, 128, 3), dtype=np.float32)

            if isinstance(pipeline, list):
                if lib == "mmcls":
                    from mmcls.datasets.pipelines import Compose
                elif lib == "mmdet":
                    from mmdet.datasets.pipelines import Compose
                elif lib == "mmseg":
                    from mmseg.datasets.pipelines import Compose
                else:
                    raise ValueError()
                pipeline = Compose(pipeline)
            self.pipeline = pipeline

        def __getitem__(self, idx):
            if self.pipeline:
                return self.pipeline(
                    dict(
                        img=self.dataset[idx],
                        filename=None,
                        ori_filename=None,
                        img_fields=["img"],
                        img_shape=self.dataset[idx].shape,
                        ori_shape=self.dataset[idx].shape,
                    )
                )
            return self.dataset[idx]

        def __len__(self):
            return len(self.dataset)

        def evaluate(self, model_output, **kwargs):
            #  if isinstance(model_output, list):
            #      model_output = torch.cat(model_output)
            #  return {"accuracy": model_output.mean()}
            return {"accuracy": torch.tensor(0.9)}

    if lib == "mmcls":
        from mmcls.datasets import DATASETS
    elif lib == "mmdet":
        from mmdet.datasets import DATASETS
    elif lib == "mmseg":
        from mmseg.datasets import DATASETS
    else:
        raise ValueError()

    DATASETS.register_module(MockDataset, force=True)
    return MockDataset(pipeline)


def create_dataloader(config=None):
    if config is None:
        config = create_config()
    pipeline = Compose(config.data.val.pipeline)
    mock_dataset = create_dataset(pipeline)
    dataloader = build_dataloader(mock_dataset, samples_per_gpu=2, workers_per_gpu=1)
    return dataloader


def create_config(lib="mmcls"):
    config = Config(
        {
            "gpu_ids": [0],
            "evaluation": {"metric": "accuracy"},
            "data": {
                "samples_per_gpu": 2,
                "workers_per_gpu": 0,
                "persistent_workers": False,
                "shuffle": False,
            },
            "model": {
                "type": "MockModel",
            },
            "nncf_config": {
                "input_info": {"sample_size": (1, 3, 128, 128)},
                "target_metric_name": "accuracy",
                "compression": [
                    {
                        "algorithm": "quantization",
                        "preset": "mixed",
                        "initializer": {
                            "range": {"num_init_samples": 10},
                            "batchnorm_adaptation": {"num_bn_adaptation_samples": 10},
                        },
                    }
                ],
                "accuracy_aware_training": {"params": {"maximal_total_epochs": 5, "mode": "early_exit"}},
            },
            "runner": {
                "type": "AccuracyAwareRunner",
                "nncf_config": {
                    "input_info": {"sample_size": (1, 3, 128, 128)},
                    "target_metric_name": "accuracy",
                    "compression": [
                        {
                            "algorithm": "quantization",
                            "preset": "mixed",
                            "initializer": {
                                "range": {"num_init_samples": 10},
                                "batchnorm_adaptation": {"num_bn_adaptation_samples": 10},
                            },
                        }
                    ],
                    "accuracy_aware_training": {"params": {"maximal_total_epochs": 5, "mode": "early_exit"}},
                },
            },
        }
    )

    for subset in ("train", "test", "val"):
        if lib == "mmcls":
            config.data[subset] = ConfigDict(
                {
                    "type": "MockDataset",
                    "pipeline": [
                        {"type": "Resize", "size": (50, 50)},
                        {"type": "Normalize", "mean": [0, 0, 0], "std": [1, 1, 1]},
                        {"type": "ImageToTensor", "keys": ["img"]},
                        {
                            "type": "Collect",
                            "keys": ["img"],
                            "meta_keys": [
                                "filename",
                                "ori_filename",
                                "ori_shape",
                                "img_shape",
                                "img_norm_cfg",
                            ],
                        },
                    ],
                }
            )
        else:
            config.data[subset] = ConfigDict(
                {
                    "type": "MockDataset",
                    "pipeline": [
                        {"type": "Resize", "img_scale": (50, 50)},
                        {"type": "Normalize", "mean": [0, 0, 0], "std": [1, 1, 1]},
                        {"type": "ImageToTensor", "keys": ["img"]},
                        {
                            "type": "Collect",
                            "keys": ["img"],
                            "meta_keys": [
                                "filename",
                                "ori_filename",
                                "ori_shape",
                                "img_shape",
                                "img_norm_cfg",
                            ],
                        },
                    ],
                }
            )
    return config


def create_eval_fn():
    def evaluate_fn(model, loader, *args, **kwargs):
        out = []
        for data in loader:
            out.append(torch.sigmoid(model(**data)))
        return torch.cat(out)

    return evaluate_fn


def create_nncf_model(workdir):
    mock_model = create_model()
    mock_config = create_config()
    mock_eval_fn = create_eval_fn()
    dataloader = create_dataloader()

    mock_config = create_config()
    mock_config.nncf_config.log_dir = workdir
    pipeline = Compose(mock_config.data.val.pipeline)
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
    return ctrl, model


def create_nncf_runner(work_dir):
    mock_config = create_config()
    dataloader = create_dataloader()
    ctrl, model = create_nncf_model(work_dir)

    runner = AccuracyAwareRunner(
        build_data_parallel(model, mock_config),
        logger=get_logger("mmcv"),
        work_dir=work_dir,
        nncf_config=mock_config["nncf_config"],
        optimizer=SGD(model.parameters(), lr=0.01),
    )
    runner.register_hook(
        EvalHook(
            dataloader,
            save_best="accuracy",
            file_client_args={"backend": "disk"},
            priority="LOW",
        )
    )
    runner.register_hook(CompressionHook(ctrl))
    return runner
