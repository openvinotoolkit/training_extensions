# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp
from contextlib import nullcontext

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.parallel import MMDataCPU

from otx.mpa.modules.hooks.recording_forward_hooks import FeatureVectorHook
from otx.mpa.registry import STAGES
from otx.mpa.stage import Stage

from .stage import SegStage


@STAGES.register_module()
class SegInferrer(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage for segmentation

        - Configuration
        - Environment setup
        - Run inference via MMSegmentation -> MMCV
        """
        self._init_logger()
        dump_features = kwargs.get("dump_features", False)
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        self.logger.info("infer!")

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        outputs = self.infer(cfg, dump_features)

        return dict(outputs=outputs)

    def infer(self, cfg, dump_features=False):
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        # Input source
        input_source = cfg.get("input_source", "test")
        self.logger.info(f"Inferring on input source: data.{input_source}")
        if input_source == "train":
            src_data_cfg = Stage.get_train_data_cfg(cfg)
        else:
            src_data_cfg = cfg.data[input_source]
        data_cfg = cfg.data.test.copy()
        # data_cfg.ann_file = src_data_cfg.ann_file
        # data_cfg.img_prefix = src_data_cfg.img_prefix
        if "classes" in src_data_cfg:
            data_cfg.classes = src_data_cfg.classes
            data_cfg.new_classes = []
        self.dataset = build_dataset(data_cfg)
        dataset = self.dataset

        # Data loader
        mm_val_dataloader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
        else:
            target_classes = dataset.CLASSES

        # Model
        cfg.model.pretrained = None
        if cfg.model.get("neck"):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get("rfp_backbone"):
                        if neck_cfg.rfp_backbone.get("pretrained"):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get("rfp_backbone"):
                if cfg.model.neck.rfp_backbone.get("pretrained"):
                    cfg.model.neck.rfp_backbone.pretrained = None
        cfg.model.test_cfg.return_repr_vector = True
        model = build_segmentor(cfg.model, train_cfg=None, test_cfg=None)
        model.CLASSES = target_classes

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        # Checkpoint
        if cfg.get("load_from", None):
            _ = load_checkpoint(model, cfg.load_from, map_location="cpu")

        # Inference
        model.eval()
        if torch.cuda.is_available():
            if self.distributed:
                model = model.cuda()
                find_unused_parameters = cfg.get("find_unused_parameters", False)
                # Sets the `find_unused_parameters` parameter in
                # torch.nn.parallel.DistributedDataParallel
                model = MMDistributedDataParallel(
                    model,
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters,
                )
            else:
                model = MMDataParallel(model.cuda(), device_ids=[0])
        else:
            model = MMDataCPU(model)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        SegStage.set_inference_progress_callback(model, cfg)

        eval_predictions = []
        feature_vectors = []
        with FeatureVectorHook(model.module) if dump_features else nullcontext() as fhook:
            for data in mm_val_dataloader:
                with torch.no_grad():
                    result = model(return_loss=False, output_logits=True, **data)
                eval_predictions.append(result)
            feature_vectors = fhook.records if dump_features else [None] * len(self.dataset)

        assert len(eval_predictions) == len(feature_vectors), (
            "Number of elements should be the same, however, number of outputs are ",
            f"{len(eval_predictions)} and {len(feature_vectors)}",
        )

        outputs = dict(
            classes=target_classes,
            eval_predictions=eval_predictions,
            feature_vectors=feature_vectors,
        )
        return outputs


import copy  # noqa: E402
import warnings  # noqa: E402


def replace_ImageToTensor(pipelines):
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline["type"] == "MultiScaleFlipAug":
            assert "transforms" in pipeline
            pipeline["transforms"] = replace_ImageToTensor(pipeline["transforms"])
        elif pipeline["type"] == "ImageToTensor":
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                "recommended to manually replace it in the test "
                "data pipeline in your config file.",
                UserWarning,
            )
            pipelines[i] = {"type": "DefaultFormatBundle"}
    return pipelines
