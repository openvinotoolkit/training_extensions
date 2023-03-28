"""Inference task for OTX Detection with MMDET."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from contextlib import nullcontext

from mmcv.utils import Config, ConfigDict
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader as mmdet_build_dataloader
from mmdet.datasets import build_dataset as mmdet_build_dataset
from mmdet.datasets import replace_ImageToTensor
from mmdet.models.detectors import TwoStageDetector

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmcv.tasks.registry import STAGES
from otx.algorithms.common.adapters.mmcv.utils import (
    build_data_parallel,
    build_dataloader,
    build_dataset,
)
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.datasets import ImageTilingDataset
from otx.algorithms.detection.adapters.mmdet.hooks.det_saliency_map_hook import (
    DetSaliencyMapHook,
)

from .stage import DetectionStage

logger = get_logger()


@STAGES.register_module()
class DetectionInferrer(DetectionStage):
    """Class for object detection inference."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = None

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage for detection.

        - Configuration
        - Environment setup
        - Run inference via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get("mode", "train")
        if mode not in self.mode:
            logger.warning(f"Supported modes are {self.mode} but '{mode}' is given.")
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        logger.info("infer!")

        model_builder = kwargs.get("model_builder", None)
        dump_features = kwargs.get("dump_features", False)
        dump_saliency_map = kwargs.get("dump_saliency_map", False)
        do_eval = kwargs.get("eval", False)
        outputs = self.infer(
            cfg,
            model_builder=model_builder,
            do_eval=do_eval,
            dump_features=dump_features,
            dump_saliency_map=dump_saliency_map,
        )

        # Save outputs
        # output_file_path = osp.join(cfg.work_dir, 'infer_result.npy')
        # np.save(output_file_path, outputs, allow_pickle=True)
        return dict(
            # output_file_path=output_file_path,
            outputs=outputs
        )

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def infer(self, cfg, model_builder=None, do_eval=False, dump_features=False, dump_saliency_map=False):
        """Main inference function."""
        # TODO: distributed inference

        data_cfg = cfg.data.test.copy()

        # Input source
        if "input_source" in cfg:
            input_source = cfg.get("input_source")
            logger.info(f"Inferring on input source: data.{input_source}")
            if input_source == "train":
                src_data_cfg = self.get_data_cfg(cfg, input_source)
            else:
                src_data_cfg = cfg.data[input_source]
            data_cfg.test_mode = src_data_cfg.get("test_mode", False)
            data_cfg.ann_file = src_data_cfg.ann_file
            data_cfg.img_prefix = src_data_cfg.img_prefix
            if "classes" in src_data_cfg:
                data_cfg.classes = src_data_cfg.classes

        data_cfg = Config(
            ConfigDict(
                data=ConfigDict(
                    samples_per_gpu=cfg.data.get("samples_per_gpu", 1),
                    workers_per_gpu=cfg.data.get("workers_per_gpu", 0),
                    test=data_cfg,
                    test_dataloader=cfg.data.get("test_dataloader", {}).copy(),
                ),
                gpu_ids=cfg.gpu_ids,
                seed=cfg.get("seed", None),
                model_task=cfg.model_task,
            )
        )
        self.configure_samples_per_gpu(data_cfg, "test", distributed=False)
        self.configure_compat_cfg(data_cfg)
        samples_per_gpu = data_cfg.data.test_dataloader.get("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            data_cfg.data.test.pipeline = replace_ImageToTensor(data_cfg.data.test.pipeline)

        # Data loader
        self.dataset = build_dataset(data_cfg, "test", mmdet_build_dataset)
        test_dataloader = build_dataloader(
            self.dataset,
            data_cfg,
            "test",
            mmdet_build_dataloader,
            distributed=False,
        )

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(
                    f"target_classes={target_classes} is empty check the metadata from model ckpt or recipe "
                    "configuration"
                )
        else:
            target_classes = self.dataset.CLASSES

        # Model
        cfg.model.pretrained = None
        # TODO: Check Inference FP16 Support
        model = self.build_model(cfg, model_builder, fp16=False)
        model.CLASSES = target_classes
        model.eval()
        feature_model = self._get_feature_module(model)
        model = build_data_parallel(model, cfg, distributed=False)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        self.set_inference_progress_callback(model, cfg)

        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        if not dump_saliency_map:
            saliency_hook = nullcontext()
        else:
            raw_model = feature_model
            if raw_model.__class__.__name__ == "NNCFNetwork":
                raw_model = raw_model.get_nncf_wrapped_model()
            if isinstance(raw_model, TwoStageDetector):
                saliency_hook = ActivationMapHook(feature_model)
            else:
                saliency_hook = DetSaliencyMapHook(feature_model)

        # pylint: disable=no-member
        eval_predictions = []
        with FeatureVectorHook(feature_model) if dump_features else nullcontext() as feature_vector_hook:
            with saliency_hook:
                eval_predictions = single_gpu_test(model, test_dataloader)
                feature_vectors = feature_vector_hook.records if dump_features else [None] * len(self.dataset)
                saliency_maps = saliency_hook.records if dump_saliency_map else [None] * len(self.dataset)

        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
            cfg.evaluation.pop(key, None)

        metric = None
        if do_eval:
            metric = self.dataset.evaluate(eval_predictions, **cfg.evaluation)
            metric = metric["mAP"] if isinstance(cfg.evaluation.metric, list) else metric[cfg.evaluation.metric]

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
        dataset = self.dataset
        while hasattr(dataset, "dataset") and not isinstance(dataset, ImageTilingDataset):
            dataset = dataset.dataset

        if isinstance(dataset, ImageTilingDataset):
            feature_vectors = [feature_vectors[i] for i in range(dataset.num_samples)]
            saliency_maps = [saliency_maps[i] for i in range(dataset.num_samples)]
            if not dataset.merged_results:
                eval_predictions = dataset.merge(eval_predictions)
            else:
                eval_predictions = dataset.merged_results

        assert len(eval_predictions) == len(feature_vectors) == len(saliency_maps), (
            "Number of elements should be the same, however, number of outputs are "
            f"{len(eval_predictions)}, {len(feature_vectors)}, and {len(saliency_maps)}"
        )

        outputs = dict(
            classes=target_classes,
            detections=eval_predictions,
            metric=metric,
            feature_vectors=feature_vectors,
            saliency_maps=saliency_maps,
        )
        return outputs
