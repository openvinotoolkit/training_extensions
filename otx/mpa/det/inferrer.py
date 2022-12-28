# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from contextlib import nullcontext

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import load_checkpoint
from mmdet.datasets import (
    ImageTilingDataset,
    build_dataloader,
    build_dataset,
    replace_ImageToTensor,
)
from mmdet.models import build_detector
from mmdet.models.detectors import TwoStageDetector
from mmdet.utils.misc import prepare_mmdet_model_for_execution

from otx.mpa.det.incremental import IncrDetectionStage
from otx.mpa.modules.hooks.recording_forward_hooks import (
    ActivationMapHook,
    DetSaliencyMapHook,
    FeatureVectorHook,
)
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class DetectionInferrer(IncrDetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run inference stage for detection

        - Configuration
        - Environment setup
        - Run inference via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get("mode", "train")
        eval = kwargs.pop("eval", False)
        dump_features = kwargs.pop("dump_features", False)
        dump_saliency_map = kwargs.pop("dump_saliency_map", False)
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        # cfg.dump(osp.join(cfg.work_dir, 'config.py'))
        # logger.info(f'Config:\n{cfg.pretty_text}')
        # logger.info('infer!')

        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        outputs = self.infer(cfg, eval=eval, dump_features=dump_features, dump_saliency_map=dump_saliency_map)

        # Save outputs
        # output_file_path = osp.join(cfg.work_dir, 'infer_result.npy')
        # np.save(output_file_path, outputs, allow_pickle=True)
        return dict(
            # output_file_path=output_file_path,
            outputs=outputs
        )
        # TODO: save in json
        """
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        a = np.array([[1, 2, 3], [4, 5, 6]])
        json_dump = json.dumps({'a': a, 'aa': [2, (2, 3, 4), a], 'bb': [2]},
                               cls=NumpyEncoder)
        print(json_dump)
        """

    # noqa: C901
    def infer(self, cfg, eval=False, dump_features=False, dump_saliency_map=False):
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

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
        self.dataset = build_dataset(data_cfg)
        dataset = self.dataset

        # Data loader
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )

        # Target classes
        if "task_adapt" in cfg:
            target_classes = cfg.task_adapt.final
            if len(target_classes) < 1:
                raise KeyError(
                    f"target_classes={target_classes} is empty check the metadata from model ckpt or recipe "
                    f"configuration"
                )
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

        model = build_detector(cfg.model)
        model.CLASSES = target_classes

        # TODO: Check Inference FP16 Support
        # fp16_cfg = cfg.get('fp16', None)
        # if fp16_cfg is not None:
        #     wrap_fp16_model(model)

        # InferenceProgressCallback (Time Monitor enable into Infer task)
        self.set_inference_progress_callback(model, cfg)

        # Checkpoint
        if cfg.get("load_from", None):
            load_checkpoint(model, cfg.load_from, map_location="cpu")

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        eval_model = prepare_mmdet_model_for_execution(model, cfg, self.distributed)

        # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
        if is_module_wrapper(model):
            model = model.module

        # Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        if not dump_saliency_map:
            saliency_hook = nullcontext()
        elif isinstance(model, TwoStageDetector):
            saliency_hook = ActivationMapHook(eval_model.module)
        else:
            saliency_hook = DetSaliencyMapHook(eval_model.module)

        eval_predictions = []
        with FeatureVectorHook(eval_model.module) if dump_features else nullcontext() as feature_vector_hook:
            with saliency_hook:
                for data in data_loader:
                    with torch.no_grad():
                        result = eval_model(return_loss=False, rescale=True, **data)
                    eval_predictions.extend(result)
                feature_vectors = feature_vector_hook.records if dump_features else [None] * len(self.dataset)
                saliency_maps = saliency_hook.records if dump_saliency_map else [None] * len(self.dataset)

        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
            cfg.evaluation.pop(key, None)

        metric = None
        if eval:
            metric = dataset.evaluate(eval_predictions, **cfg.evaluation)
            metric = metric["mAP"] if isinstance(cfg.evaluation.metric, list) else metric[cfg.evaluation.metric]

        # Check and unwrap ImageTilingDataset object from TaskAdaptEvalDataset
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
