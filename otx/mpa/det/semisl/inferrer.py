# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import List, Tuple

import torch
from mmcv.parallel import MMDataParallel, is_module_wrapper
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.parallel import MMDataCPU
from mmdet.utils.deployment import get_feature_vector, get_saliency_map

from otx.mpa.det.semisl.stage import SemiSLDetectionStage
from otx.mpa.registry import STAGES
from otx.mpa.utils.logger import get_logger

logger = get_logger()


@STAGES.register_module()
class SemiSLDetectionInferrer(SemiSLDetectionStage):
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
        eval = kwargs.get("eval", False)
        dump_features = kwargs.get("dump_features", False)
        dump_saliency_map = kwargs.get("dump_saliency_map", False)
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
                src_data_cfg = self.get_data_cfg(cfg, "train")
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
            eval_model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        else:
            eval_model = MMDataCPU(model)

        eval_predictions = []
        feature_vectors = []
        saliency_maps = []

        def dump_features_hook(mod, inp, out):
            with torch.no_grad():
                feature_vector = get_feature_vector(out)
                assert feature_vector.size(0) == 1
            feature_vectors.append(feature_vector.view(-1).detach().cpu().numpy())

        def dummy_dump_features_hook(mod, inp, out):
            feature_vectors.append(None)

        def dump_saliency_hook(model: torch.nn.Module, input: Tuple, out: List[torch.Tensor]):
            """Dump the last feature map to `saliency_maps` cache

            Args:
                model (torch.nn.Module): PyTorch model
                input (Tuple): input
                out (List[torch.Tensor]): a list of feature maps
            """
            with torch.no_grad():
                saliency_map = get_saliency_map(out[-1])
            saliency_maps.append(saliency_map.squeeze(0).detach().cpu().numpy())

        def dummy_dump_saliency_hook(model, input, out):
            saliency_maps.append(None)

        feature_vector_hook = dump_features_hook if dump_features else dummy_dump_features_hook
        saliency_map_hook = dump_saliency_hook if dump_saliency_map else dummy_dump_saliency_hook

        with eval_model.module.model_t.backbone.register_forward_hook(feature_vector_hook):
            with eval_model.module.model_t.backbone.register_forward_hook(saliency_map_hook):
                for data in data_loader:
                    with torch.no_grad():
                        result = eval_model(return_loss=False, rescale=True, **data)
                    eval_predictions.extend(result)

        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
            cfg.evaluation.pop(key, None)

        metric = None
        if eval:
            metric = dataset.evaluate(eval_predictions, **cfg.evaluation)[cfg.evaluation.metric]

        outputs = dict(
            classes=target_classes,
            detections=eval_predictions,
            metric=metric,
            feature_vectors=feature_vectors,
            saliency_maps=saliency_maps,
        )
        return outputs
