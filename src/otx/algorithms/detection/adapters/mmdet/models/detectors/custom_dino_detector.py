"""OTX DINO Class for mmdetection detectors."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import DETECTORS

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.detection.adapters.mmdet.models.detectors import CustomDeformableDETR
from otx.utils.logger import get_logger

logger = get_logger()


@DETECTORS.register_module()
class CustomDINO(CustomDeformableDETR):
    """Custom DINO detector."""

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, task_adapt=task_adapt, **kwargs)

        self.cls_layers.append("dn_query_generator.label_embedding.weight")

    def load_state_dict_pre_hook(self, model_classes, ckpt_classes, ckpt_dict, *args, **kwargs):
        """Modify mmdet3.x version's weights before weight loading."""

        if list(ckpt_dict.keys())[0] == "level_embed":
            logger.info("----------------- CustomDINO.load_state_dict_pre_hook() called")
            # This ckpt_dict comes from mmdet3.x
            ckpt_classes = [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
            ckpt_dict["bbox_head.transformer.level_embeds"] = ckpt_dict.pop("level_embed")
            replaced_params = {}
            for param in ckpt_dict:
                new_param = None
                if "encoder" in param or "decoder" in param:
                    new_param = "bbox_head.transformer." + param
                    new_param = new_param.replace("self_attn", "attentions.0")
                    new_param = new_param.replace("cross_attn", "attentions.1")
                    new_param = new_param.replace("ffn", "ffns.0")
                elif param == "query_embedding.weight":
                    new_param = "bbox_head." + param
                elif param == "dn_query_generator.label_embedding.weight":
                    new_param = "bbox_head.transformer." + param
                elif "memory_trans" in param:
                    new_param = "bbox_head.transformer." + param
                    new_param = new_param.replace("memory_trans_fc", "enc_output")
                    new_param = new_param.replace("memory_trans_norm", "enc_output_norm")
                if new_param is not None:
                    replaced_params[param] = new_param

            for origin, new in replaced_params.items():
                ckpt_dict[new] = ckpt_dict.pop(origin)
        super().load_state_dict_pre_hook(model_classes, ckpt_classes, ckpt_dict, *args, **kwargs)


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_dino_detector.CustomDINO.simple_test"
    )
    def custom_dino__simple_test(ctx, self, img, img_metas, **kwargs):
        """Function for custom_dino__simple_test."""
        height = int(img_metas[0]["img_shape"][0])
        width = int(img_metas[0]["img_shape"][1])
        img_metas[0]["batch_input_shape"] = (height, width)
        img_metas[0]["img_shape"] = (height, width, 3)
        feats = self.extract_feat(img)
        gt_bboxes = [None] * len(feats)
        gt_labels = [None] * len(feats)
        hidden_states, references, enc_output_class, enc_output_coord, _ = self.bbox_head.forward_transformer(
            feats, gt_bboxes, gt_labels, img_metas
        )
        cls_scores, bbox_preds = self.bbox_head(hidden_states, references)
        bbox_results = self.bbox_head.get_bboxes(
            cls_scores, bbox_preds, enc_output_class, enc_output_coord, img_metas=img_metas, **kwargs
        )

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(feats)
            saliency_map = ActivationMapHook(self).func(cls_scores)
            return (*bbox_results, feature_vector, saliency_map)

        return bbox_results
