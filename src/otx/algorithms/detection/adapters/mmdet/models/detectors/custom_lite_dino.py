"""OTX Lite-DINO Class for object detection."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import DETECTORS

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.detection.adapters.mmdet.models.detectors import CustomDINO

logger = get_logger()


@DETECTORS.register_module()
class CustomLiteDINO(CustomDINO):
    """Custom Lite-DINO <https://arxiv.org/pdf/2303.07335.pdf> for object detection."""

    def load_state_dict_pre_hook(self, model_classes, ckpt_classes, ckpt_dict, *args, **kwargs):
        """Modify official lite dino version's weights before weight loading."""
        if list(ckpt_dict.keys())[0] == "transformer.level_embed":
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
                "street sign",
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
                "hat",
                "backpack",
                "umbrella",
                "shoe",
                "eyeglasses",
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
                "plate",
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
                "mirror",
                "dining table",
                "window",
                "desk",
                "toilet",
                "door",
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
                "blender",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
                "hairbrush",
            ]
            logger.info("----------------- CustomLiteDINO.load_state_dict_pre_hook() called")
            ckpt_dict["bbox_head.transformer.level_embeds"] = ckpt_dict.pop("transformer.level_embed")
            ckpt_dict["bbox_head.transformer.dn_query_generator.label_embedding.weight"] = ckpt_dict.pop(
                "label_enc.weight"
            )
            ckpt_dict["bbox_head.query_embedding.weight"] = ckpt_dict.pop("transformer.tgt_embed.weight")
            replaced_params = {}
            unused_params = []
            for param in ckpt_dict:
                new_param = None
                if "backbone.0.body" in param:
                    new_param = param.replace("0.body.", "")
                elif "encoder" in param or "decoder" in param:
                    new_param = "bbox_head." + param
                    if "encoder" in param:
                        new_param = new_param.replace("self_attn", "attentions.0")
                    else:
                        new_param = new_param.replace("self_attn", "attentions.0.attn")
                    new_param = new_param.replace("cross_attn", "attentions.1")
                    new_param = new_param.replace("norm1", "norms.0")
                    if "norm22" in new_param:
                        new_param = new_param.replace("norm22", "ffns.0.norm2")
                    elif "norm2" in new_param:
                        if param.replace("norm2", "norm22") in ckpt_dict:
                            new_param = new_param.replace("norm2", "ffns.0.norm1")
                        else:
                            new_param = new_param.replace("norm2", "norms.1")
                    new_param = new_param.replace("norm3", "norms.2")
                    new_param = new_param.replace("transformer.decoder.class_embed", "cls_branches")
                    new_param = new_param.replace("transformer.decoder.bbox_embed", "reg_branches")
                    if "reg_branches" in new_param:
                        new_param = new_param.replace("layers.0", "0")
                        new_param = new_param.replace("layers.1", "2")
                        new_param = new_param.replace("layers.2", "4")
                    elif "cls_branches" in new_param:
                        new_param = new_param.replace("0.layers.0", "0.0")
                        new_param = new_param.replace("0.layers.1", "0.2")
                        new_param = new_param.replace("0.layers.2", "0.4")
                    new_param = new_param.replace("linear12", "ffns.0.small_expand_layers.0.0")
                    new_param = new_param.replace("linear22", "ffns.0.small_expand_layers.1")
                    new_param = new_param.replace("linear1", "ffns.0.layers.0.0")
                    new_param = new_param.replace("linear2", "ffns.0.layers.1")
                elif "input_proj" in param:
                    new_param = param.replace("input_proj.0.0", "neck.convs.0.conv")
                    new_param = new_param.replace("input_proj.0.1", "neck.convs.0.gn")
                    new_param = new_param.replace("input_proj.1.0", "neck.convs.1.conv")
                    new_param = new_param.replace("input_proj.1.1", "neck.convs.1.gn")
                    new_param = new_param.replace("input_proj.2.0", "neck.convs.2.conv")
                    new_param = new_param.replace("input_proj.2.1", "neck.convs.2.gn")
                    new_param = new_param.replace("input_proj.3.0", "neck.extra_convs.0.conv")
                    new_param = new_param.replace("input_proj.3.1", "neck.extra_convs.0.gn")
                elif "enc_output" in param:
                    new_param = "bbox_head." + param
                elif "transformer.enc_out" in param:
                    new_param = param.replace("transformer.enc_out_class_embed", "bbox_head.cls_branches.6")
                    new_param = new_param.replace(
                        "transformer.enc_out_bbox_embed.layers.0", "bbox_head.reg_branches.6.0"
                    )
                    new_param = new_param.replace(
                        "transformer.enc_out_bbox_embed.layers.1", "bbox_head.reg_branches.6.2"
                    )
                    new_param = new_param.replace(
                        "transformer.enc_out_bbox_embed.layers.2", "bbox_head.reg_branches.6.4"
                    )
                elif "_embed." in param:
                    unused_params.append(param)

                if new_param is not None:
                    replaced_params[param] = new_param

            for origin, new in replaced_params.items():
                ckpt_dict[new] = ckpt_dict.pop(origin)
            for param in unused_params:
                ckpt_dict.pop(param)
            breakpoint()
        super(CustomDINO, self).load_state_dict_pre_hook(model_classes, ckpt_classes, ckpt_dict, *args, *kwargs)
