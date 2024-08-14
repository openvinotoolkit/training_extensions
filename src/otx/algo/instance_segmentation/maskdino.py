from typing import Tuple

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.config import configurable, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList
from torch import Tensor, nn
from torch.nn.modules import Module
from torchvision import tv_tensors

from otx.algo.instance_segmentation.mask_dino import box_ops
from otx.algo.instance_segmentation.mask_dino.criterion import SetCriterion
from otx.algo.instance_segmentation.mask_dino.maskdino_head import MaskDINOHead
from otx.algo.instance_segmentation.mask_dino.matcher import HungarianMatcher
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel
from otx.core.utils.mask_util import polygon_to_bitmap


class MaskDINO(nn.Module):
    """Main class for mask classification semantic segmentation architectures."""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
    ):
        """Args:
        backbone: a backbone module, must follow detectron2's backbone interface
        sem_seg_head: a module that predicts semantic segmentation from backbone features
        criterion: a module that defines the loss
        num_queries: int, number of queries
        object_mask_threshold: float, threshold to filter query based on classification score
        for panoptic segmentation inference
        overlap_threshold: overlap threshold used in general inference for panoptic segmentation
        metadata: dataset meta, get `thing` and `stuff` category names for panoptic
        segmentation inference
        size_divisibility: Some backbones require the input height and width to be divisible by a
        specific integer. We can use this to override such requirement.
        sem_seg_postprocess_before_inference: whether to resize the prediction back
        to original input size before semantic segmentation inference or after.
        For high-resolution dataset like Mapillary, resizing predictions before
        inference will cause OOM error.
        pixel_mean, pixel_std: list or tuple with #channels element, representing
        the per-channel mean and std to be used to normalize the input image
        semantic_on: bool, whether to output semantic segmentation prediction
        instance_on: bool, whether to output instance segmentation prediction
        panoptic_on: bool, whether to output panoptic segmentation prediction
        test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        transform_eval: transform sigmoid score into softmax score to make score sharper
        semantic_ce_loss: whether use cross-entroy loss in classification
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        print("criterion.weight_dict ", self.criterion.weight_dict)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        sem_seg_head = MaskDINOHead(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox": box_weight, "loss_giou": giou_weight})
        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + "_interm": v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + "_dn": v for k, v in weight_dict.items() if k != "loss_mask" and k != "loss_dice"})
            dn_losses = ["labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + "_dn": v for k, v in weight_dict.items()})
            dn_losses = ["labels", "masks", "boxes"]
        else:
            dn_losses = []
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks", "boxes"]
        else:
            losses = ["labels", "masks"]
        # building criterion
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
            and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS
            and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
                or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MaskDINO.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL,
            "pano_temp": cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
            and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS
            and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
        }

    def forward(self, entity: InstanceSegBatchDataEntity):
        img_shapes = [img_info.img_shape for img_info in entity.imgs_info]
        images = ImageList(entity.images, img_shapes)

        features = self.backbone(images.tensor)

        if self.training:
            targets = []
            for img_info, bboxes, labels, masks, polygons in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
                entity.masks,
                entity.polygons,
            ):
                masks = polygon_to_bitmap(polygons, *img_info.img_shape)
                masks = tv_tensors.Mask(masks, device=img_info.device, dtype=torch.bool)
                norm_shape = torch.tile(torch.tensor(img_info.img_shape, device=img_info.device), (2,))

                targets.append(
                    {
                        "boxes": box_ops.box_xyxy_to_cxcywh(bboxes) / norm_shape,
                        "labels": labels,
                        "masks": masks,
                    },
                )

            outputs, mask_dict = self.sem_seg_head(features, targets=targets)
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, mask_dict)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses

        outputs, _ = self.sem_seg_head(features)
        return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor) / image_size_xyxy,
                },
            )
        return new_targets

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes


class MaskDINOR50(ExplainableOTXInstanceSegModel):
    load_from = "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth"

    @staticmethod
    def add_maskdino_config(cfg):
        """Add config for MaskDINO."""
        # NOTE: configs from original mask2former
        # data config
        # select the dataset mapper
        cfg.INPUT.DATASET_MAPPER_NAME = "MaskDINO_semantic"
        # Color augmentation
        cfg.INPUT.COLOR_AUG_SSD = False
        # We retry random cropping until no single category in semantic segmentation GT occupies more
        # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
        # Pad image and segmentation GT in dataset mapper.
        cfg.INPUT.SIZE_DIVISIBILITY = -1

        # solver config
        # weight decay on embedding
        cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
        # optimizer
        cfg.SOLVER.OPTIMIZER = "ADAMW"
        cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

        # MaskDINO model config
        cfg.MODEL.MaskDINO = CN()
        cfg.MODEL.MaskDINO.LEARN_TGT = False

        # loss
        cfg.MODEL.MaskDINO.PANO_BOX_LOSS = False
        cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS = False
        cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
        cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = 0.1
        cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
        cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
        cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
        cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.0
        cfg.MODEL.MaskDINO.GIOU_WEIGHT = 2.0

        # cost weight
        cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT = 4.0
        cfg.MODEL.MaskDINO.COST_DICE_WEIGHT = 5.0
        cfg.MODEL.MaskDINO.COST_MASK_WEIGHT = 5.0
        cfg.MODEL.MaskDINO.COST_BOX_WEIGHT = 5.0
        cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT = 2.0

        # transformer config
        cfg.MODEL.MaskDINO.NHEADS = 8
        cfg.MODEL.MaskDINO.DROPOUT = 0.1
        cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048
        cfg.MODEL.MaskDINO.ENC_LAYERS = 0
        cfg.MODEL.MaskDINO.DEC_LAYERS = 6
        cfg.MODEL.MaskDINO.INITIAL_PRED = True
        cfg.MODEL.MaskDINO.PRE_NORM = False
        cfg.MODEL.MaskDINO.BOX_LOSS = True
        cfg.MODEL.MaskDINO.HIDDEN_DIM = 256
        cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100

        cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
        cfg.MODEL.MaskDINO.TWO_STAGE = True
        cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = "no"  # ['no', 'bitmask', 'mask2box']
        cfg.MODEL.MaskDINO.DN = "seg"
        cfg.MODEL.MaskDINO.DN_NOISE_SCALE = 0.4
        cfg.MODEL.MaskDINO.DN_NUM = 100
        cfg.MODEL.MaskDINO.PRED_CONV = False

        cfg.MODEL.MaskDINO.EVAL_FLAG = 1

        # MSDeformAttn encoder configs
        cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
        cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
        cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
        cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
        cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
        cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
        cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = (
            "high2low"
        )  # ['low2high', 'high2low'] high2low: from high level to low level

        #####################

        # MaskDINO inference config
        cfg.MODEL.MaskDINO.TEST = CN()
        cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = False
        cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = True
        cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False
        cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.0
        cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.0
        cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
        cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL = True
        cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE = 0.06
        # cfg.MODEL.MaskDINO.TEST.EVAL_FLAG = 1

        # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
        # you can use this config to override
        cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32

        # pixel decoder config
        cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
        # adding transformer in pixel decoder
        cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
        # pixel decoder
        cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"

        # transformer module
        cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"

        # LSJ aug
        cfg.INPUT.IMAGE_SIZE = 1024
        cfg.INPUT.MIN_SCALE = 0.1
        cfg.INPUT.MAX_SCALE = 2.0

        # point loss configs
        # Number of points sampled during training for a mask point head.
        cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 112 * 112
        # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
        # original paper.
        cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = 3.0
        # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
        # the original paper.
        cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = 0.75

        # swin transformer backbone
        cfg.MODEL.SWIN = CN()
        cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
        cfg.MODEL.SWIN.PATCH_SIZE = 4
        cfg.MODEL.SWIN.EMBED_DIM = 96
        cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
        cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
        cfg.MODEL.SWIN.WINDOW_SIZE = 7
        cfg.MODEL.SWIN.MLP_RATIO = 4.0
        cfg.MODEL.SWIN.QKV_BIAS = True
        cfg.MODEL.SWIN.QK_SCALE = None
        cfg.MODEL.SWIN.DROP_RATE = 0.0
        cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
        cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
        cfg.MODEL.SWIN.APE = False
        cfg.MODEL.SWIN.PATCH_NORM = True
        cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.SWIN.USE_CHECKPOINT = False

        cfg.Default_loading = True  # a bug in my d2. resume use this; if first time ResNet load, set it false

    def _create_model(self) -> nn.Module:
        detector = self._build_model(num_classes=self.label_info.num_classes)
        self.classification_layers = self.get_classification_layers("model.")

        if self.load_from is not None:
            DetectionCheckpointer(detector).resume_or_load(
                self.load_from,
                resume=False,
            )
        return detector

    def _build_model(self, num_classes: int) -> Module:
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        MaskDINOR50.add_maskdino_config(cfg)
        cfg.merge_from_file("src/otx/recipe/instance_segmentation/config.yaml")
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
        return MaskDINO(cfg)

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity):
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)
        return {"entity": entity}

    def _customize_outputs(
        self,
        outputs,
        inputs: InstanceSegBatchDataEntity,
    ):
        if self.training:
            return outputs

        target_sizes = [(max(m.shape), max(m.shape)) for m in inputs.masks]
        masks, bboxes, labels, scores = self.post_process_instance_segmentation(
            outputs,
            inputs.imgs_info,
            target_sizes=target_sizes,
        )

        if self.explain_mode:
            msg = "Explain mode is not supported yet."
            raise NotImplementedError(msg)

        return InstanceSegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )

    def post_process_instance_segmentation(
        self,
        outputs,
        imgs_info,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        class_queries_logits = outputs["pred_logits"]
        masks_queries_logits = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
        )

        device = masks_queries_logits.device
        num_classes = self.model.sem_seg_head.num_classes
        num_queries = self.model.test_topk_per_image

        batch_scores: list[Tensor] = []
        batch_bboxes: list[tv_tensors.BoundingBoxes] = []
        batch_labels: list[torch.LongTensor] = []
        batch_masks: list[tv_tensors.Mask] = []

        for mask_pred, mask_cls, pred_boxes, img_info, target_size in zip(
            masks_queries_logits,
            class_queries_logits,
            mask_box_results,
            imgs_info,
            target_sizes,
        ):
            ori_h, ori_w = img_info.ori_shape
            scores = mask_cls.sigmoid()
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
            mask_pred = mask_pred[topk_indices]
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.unsqueeze(0),
                size=target_size,
                mode="nearest",
            )[0][:, :ori_h, :ori_w]

            pred_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
            pred_boxes *= pred_boxes.new_tensor(img_info.img_shape).repeat((1, 2))
            pred_boxes /= pred_boxes.new_tensor(img_info.scale_factor[::-1]).repeat((1, 2))
            area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

            keep = (pred_masks.sum((1, 2)) > 10) & (pred_scores > 0.05) & (area > 10)
            batch_masks.append(pred_masks[keep])
            batch_bboxes.append(pred_boxes[keep])
            batch_labels.append(pred_classes[keep])
            batch_scores.append(pred_scores[keep])

        return batch_masks, batch_bboxes, batch_labels, batch_scores
