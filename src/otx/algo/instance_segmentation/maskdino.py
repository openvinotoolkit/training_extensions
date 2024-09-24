from __future__ import annotations

import copy
import itertools

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from torch import Tensor, nn
from torch.nn.modules import Module
from torchvision import tv_tensors

from otx.algo.instance_segmentation.mask_dino import box_ops
from otx.algo.instance_segmentation.mask_dino.criterion import SetCriterion
from otx.algo.instance_segmentation.mask_dino.maskdino_head import MaskDINOHead
from otx.algo.instance_segmentation.mask_dino.matcher import HungarianMatcher
from otx.algo.instance_segmentation.mask_dino.misc import ShapeSpec
from otx.algo.instance_segmentation.mask_dino.pixel_decoder.maskdino_encoder import MaskDINOEncoder
from otx.algo.instance_segmentation.mask_dino.resnet import build_resnet_backbone
from otx.algo.instance_segmentation.mask_dino.transformer_decoder.maskdino_decoder import MaskDINODecoder
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel
from otx.core.utils.mask_util import polygon_to_bitmap


class MaskDINO(nn.Module):
    """Main class for mask classification semantic segmentation architectures."""

    def __init__(
        self,
        backbone: nn.Module,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: tuple[float],
        pixel_std: tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
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

        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        print("criterion.weight_dict ", self.criterion.weight_dict)

    @classmethod
    def from_config(cls, num_classes):
        backbone = build_resnet_backbone(
            norm="FrozenBN",
            stem_out_channels=64,
            input_shape=ShapeSpec(channels=3),
            freeze_at=0,
            out_features=["res2", "res3", "res4", "res5"],
            depth=50,
            num_groups=1,
            width_per_group=64,
            in_channels=64,
            out_channels=256,
            stride_in_1x1=False,
            res5_dilation=1,
        )

        sem_seg_head = MaskDINOHead(
            input_shape={k: v for k, v in backbone.output_shape().items()},
            ignore_value=255,
            num_classes=num_classes,
            pixel_decoder=MaskDINOEncoder(
                input_shape={k: v for k, v in backbone.output_shape().items()},
                conv_dim=256,
                mask_dim=256,
                norm="GN",
                transformer_dropout=0.0,
                transformer_nheads=8,
                transformer_dim_feedforward=2048,
                transformer_enc_layers=6,
                transformer_in_features=["res3", "res4", "res5"],
                common_stride=4,
                total_num_feature_levels=4,
                num_feature_levels=3,
                feature_order="low2high",
            ),
            loss_weight=1.0,
            transformer_predictor=MaskDINODecoder(
                in_channels=256,
                mask_classification=True,
                num_classes=num_classes,
                hidden_dim=256,
                num_queries=300,
                nheads=8,
                dim_feedforward=2048,
                dec_layers=9,
                enforce_input_project=False,
                mask_dim=256,
                two_stage=True,
                initialize_box_type="mask2box",
                dn="seg",
                noise_scale=0.4,
                dn_num=100,
                initial_pred=True,
                learn_tgt=False,
                total_num_feature_levels=4,
                semantic_ce_loss=False,
            ),
        )

        # Loss parameters:
        no_object_weight = 0.1

        # loss weights
        class_weight = 4.0
        cost_class_weight = 4.0
        cost_dice_weight = 5.0
        dice_weight = 5.0
        cost_mask_weight = 5.0
        mask_weight = 5.0
        cost_box_weight = 5.0
        box_weight = 5.0
        cost_giou_weight = 2.0
        giou_weight = 2.0
        train_num_points = 12544
        oversample_ratio = 3.0
        importance_sample_ratio = 0.75

        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=train_num_points,
        )

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox": box_weight, "loss_giou": giou_weight})

        # two stage is the query selection scheme
        interm_weight_dict = {}
        interm_weight_dict.update({k + "_interm": v for k, v in weight_dict.items()})
        weight_dict.update(interm_weight_dict)

        # denoising training
        weight_dict.update({k + "_dn": v for k, v in weight_dict.items()})
        dn_losses = ["labels", "masks", "boxes"]

        dec_layers = 9
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "boxes"]

        # building criterion
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=train_num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            dn="seg",
            dn_losses=dn_losses,
            panoptic_on=False,
            semantic_ce_loss=False,
        )

        return MaskDINO(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=300,
            object_mask_threshold=0.25,
            overlap_threshold=0.8,
            size_divisibility=32,
            sem_seg_postprocess_before_inference=True,
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
            semantic_on=False,
            instance_on=True,
            panoptic_on=False,
            test_topk_per_image=100,
            focus_on_box=False,
            transform_eval=True,
            pano_temp=0.06,
            semantic_ce_loss=False,
        )

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

    def export(self, batch_inputs: Tensor, batch_img_metas: list[dict]):
        b, c, h, w = batch_inputs.size()
        if b != 1:
            raise ValueError("Only support batch size 1 for export")

        features = self.backbone(batch_inputs)
        outputs, _ = self.sem_seg_head(features)
        mask_cls = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]

        num_classes = self.sem_seg_head.num_classes
        num_queries = self.num_queries
        test_topk_per_image = self.test_topk_per_image

        scores = mask_cls.sigmoid()
        labels = torch.arange(num_classes).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes

        mask_pred = torch.nn.functional.interpolate(
            mask_pred.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]

        mask_pred = mask_pred[topk_indices]
        pred_boxes = pred_boxes[topk_indices]
        pred_masks = (mask_pred > 0).float()

        # Calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
            pred_masks.flatten(1).sum(1) + 1e-6
        )
        pred_scores = scores_per_image * mask_scores_per_image
        pred_classes = labels_per_image

        pred_boxes = pred_boxes.new_tensor([[w, h, w, h]]) * box_ops.box_cxcywh_to_xyxy(pred_boxes)

        boxes_with_scores = torch.cat([pred_boxes, pred_scores[:, None]], dim=1)

        batch_masks, batch_bboxes, batch_labels = [], [], []

        batch_masks.append(pred_masks)
        batch_bboxes.append(boxes_with_scores)
        batch_labels.append(pred_classes)

        return (
            batch_bboxes,
            batch_labels,
            batch_masks,
        )


class MaskDINOR50(ExplainableOTXInstanceSegModel):
    load_from = "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth"
    image_size = (1, 3, 1024, 1024)
    tile_image_size = (1, 3, 512, 512)
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def _build_model(self, num_classes: int) -> Module:
        return MaskDINO.from_config(num_classes)

    def _create_model(self) -> nn.Module:
        detector = self._build_model(num_classes=self.label_info.num_classes)
        self.classification_layers = self.get_classification_layers("model.")

        if self.load_from is not None:
            DetectionCheckpointer(detector).resume_or_load(
                self.load_from,
                resume=False,
            )
        return detector

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            msg = f"Image size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        input_size = self.tile_image_size if self.tile_config.enable_tiler else self.image_size

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=input_size,
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "masks": {0: "batch", 1: "num_dets", 2: "height", 3: "width"},
                },
                "opset_version": 16,
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "masks"],
        )

    def configure_optimizers(self):
        optimizer = self.build_optimizer(self.model)

        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item) -> list:
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    def build_optimizer(self, model):
        base_lr = 0.0001
        weight_decay_norm = 0.0
        weight_decay_embed = 0.0
        backbone_multiplier = 0.1
        clip_gradients_value = 0.01

        defaults = {}
        defaults["lr"] = base_lr
        defaults["weight_decay"] = 0.05

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params = []
        uniques = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in uniques:
                    continue
                uniques.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * backbone_multiplier
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def add_full_model_gradient_clipping(optim):
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_gradients_value)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer

        optimizer = add_full_model_gradient_clipping(torch.optim.AdamW)(params, base_lr)
        return optimizer

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
            return sum(outputs.values())

        masks, bboxes, labels, scores = self.post_process_instance_segmentation(
            outputs,
            inputs.imgs_info,
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        class_queries_logits = outputs["pred_logits"]
        masks_queries_logits = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]

        device = masks_queries_logits.device
        num_classes = self.model.sem_seg_head.num_classes
        num_queries = self.model.num_queries
        test_topk_per_image = self.model.test_topk_per_image

        batch_scores: list[Tensor] = []
        batch_bboxes: list[tv_tensors.BoundingBoxes] = []
        batch_labels: list[torch.LongTensor] = []
        batch_masks: list[tv_tensors.Mask] = []

        for mask_pred, mask_cls, pred_boxes, img_info in zip(
            masks_queries_logits,
            class_queries_logits,
            mask_box_results,
            imgs_info,
        ):
            ori_h, ori_w = img_info.ori_shape
            scores = mask_cls.sigmoid()
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes

            mask_pred = torch.nn.functional.interpolate(
                mask_pred.unsqueeze(0),
                size=(ori_h, ori_w),
                mode="bilinear",
                align_corners=False,
            )[0]

            mask_pred = mask_pred[topk_indices]
            pred_boxes = pred_boxes[topk_indices]
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image

            pred_boxes = pred_boxes.new_tensor([[ori_w, ori_h, ori_w, ori_h]]) * box_ops.box_cxcywh_to_xyxy(pred_boxes)
            pred_boxes[:, 0::2].clamp_(min=0, max=ori_w - 1)
            pred_boxes[:, 1::2].clamp_(min=0, max=ori_h - 1)

            area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
            keep = (pred_masks.sum((1, 2)) > 5) & (area > 10) & (pred_scores > 0.05)

            batch_masks.append(pred_masks[keep])
            batch_bboxes.append(pred_boxes[keep])
            batch_labels.append(pred_classes[keep])
            batch_scores.append(pred_scores[keep])

        return batch_masks, batch_bboxes, batch_labels, batch_scores
