# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything model for the OTX zero-shot visual prompting."""

from __future__ import annotations

import logging as log
import os
from collections import defaultdict
from copy import deepcopy
from itertools import product
from typing import Any, Literal
import numpy as np

import torch
from datumaro import Polygon as dmPolygon
from torch import LongTensor, Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision import tv_tensors
import torchvision.transforms.v2 as tvt_v2
from torchvision.tv_tensors import BoundingBoxes, Image

from otx.algo.visual_prompting.segment_anything import (
    DEFAULT_CONFIG_SEGMENT_ANYTHING,
    SegmentAnything,
)
from otx.core.data.entity.base import OTXBatchLossEntity, Points
from otx.core.data.entity.visual_prompting import (
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.model.entity.visual_prompting import OTXZeroShotVisualPromptingModel


class PromptGetter(nn.Module):
    """Prompt getter for zero-shot learning."""

    default_threshold_reference = 0.3
    default_threshold_target = 0.65

    def __init__(self, image_size: int, downsizing: int = 64) -> None:
        super().__init__()
        self.image_size = image_size
        self.downsizing = downsizing

        self.zero_tensor = torch.tensor(0)

    def set_default_thresholds(self, default_threshold_reference: float, default_threshold_target: float) -> None:
        """Set default thresholds."""
        self.default_threshold_reference = default_threshold_reference
        self.default_threshold_target = default_threshold_target

    def get_prompt_candidates(
        self,
        image_embeddings: Tensor,
        reference_feats: Tensor,
        used_indices: Tensor,
        ori_shape: Tensor,
        threshold: Tensor | None = None,
        num_bg_points: Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        """Get prompt candidates."""
        if threshold is None:
            threshold = torch.tensor([[0.0]], dtype=torch.float32)
        if num_bg_points is None:
            num_bg_points = torch.tensor([[1]], dtype=torch.int64)

        threshold = threshold.to(device)

        total_points_scores: dict[int, Tensor] = {}
        total_bg_coords: dict[int, Tensor] = {}
        for label in map(int, used_indices):
            points_scores, bg_coords = self(
                image_embeddings=image_embeddings,
                reference_feat=reference_feats[label],
                ori_shape=ori_shape,
                threshold=threshold,
                num_bg_points=num_bg_points,
            )

            total_points_scores[label] = points_scores
            total_bg_coords[label] = bg_coords

        return total_points_scores, total_bg_coords

    def forward(
        self,
        image_embeddings: Tensor,
        reference_feat: Tensor,
        ori_shape: Tensor,
        threshold: Tensor | None = None,
        num_bg_points: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Get prompt candidates from given reference and target features."""
        if threshold is None:
            threshold = torch.tensor([[0.0]], dtype=torch.float32)
        if num_bg_points is None:
            num_bg_points = torch.tensor([[1]], dtype=torch.int64)

        ori_shape = ori_shape.squeeze()
        threshold = threshold.squeeze()
        num_bg_points = num_bg_points.squeeze()

        target_feat = image_embeddings.squeeze()
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        sim = reference_feat @ target_feat
        sim = sim.reshape(1, 1, h_feat, w_feat)
        sim = ZeroShotSegmentAnything.postprocess_masks(sim, self.image_size, ori_shape)

        threshold = (threshold == 0) * self.default_threshold_target + threshold
        points_scores, bg_coords = self._point_selection(
            mask_sim=sim[0, 0],
            ori_shape=ori_shape,
            threshold=threshold,
            num_bg_points=num_bg_points,
        )

        return points_scores, bg_coords

    def _point_selection(
        self,
        mask_sim: Tensor,
        ori_shape: Tensor,
        threshold: Tensor | None = None,
        num_bg_points: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Select point used as point prompts."""
        if threshold is None:
            threshold = torch.tensor([0.0], dtype=torch.float32)
        if num_bg_points is None:
            num_bg_points = torch.tensor(1, dtype=torch.int64)

        _, w_sim = mask_sim.shape

        # Top-last point selection
        bg_indices = mask_sim.flatten().topk(num_bg_points.item(), largest=False)[1]
        bg_x = (bg_indices // w_sim).unsqueeze(0)
        bg_y = bg_indices - bg_x * w_sim
        bg_coords = torch.cat((bg_y, bg_x), dim=0).permute(1, 0)
        bg_coords = bg_coords.to(torch.float32)

        point_coords = torch.where(mask_sim > threshold)
        fg_coords_scores = torch.stack(point_coords[::-1] + (mask_sim[point_coords],), dim=0).T

        # to handle empty tensor
        len_fg_coords_scores = len(fg_coords_scores)
        fg_coords_scores = F.pad(fg_coords_scores, (0, 0, 0, max(0, 1 - len_fg_coords_scores)), value=-1)

        ratio = self.image_size / ori_shape.max()
        width = (ori_shape[1] * ratio).to(torch.int64)
        n_w = width // self.downsizing

        # get grid numbers
        idx_grid = (
            fg_coords_scores[:, 1] * ratio // self.downsizing * n_w + fg_coords_scores[:, 0] * ratio // self.downsizing
        )
        idx_grid_unique = torch.unique(
            idx_grid.to(torch.int64),
        )  # unique op only supports INT64, INT8, FLOAT, STRING in ORT

        # get matched indices
        matched_matrix = idx_grid.unsqueeze(-1) == idx_grid_unique  # (totalN, uniqueN)

        # sample fg_coords_scores matched by matched_matrix
        matched_grid = fg_coords_scores.unsqueeze(1) * matched_matrix.unsqueeze(-1)

        matched_indices = matched_grid[..., -1].topk(k=1, dim=0, largest=True)[1][0].to(torch.int64)
        points_scores = matched_grid[matched_indices].diagonal().T

        # sort by the highest score
        sorted_points_scores_indices = torch.argsort(points_scores[:, -1], descending=True).to(torch.int64)
        points_scores = points_scores[sorted_points_scores_indices]

        return points_scores, bg_coords


class ZeroShotSegmentAnything(SegmentAnything):
    """Zero-shot learning module using Segment Anything."""

    def __init__(
        self,
        default_threshold_reference: float = 0.3,
        default_threshold_target: float = 0.65,
        *args,
        **kwargs,
    ) -> None:
        msg = ""
        if len(kwargs) == 0:
            msg += "There isn't any given argument. Default setting will be used."
        elif len(kwargs) == 1 and "backbone" in kwargs:
            msg += (
                f"There is only backbone (={kwargs.get('backbone')}) argument. "
                f"Other parameters will be set along with backbone (={kwargs.get('backbone')})."
            )
        elif "backbone" not in kwargs:
            msg += "There isn't a backbone argument, it will be reset with backbone=tiny_vit."
        if len(msg) > 0:
            log.info(msg)
            kwargs = self.set_default_config(**kwargs)

        # check freeze conditions
        for condition in ["freeze_image_encoder", "freeze_prompt_encoder", "freeze_mask_decoder"]:
            if not kwargs.get(condition, False):
                log.warning(f"{condition}(=False) must be set to True, changed.")
                kwargs[condition] = True

        self.is_cascade = kwargs.pop("is_cascade", True)
        self.save_outputs = kwargs.pop("save_outputs", True)
        self.root_reference_info = kwargs.pop("root_reference_info", "vpm_zsl_reference_infos")
        super().__init__(*args, **kwargs)

        self.initialize_reference_info()

        self.prompt_getter = PromptGetter(image_size=self.image_size)
        self.prompt_getter.set_default_thresholds(
            default_threshold_reference=default_threshold_reference,
            default_threshold_target=default_threshold_target,
        )

        self.point_labels_box = torch.tensor([[2, 3]], dtype=torch.float32)
        self.has_mask_inputs = [torch.tensor([[0.0]]), torch.tensor([[1.0]])]

    def set_default_config(self, **kwargs) -> dict[str, Any]:
        """Set default config when using independently."""
        backbone = kwargs.get("backbone", "tiny_vit")
        kwargs.update(
            {
                "backbone": backbone,
                "load_from": kwargs.get("load_from", DEFAULT_CONFIG_SEGMENT_ANYTHING[backbone]["load_from"]),
                "freeze_image_encoder": kwargs.get("freeze_image_encoder", True),
                "freeze_mask_decoder": kwargs.get("freeze_mask_decoder", True),
                "freeze_prompt_encoder": kwargs.get("freeze_prompt_encoder", True),
            },
        )
        return kwargs

    def initialize_reference_info(self) -> None:
        """Initialize reference information."""
        self.register_buffer("reference_feats", torch.zeros(0, 1, self.embed_dim), False)
        self.register_buffer("used_indices", torch.tensor([], dtype=torch.int64), False)

    def expand_reference_info(self, new_largest_label: int) -> None:
        """Expand reference info dimensions if newly given processed prompts have more lables."""            
        if new_largest_label > (cur_largest_label := len(self.reference_feats) - 1):
            diff = new_largest_label - cur_largest_label
            self.reference_feats = F.pad(self.reference_feats, (0, 0, 0, 0, 0, diff), value=0.0)

    @torch.no_grad()
    def learn(
        self,
        images: list[tv_tensors.Image],
        processed_prompts: list[dict[int, list[tv_tensors.TVTensor]]],
        ori_shapes: list[Tensor],
        reset_feat: bool = False,
    ) -> tuple[dict[str, Tensor], list[Tensor]] | None:
        """Get reference features.

        Using given images, get reference features.
        These reference features will be used for `infer` to get target results.
        Currently, single batch is only supported.

        Args:
            images (list[tv_tensors.Image]): List of given images for reference features.
            processed_prompts (dict[int, list[tv_tensors.TVTensor]]): The class-wise prompts
                processed at OTXZeroShotSegmentAnything._gather_prompts_with_labels.
            ori_shapes (List[Tensor]): List of original shapes per image.
            reset_feat (bool): Whether reset reference_info.
                For OTX standalone, resetting reference_info will be conducted in on_train_start.
                For other frameworks, setting it to True is required to reset reference_info. Defaults to False.
        """
        if reset_feat:
            self.initialize_reference_info()

        # initialize tensors to contain reference features and prompts
        largest_label = max(sum([[int(p) for p in prompt] for prompt in processed_prompts], []))
        self.expand_reference_info(largest_label)
        # TODO(sungchul): consider who to handle multiple reference features, currently replace it # noqa: TD003

        reference_masks: list[Tensor] = []
        for image, prompts, ori_shape in zip(images, processed_prompts, ori_shapes):
            image_embeddings = self.image_encoder(image)
            processed_embedding = image_embeddings.squeeze().permute(1, 2, 0)

            ref_masks = torch.zeros(largest_label + 1, *map(int, ori_shape))
            for label, input_prompts in prompts.items():
                # TODO (sungchul): how to skip background class # noqa: TD003
                # TODO (sungchul): ensemble multi reference features (current : use merged masks) # noqa: TD003
                ref_mask = torch.zeros(*map(int, ori_shape), dtype=torch.uint8, device=image.device)
                for input_prompt in input_prompts:
                    if isinstance(input_prompt, tv_tensors.Mask):
                        # directly use annotation information as a mask
                        ref_mask[
                            input_prompt == 1
                        ] += 1  # TODO(sungchul): check if the mask is bool or int # noqa: TD003
                    else:
                        if isinstance(input_prompt, BoundingBoxes):
                            point_coords = input_prompt.reshape(1, 2, 2)
                            point_labels = torch.tensor([[2, 3]], device=point_coords.device)
                        elif isinstance(input_prompt, Points):
                            point_coords = input_prompt.reshape(1, 1, 2)
                            point_labels = torch.tensor([[1]], device=point_coords.device)
                        elif isinstance(
                            input_prompt,
                            dmPolygon,
                        ):  # TODO(sungchul): add other polygon types # noqa: TD003
                            # TODO(sungchul): convert polygon to mask # noqa: TD003
                            continue
                        else:
                            log.info(f"Current input prompt ({input_prompt.__class__.__name__}) is not supported.")
                            continue

                        masks = self._predict_masks(
                            mode="learn",
                            image_embeddings=image_embeddings,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            ori_shape=ori_shape,
                            is_cascade=False,
                        )
                        ref_mask[masks] += 1
                ref_mask = torch.clip(ref_mask, 0, 1).to(torch.float32)

                ref_feat: Tensor | None = None
                default_threshold_reference = deepcopy(self.prompt_getter.default_threshold_reference)
                while ref_feat is None:
                    log.info(f"[*] default_threshold_reference : {default_threshold_reference:.4f}")
                    ref_feat = self._generate_masked_features(
                        processed_embedding,
                        ref_mask,
                        default_threshold_reference,
                    )
                    default_threshold_reference -= 0.05

                self.reference_feats[label] = ref_feat.detach().cpu()
                self.used_indices = torch.cat((self.used_indices, torch.tensor([label])), dim=0)
                ref_masks[label] = ref_mask.detach().cpu()
            reference_masks.append(ref_masks)
        return {"reference_feats": self.reference_feats, "used_indices": self.used_indices}, reference_masks

    @torch.no_grad()
    def infer(
        self,
        images: list[tv_tensors.Image],
        reference_feats: Tensor,
        used_indices: Tensor,
        ori_shapes: list[Tensor],
        is_cascade: bool = False,
    ) -> list[list[defaultdict[int, list[Tensor]]]]:
        """Zero-shot inference with reference features.

        Get target results by using reference features and target images' features.

        Args:
            images (list[tv_tensors.Image]): Given images for target results.
            reference_feats (Tensor): Reference features for target prediction.
            used_indices (Tensor): To check which indices of reference features are validate.
            ori_shapes (list[Tensor]): Original image size.
            is_cascade (bool): Whether use cascade inference. Defaults to False.

        Returns:
            (list[list[defaultdict[int, list[Tensor]]]]): List of predicted masks and used points.
        """
        total_results = []
        for image, ori_shape in zip(images, ori_shapes):
            if image.ndim == 3:
                image = image.unsqueeze(0)  # noqa: PLW2901

            # get image embeddings
            image_embeddings = self.image_encoder(image)

            total_points_scores, total_bg_coords = self.prompt_getter.get_prompt_candidates(
                image_embeddings=image_embeddings,
                reference_feats=reference_feats,
                used_indices=used_indices,
                ori_shape=ori_shape,
                device=image_embeddings.device,
            )
            predicted_masks: defaultdict = defaultdict(list)
            used_points: defaultdict = defaultdict(list)
            for label in total_points_scores:
                points_scores, bg_coords = total_points_scores[label], total_bg_coords[label]
                for point_score in points_scores:
                    x, y = point_score[:2]
                    is_done = False
                    for pm in predicted_masks.get(label, []):
                        # check if that point is already assigned
                        if pm[int(y), int(x)] > 0:
                            is_done = True
                            break
                    if is_done:
                        continue

                    point_coords = torch.cat((point_score[:2].unsqueeze(0), bg_coords), dim=0).unsqueeze(0)
                    point_coords = self._preprocess_coords(point_coords, ori_shape, self.image_size)
                    point_labels = torch.tensor(
                        [1] + [0] * len(bg_coords),
                        dtype=torch.float32,
                        device=point_coords.device,
                    ).unsqueeze(0)
                    mask = self._predict_masks(
                        mode="infer",
                        image_embeddings=image_embeddings,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        ori_shape=ori_shape,
                        is_cascade=is_cascade,
                    )
                    predicted_masks[label].append(mask * point_score[2])
                    used_points[label].append(point_score)

            # check overlapping area between different label masks
            self._inspect_overlapping_areas(predicted_masks, used_points)
            total_results.append([predicted_masks, used_points])
        return total_results

    def _inspect_overlapping_areas(
        self,
        predicted_masks: dict[int, list[Tensor]],
        used_points: dict[int, list[Tensor]],
        threshold_iou: float = 0.8,
    ) -> None:
        def _calculate_mask_iou(mask1: Tensor, mask2: Tensor) -> Tensor:
            intersection = torch.logical_and(mask1, mask2).sum().item()
            union = torch.logical_or(mask1, mask2).sum().item()
            if union == 0:
                # Avoid division by zero
                return 0.0
            return intersection / union

        for (label, masks), (other_label, other_masks) in product(predicted_masks.items(), predicted_masks.items()):
            if other_label <= label:
                continue

            overlapped_label = []
            overlapped_other_label = []
            for (im, mask), (jm, other_mask) in product(enumerate(masks), enumerate(other_masks)):
                if _calculate_mask_iou(mask, other_mask) > threshold_iou:
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        overlapped_other_label.append(jm)
                    else:
                        overlapped_label.append(im)

            for im in sorted(list(set(overlapped_label)), reverse=True):  # noqa: C414
                masks.pop(im)
                used_points[label].pop(im)

            for jm in sorted(list(set(overlapped_other_label)), reverse=True):  # noqa: C414
                other_masks.pop(jm)
                used_points[other_label].pop(jm)

    def _predict_masks(
        self,
        mode: str,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        ori_shape: Tensor,
        is_cascade: bool = True,
    ) -> Tensor:
        """Predict target masks."""
        masks: Tensor
        logits: Tensor
        scores: Tensor
        num_iter = 3 if is_cascade else 1
        for i in range(num_iter):
            if i == 0:
                # First-step prediction
                mask_input = torch.zeros(
                    1,
                    1,
                    *(x * 4 for x in image_embeddings.shape[2:]),
                    device=image_embeddings.device,
                )
                has_mask_input = self.has_mask_inputs[0].to(mask_input.device)

            elif i == 1:
                # Cascaded Post-refinement-1
                mask_input, best_masks = self._decide_cascade_results(masks, logits, scores, is_single=True)  # noqa: F821
                if best_masks.sum() == 0:
                    return best_masks

                has_mask_input = self.has_mask_inputs[1].to(mask_input.device)

            elif i == 2:
                # Cascaded Post-refinement-2
                mask_input, best_masks = self._decide_cascade_results(masks, logits, scores)  # noqa: F821
                if best_masks.sum() == 0:
                    return best_masks

                has_mask_input = self.has_mask_inputs[1].to(mask_input.device)
                coords = torch.nonzero(best_masks)
                y, x = coords[:, 0], coords[:, 1]
                box_coords = self._preprocess_coords(
                    torch.tensor([[[x.min(), y.min()], [x.max(), y.max()]]], dtype=torch.float32, device=coords.device),
                    ori_shape,
                    self.image_size,
                )
                point_coords = torch.cat((point_coords, box_coords), dim=1)
                point_labels = torch.cat((point_labels, self.point_labels_box.to(point_labels.device)), dim=1)

            high_res_masks, scores, logits = self(
                mode=mode,
                image_embeddings=image_embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                has_mask_input=has_mask_input,
                ori_shape=ori_shape,
            )
            masks = high_res_masks > self.mask_threshold
        _, best_masks = self._decide_cascade_results(masks, logits, scores)
        return best_masks

    def _preprocess_coords(
        self,
        coords: Tensor,
        ori_shape: list[int] | tuple[int, int] | Tensor,
        target_length: int,
    ) -> Tensor:
        """Expects a torch tensor of length 2 in the final dimension.

        Requires the original image size in (H, W) format.

        Args:
            coords (Tensor): Coordinates tensor.
            ori_shape (List[int] | Tuple[int, int] | Tensor]): Original size of image.
            target_length (int): The length of the longest side of the image.

        Returns:
            (Tensor): Resized coordinates.
        """
        old_h, old_w = ori_shape
        new_h, new_w = self.get_prepadded_size(ori_shape, target_length)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def _generate_masked_features(
        self,
        feats: Tensor,
        masks: Tensor,
        threshold_mask: float,
    ) -> tuple[Tensor, ...] | None:
        """Generate masked features.

        Args:
            feats (Tensor): Raw reference features. It will be filtered with masks.
            masks (Tensor): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.

        Returns:
            (Tensor): Masked features.
        """
        scale_factor = self.image_size / max(masks.shape)

        # Post-process masks
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode="bilinear").squeeze()
        masks = self._pad_to_square(masks)
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=feats.shape[0:2], mode="bilinear").squeeze()

        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0).unsqueeze(0)
        return masked_feat / masked_feat.norm(dim=-1, keepdim=True)

    def _pad_to_square(self, x: Tensor) -> Tensor:
        """Pad to a square input.

        Args:
            x (Tensor): Mask to be padded.

        Returns:
            (Tensor): Padded mask.
        """
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        return F.pad(x, (0, padw, 0, padh))

    def _decide_cascade_results(
        self,
        masks: Tensor,
        logits: Tensor,
        scores: Tensor,
        is_single: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Post-process masks for cascaded post-refinements."""
        if is_single:
            best_idx = 0
        else:
            # skip the first index components
            scores, masks, logits = (x[:, 1:] for x in (scores, masks, logits))

            # filter zero masks
            while len(scores[0]) > 0 and masks[0, (best_idx := torch.argmax(scores[0]))].sum() == 0:
                scores, masks, logits = (
                    torch.cat((x[:, :best_idx], x[:, best_idx + 1 :]), dim=1) for x in (scores, masks, logits)
                )

            if len(scores[0]) == 0:
                # all predicted masks were zero masks, ignore them.
                return None, torch.zeros(masks.shape[-2:], device="cpu")

            best_idx = torch.argmax(scores[0])
        return logits[:, best_idx], masks[0, best_idx]
    
    def _find_latest_reference_info(self, root: str = "vpm_zsl_reference_infos") -> str | None:
        """Find latest reference info to be used."""
        if not os.path.isdir(root):
            return None
        if len(stamps := sorted(os.listdir(root), reverse=True)) > 0:
            return stamps[0]
        return None
    
    def _load_latest_reference_info(self, device: str | torch.device = "cpu") -> bool:
        """Load latest reference info to be used."""
        if (latest_stamp := self._find_latest_reference_info(self.root_reference_info)) is not None:
            latest_reference_info = os.path.join(self.root_reference_info, latest_stamp, "reference_info.pt")
            reference_info = torch.load(latest_reference_info)
            self.register_buffer("reference_feats", reference_info.get("reference_feats", torch.zeros(0, 1, self.embed_dim)).to(device), False)
            self.register_buffer("used_indices", reference_info.get("used_indices", torch.tensor([], dtype=torch.int64)).to(device), False)
            log.info(f"reference info saved at {latest_reference_info} was successfully loaded.")
            return True
        return False


class OTXZeroShotSegmentAnything(OTXZeroShotVisualPromptingModel):
    """Zero-Shot Visual Prompting model."""

    def __init__(
        self,
        backbone: Literal["tiny_vit", "vit_b"],
        num_classes: int = 0,
        pixel_mean: list[float] = [123.675, 116.28, 103.53],
        pixel_std: list[float] = [58.395, 57.12, 57.375],
        **kwargs):
        self.config = {"backbone": backbone, **DEFAULT_CONFIG_SEGMENT_ANYTHING[backbone], **kwargs}
        super().__init__(num_classes=num_classes)
        
        self.register_buffer("pixel_mean", Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", Tensor(pixel_std).view(-1, 1, 1), False)

    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""
        return ZeroShotSegmentAnything(**self.config)

    def forward(  # type: ignore[override]
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        forward_fn = self.learn if self.training else self.infer
        return forward_fn(inputs)
    
    def learn(self, inputs: ZeroShotVisualPromptingBatchDataEntity) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Learn to directly connect to the model."""
        outputs = self.model.learn(**self._customize_inputs(inputs))
        return self._customize_outputs(outputs, inputs)
    
    def infer(self, inputs: ZeroShotVisualPromptingBatchDataEntity) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Infer to directly connect to the model."""
        outputs = self.model.infer(**self._customize_inputs(inputs))
        return self._customize_outputs(outputs, inputs)

    def _customize_inputs(self, inputs: ZeroShotVisualPromptingBatchDataEntity) -> dict[str, Any]:  # type: ignore[override]
        """Customize the inputs for the model."""
        inputs = self.transforms(inputs)
        if self.training:
            # learn
            return {
                "images": [tv_tensors.wrap(image.unsqueeze(0), like=image) for image in inputs.images],
                "ori_shapes": [torch.tensor(info.ori_shape) for info in inputs.imgs_info],
                "processed_prompts": self._gather_prompts_with_labels(inputs.prompts, inputs.labels),
            }

        # infer
        return {
            "images": [tv_tensors.wrap(image.unsqueeze(0), like=image) for image in inputs.images],
            "reference_feats": self.model.reference_feats,
            "used_indices": self.model.used_indices,
            "ori_shapes": [torch.tensor(info.ori_shape) for info in inputs.imgs_info],
            "is_cascade": self.model.is_cascade,
        }

    def _customize_outputs(  # type: ignore[override]
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for you model."""
        if self.training:            
            return outputs

        masks: list[tv_tensors.Mask] = []
        prompts: list[Points] = []
        scores: list[Tensor] = []
        labels: list[LongTensor] = []
        for predicted_masks, used_points in outputs:
            for label, predicted_mask in predicted_masks.items():
                if len(predicted_mask) == 0:
                    continue
                masks.append(tv_tensors.Mask(torch.stack(predicted_mask, dim=0), dtype=torch.float32))
                prompts.append(
                    Points(
                        torch.stack([p[:2] for p in used_points[label]], dim=0),
                        canvas_size=inputs.imgs_info[0].ori_shape,
                        dtype=torch.float32,
                    ),
                )
                scores.append(torch.stack([p[2] for p in used_points[label]], dim=0))
                labels.append(torch.stack([LongTensor([label]) for _ in range(scores[-1].shape[0])], dim=0))

        return ZeroShotVisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            prompts=prompts,
            masks=masks,
            polygons=[],
            labels=labels,
        )

    def _gather_prompts_with_labels(
        self,
        prompts: list[list[tv_tensors.TVTensor]],
        labels: list[Tensor],
    ) -> list[dict[int, list[tv_tensors.TVTensor]]]:
        """Gather prompts according to labels."""
        total_processed_prompts: list[dict[int, list[tv_tensors.TVTensor]]] = []
        for prompt, label in zip(prompts, labels):
            processed_prompts = defaultdict(list)
            for _prompt, _label in zip(prompt, label):
                processed_prompts[int(_label)].append(_prompt)
            sorted_processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x))
            total_processed_prompts.append(sorted_processed_prompts)
        return total_processed_prompts

    def apply_image(self, image: tv_tensors.Image | np.ndarray, target_length: int = 1024) -> tv_tensors.Image:
        """Preprocess image to be used in the model."""
        h, w = image.shape[-2:]
        target_size = self.get_preprocess_shape(h, w, target_length)
        return tvt_v2.functional.resize(tvt_v2.functional.to_image(image), target_size, antialias=True)

    def apply_coords(self, coords: Tensor, ori_shape: tuple[int, ...], target_length: int = 1024) -> Tensor:
        """Preprocess points to be used in the model."""
        old_h, old_w = ori_shape
        new_h, new_w = self.get_preprocess_shape(ori_shape[0], ori_shape[1], target_length)
        coords = deepcopy(coords).to(torch.float32)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    def apply_points(self, points: Points, ori_shape: tuple[int, ...], target_length: int = 1024) -> Points:
        """Preprocess points to be used in the model."""
        return Points(self.apply_coords(points, ori_shape, target_length), canvas_size=(target_length, target_length))
    
    def apply_boxes(self, boxes: BoundingBoxes, ori_shape: tuple[int, ...], target_length: int = 1024) -> BoundingBoxes:
        """Preprocess boxes to be used in the model."""
        return BoundingBoxes(
            self.apply_coords(boxes.reshape(-1, 2, 2), ori_shape, target_length).reshape(-1, 4),
            format=boxes.format,
            canvas_size=(target_length, target_length))
    
    def apply_prompts(self, prompts: list[Points | BoundingBoxes], ori_shape: tuple[int, ...], target_length: int = 1024) -> list[Points | BoundingBoxes]:
        """Preprocess prompts to be used in the model."""
        transformed_prompts: list[Points, BoundingBoxes] = []
        for prompt in prompts:
            if isinstance(prompt, Points):
                transformed_prompts.append(self.apply_points(prompt, ori_shape, target_length))
            elif isinstance(prompt, BoundingBoxes):
                transformed_prompts.append(self.apply_boxes(prompt, ori_shape, target_length))
            else:
                log.info(f"Current prompt ({prompt.__class__.__name__}) is not supported, saved as it is.")
                transformed_prompts.append(prompt)
        return transformed_prompts

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """Get preprocess shape."""
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    def preprocess(self, x: Image) -> Image:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        x = self.model._pad_to_square(x)
        return Image(x)
    
    def transforms(self, entity: ZeroShotVisualPromptingBatchDataEntity) -> ZeroShotVisualPromptingBatchDataEntity:
        """Transforms for ZeroShotVisualPromptingBatchDataEntity."""
        entity.images = [self.preprocess(self.apply_image(image)) for image in entity.images]
        entity.prompts = [self.apply_prompts(prompt, info.ori_shape, self.model.image_size) for prompt, info in zip(entity.prompts, entity.imgs_info)]
        return entity
