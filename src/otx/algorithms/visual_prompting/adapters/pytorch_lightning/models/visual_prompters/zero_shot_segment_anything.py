"""SAM module for visual prompting zero-shot learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import product
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict
from torch.nn import functional as F

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.dataset import get_transform
from otx.api.entities.scored_label import ScoredLabel
from otx.utils.logger import get_logger

from .segment_anything import SegmentAnything

logger = get_logger()


class PromptGetter(nn.Module):
    """Prompt getter for zero-shot learning."""

    default_threshold_reference = 0.3
    default_threshold_target = 0.65

    def __init__(self, image_size: int, downsizing: int = 64) -> None:
        super().__init__()
        self.image_size = image_size
        self.downsizing = downsizing

    def set_default_thresholds(self, default_threshold_reference: float, default_threshold_target: float) -> None:
        """Set default thresholds."""
        self.default_threshold_reference = default_threshold_reference
        self.default_threshold_target = default_threshold_target

    def get_prompt_candidates(
        self,
        image_embeddings: Tensor,
        reference_feats: Tensor,
        used_indices: Tensor,
        original_size: Tensor,
        threshold: Tensor = torch.tensor([[0.0]], dtype=torch.float32),
        num_bg_points: Tensor = torch.tensor([[1]], dtype=torch.int64),
        device: Union[torch.device, str] = torch.device("cpu"),
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        """Get prompt candidates."""
        threshold = threshold.to(device)

        total_points_scores: Dict[int, Tensor] = {}
        total_bg_coords: Dict[int, Tensor] = {}
        for label in map(int, used_indices):
            points_scores, bg_coords = self(
                image_embeddings=image_embeddings,
                reference_feat=reference_feats[label],
                original_size=original_size,
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
        original_size: Tensor,
        threshold: Tensor = torch.tensor([[0.0]], dtype=torch.float32),
        num_bg_points: Tensor = torch.tensor([[1]], dtype=torch.int64),
    ) -> Tuple[Tensor, Tensor]:
        """Get prompt candidates from given reference and target features."""
        original_size = original_size.squeeze()
        threshold = threshold.squeeze()
        num_bg_points = num_bg_points.squeeze()

        target_feat = image_embeddings.squeeze()
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        sim = reference_feat @ target_feat
        sim = sim.reshape(1, 1, h_feat, w_feat)
        sim = ZeroShotSegmentAnything.postprocess_masks(sim, self.image_size, original_size)

        threshold = (threshold == 0) * self.default_threshold_target + threshold
        points_scores, bg_coords = self._point_selection(
            mask_sim=sim[0, 0],
            original_size=original_size,
            threshold=threshold,
            num_bg_points=num_bg_points,
        )

        return points_scores, bg_coords

    def _point_selection(
        self,
        mask_sim: Tensor,
        original_size: Tensor,
        threshold: Union[Tensor, float] = 0.0,
        num_bg_points: Union[Tensor, int] = 1,
    ) -> Tuple[Tensor, Tensor]:
        """Select point used as point prompts."""
        _, w_sim = mask_sim.shape

        # Top-last point selection
        bg_indices = mask_sim.flatten().topk(num_bg_points, largest=False)[1]
        bg_x = (bg_indices // w_sim).unsqueeze(0)
        bg_y = bg_indices - bg_x * w_sim
        bg_coords = torch.cat((bg_y, bg_x), dim=0).permute(1, 0)
        bg_coords = bg_coords.to(torch.float32)

        point_coords = torch.where(mask_sim > threshold)
        fg_coords_scores = torch.stack(point_coords[::-1] + (mask_sim[point_coords],), dim=0).T

        # to handle empty tensor
        len_fg_coords_scores = len(fg_coords_scores)
        fg_coords_scores = F.pad(fg_coords_scores, (0, 0, 0, max(0, 1 - len_fg_coords_scores)), value=-1)

        ratio = self.image_size / original_size.max()
        width = (original_size[1] * ratio).to(torch.int64)
        n_w = width // self.downsizing

        # get grid numbers
        idx_grid = (
            fg_coords_scores[:, 1] * ratio // self.downsizing * n_w + fg_coords_scores[:, 0] * ratio // self.downsizing
        )
        idx_grid_unique = torch.unique(
            idx_grid.to(torch.int64)
        )  # unique op only supports INT64, INT8, FLOAT, STRING in ORT

        # get matched indices
        matched_matrix = idx_grid.unsqueeze(-1) == idx_grid_unique  # (totalN, uniqueN)

        # sample fg_coords_scores matched by matched_matrix
        matched_grid = fg_coords_scores.unsqueeze(1) * matched_matrix.unsqueeze(-1)

        # sample the highest score one of the samples that are in the same grid
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
        config: Optional[DictConfig] = None,
        manual_config_update: Optional[Dict] = None,
        state_dict: Optional[OrderedDict] = None,
    ) -> None:
        if config is None:
            config = self.set_default_config()

        if (
            manual_config_update is not None
            and isinstance(manual_config_update, dict)
            and len(manual_config_update) > 0
        ):
            for k, v in manual_config_update.items():
                exec(f"config.{k} = {v}")

        # check freeze conditions
        for condition in ["freeze_image_encoder", "freeze_prompt_encoder", "freeze_mask_decoder"]:
            if not getattr(config.model, condition, False):
                logger.warning(f"config.model.{condition}(=False) must be set to True, changed.")
                setattr(config.model, condition, True)

        super().__init__(config, state_dict)

        self.set_empty_reference_info()

        self.prompt_getter = PromptGetter(image_size=config.model.image_size)
        self.prompt_getter.set_default_thresholds(
            default_threshold_reference=config.model.default_threshold_reference,
            default_threshold_target=config.model.default_threshold_target,
        )

        self.point_labels_box = torch.tensor([[2, 3]], dtype=torch.float32)
        self.has_mask_inputs = [torch.tensor([[0.0]]), torch.tensor([[1.0]])]

        self.transforms = get_transform(
            image_size=config.model.image_size, mean=config.dataset.normalize.mean, std=config.dataset.normalize.std
        )

        self.path_reference_info = "vpm_zsl_reference_infos/{}/reference_info.pt"

    def load_state_dict_pre_hook(self, state_dict: Dict[str, Any], prefix: str = "", *args, **kwargs) -> None:
        """Load reference info manually."""
        _reference_feats: Tensor = state_dict.get(
            "reference_info.reference_feats", torch.tensor([], dtype=torch.float32)
        )
        _used_indices: Tensor = state_dict.get("reference_info.used_indices", torch.tensor([], dtype=torch.int64))
        self.reference_info = ParameterDict(
            {
                "reference_feats": Parameter(_reference_feats, requires_grad=False),
                "used_indices": Parameter(_used_indices, requires_grad=False),
            },
        )

    def set_default_config(self) -> DictConfig:
        """Set default config when using independently."""
        return DictConfig(
            {
                "model": {
                    "backbone": "tiny_vit",
                    "checkpoint": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
                    "default_threshold_reference": 0.3,
                    "default_threshold_target": 0.65,
                    "freeze_image_encoder": True,
                    "freeze_mask_decoder": True,
                    "freeze_prompt_encoder": True,
                    "image_size": 1024,
                    "mask_threshold": 0.0,
                    "return_single_mask": False,
                    "use_stability_score": False,
                    "stability_score_offset": 1.0,
                    "return_extra_metrics": False,
                },
                "dataset": {
                    "normalize": {
                        "mean": [123.675, 116.28, 103.53],
                        "std": [58.395, 57.12, 57.375],
                    }
                },
            }
        )

    def set_empty_reference_info(self) -> None:
        """Set empty reference information."""
        reference_feats: Parameter = Parameter(torch.tensor([], dtype=torch.float32), requires_grad=False)
        used_indices: Parameter = Parameter(torch.tensor([], dtype=torch.int64), requires_grad=False)
        self.reference_info = ParameterDict(
            {
                "reference_feats": reference_feats,
                "used_indices": used_indices,
            },
        )
        self.is_reference_info_empty = True

    def initialize_reference_info(self) -> None:
        """Initialize reference information."""
        self.reference_info["reference_feats"] = Parameter(torch.zeros(0, 1, 256), requires_grad=False)
        self.reference_info["used_indices"] = Parameter(torch.tensor([], dtype=torch.int64), requires_grad=False)
        self.is_reference_info_empty = False

    def expand_reference_info(self, new_largest_label: int) -> None:
        """Expand reference info dimensions if newly given processed prompts have more lables."""
        if new_largest_label > (cur_largest_label := len(self.reference_info["reference_feats"]) - 1):
            diff = new_largest_label - cur_largest_label
            padded_reference_feats = F.pad(self.reference_info["reference_feats"], (0, 0, 0, 0, 0, diff), value=0.0)
            self.reference_info["reference_feats"] = Parameter(padded_reference_feats, requires_grad=False)

    @torch.no_grad()
    def learn(
        self, batch: List[Dict[str, Any]], reset_feat: bool = False, is_cascade: bool = False
    ) -> Union[None, Tuple[ParameterDict, Tensor]]:
        """Get reference features.

        Using given images, get reference features and save it to PromptGetter.
        These reference features will be used for `infer` to get target results.
        Currently, single batch is only supported.

        Args:
            batch (List[Dict[str, Any]]): List of dictionaries containing images, prompts, and metas.
                `batch` must contain images, prompts with bboxes, points, annotations, and polygons.
            reset_feat (bool): Whether reset reference_info.
                For OTX standalone, resetting reference_info will be conducted in on_train_start.
                For other frameworks, setting it to True is required to reset reference_info. Defaults to False.
            is_cascade (bool): Whether use cascade inference. Defaults to False.

        Returns:
            (Tuple[ParameterDict, Tensor]): reference_info and ref_masks.
        """
        if reset_feat:
            self.initialize_reference_info()

        # preprocess images and prompts
        transformed_batch = [self.transforms(b.copy()) for b in batch]
        processed_prompts = [self._preprocess_prompts(tb) for tb in transformed_batch]

        # initialize tensors to contain reference features and prompts
        largest_label = max([label for pp in processed_prompts for label in pp.keys()])
        self.expand_reference_info(largest_label)
        # TODO(sungchul): consider who to handle multiple reference features, currently replace it

        batch_ref_masks: List[Tensor] = []
        for tb, pp in zip(transformed_batch, processed_prompts):
            # assign components
            images = tb["images"].unsqueeze(0).to(self.device)  # type: ignore[union-attr]
            original_size = torch.as_tensor(tb["original_size"])

            image_embeddings = self.image_encoder(images)
            processed_embedding = image_embeddings.squeeze().permute(1, 2, 0)

            ref_masks = torch.zeros(largest_label + 1, *map(int, original_size))
            for label, input_prompts in pp.items():
                # TODO (sungchul): how to skip background class

                # generate reference mask
                # TODO (sungchul): ensemble multi reference features (current : use merged masks)
                ref_mask = torch.zeros(*map(int, original_size), dtype=torch.uint8, device=self.device)
                for input_prompt in input_prompts:
                    if (prompt := input_prompt.get("annotations", None)) is not None:
                        # directly use annotation information as a mask
                        ref_mask[prompt == 1] += 1
                    elif (prompt := input_prompt.get("polygons", None)) is not None:
                        for polygon in prompt["polygons"]:
                            contour = [[int(point[0]), int(point[1])] for point in polygon]
                            mask_from_polygon = np.zeros(original_size, dtype=np.uint8)
                            mask_from_polygon = cv2.drawContours(mask_from_polygon, np.asarray([contour]), 0, 1, -1)
                            ref_mask[mask_from_polygon == 1] += 1
                    elif (prompt := input_prompt.get("scribble_annotation", None)) is not None:
                        logger.warning("scribble_annotation is not supported yet.")
                        continue
                    elif (prompt := input_prompt.get("scribble_polygon", None)) is not None:
                        logger.warning("scribble_polygon is not supported yet.")
                        continue
                    else:
                        point_coords = []
                        point_labels = []
                        if (prompt := input_prompt.get("bboxes", None)) is not None:
                            point_coords = prompt["point_coords"].reshape(1, 2, 2)

                        elif (prompt := input_prompt.get("points", None)) is not None:
                            point_coords = prompt["point_coords"].reshape(1, 1, 2)

                        point_labels = prompt["point_labels"]

                        masks = self._predict_masks(
                            image_embeddings=image_embeddings,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            original_size=original_size,
                            is_cascade=is_cascade,
                        )
                        ref_mask[masks] += 1
                ref_mask = torch.clip(ref_mask, 0, 1).to(torch.float32)

                ref_feat = None
                default_threshold_reference = deepcopy(self.prompt_getter.default_threshold_reference)
                while ref_feat is None:
                    logger.info(f"[*] default_threshold_reference : {default_threshold_reference:.4f}")
                    ref_feat = self._generate_masked_features(
                        processed_embedding, ref_mask, default_threshold_reference
                    )
                    default_threshold_reference -= 0.05

                self.reference_info["reference_feats"][label] = ref_feat.detach().cpu()
                self.reference_info["used_indices"] = Parameter(
                    torch.cat((self.reference_info["used_indices"], torch.tensor([[label]]))),
                    requires_grad=False,
                )
                ref_masks[label] = ref_mask.detach().cpu()
            batch_ref_masks.append(ref_masks)
        return self.reference_info, batch_ref_masks

    @torch.no_grad()
    def infer(
        self,
        batch: List[Dict[str, Any]],
        reference_feats: Union[np.ndarray, Tensor],
        used_indices: Union[np.ndarray, Tensor],
        is_cascade: bool = True,
    ) -> List[List[DefaultDict[int, List[Tensor]]]]:
        """Zero-shot inference with reference features.

        Get target results by using reference features and target images' features.

        Args:
            batch (List[Dict[str, Any]]): List of dictionaries containing images and metas.
            reference_feats (Union[np.ndarray, Tensor]): Reference features for target prediction.
                If it is np.ndarray, it will be converted to torch tensor.
            used_indices (Union[np.ndarray, Tensor]): To check which indices of reference features are validate.
                If it is np.ndarray, it will be converted to torch tensor.
            is_cascade (bool): Whether use cascade inference. Defaults to True.

        Returns:
            (List[List[DefaultDict[int, List[Tensor]]]]): Target results.
                Lists wrapping results is following this order:
                    1. Target images
                    2. Tuple of predicted masks and used points gotten by point selection
        """
        if isinstance(reference_feats, np.ndarray):
            reference_feats = torch.as_tensor(reference_feats, device=self.device)
        if isinstance(used_indices, np.ndarray):
            used_indices = torch.as_tensor(used_indices, device=self.device)

        # preprocess images and prompts
        transformed_batch = [self.transforms(b.copy()) for b in batch]

        total_results: List[List[Tensor]] = []
        for tb in transformed_batch:
            # assign components
            images = tb["images"].unsqueeze(0).to(self.device)  # type: ignore[union-attr]
            original_size = torch.as_tensor(tb["original_size"])

            image_embeddings = self.image_encoder(images)
            total_points_scores, total_bg_coords = self.prompt_getter.get_prompt_candidates(
                image_embeddings=image_embeddings,
                reference_feats=reference_feats,
                used_indices=used_indices,
                original_size=original_size,
                device=self.device,
            )
            predicted_masks: defaultdict = defaultdict(list)
            used_points: defaultdict = defaultdict(list)
            for label in total_points_scores.keys():
                points_scores = total_points_scores[label]
                bg_coords = total_bg_coords[label]
                for points_score in points_scores:
                    x, y = points_score[:2]
                    is_done = False
                    for pm in predicted_masks.get(label, []):
                        # check if that point is already assigned
                        if pm[int(y), int(x)] > 0:
                            is_done = True
                            break
                    if is_done:
                        continue

                    point_coords = torch.cat((points_score[:2].unsqueeze(0), bg_coords), dim=0).unsqueeze(0)
                    point_coords = self._preprocess_coords(point_coords, original_size, self.config.model.image_size)
                    point_labels = torch.tensor(
                        [1] + [0] * len(bg_coords), dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    mask = self._predict_masks(
                        image_embeddings=image_embeddings,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        original_size=original_size,
                        is_cascade=is_cascade,
                    )
                    predicted_masks[label].append((mask * points_score[2]).detach().cpu())
                    used_points[label].append(points_score.detach().cpu())

            # check overlapping area between different label masks
            self._inspect_overlapping_areas(predicted_masks, used_points)
            total_results.append([predicted_masks, used_points])
        return total_results

    def _inspect_overlapping_areas(
        self,
        predicted_masks: Dict[int, List[Tensor]],
        used_points: Dict[int, List[Tensor]],
        threshold_iou: float = 0.8,
    ) -> None:
        def _calculate_mask_iou(mask1: Tensor, mask2: Tensor):
            assert mask1.ndim == 2 and mask2.ndim == 2
            intersection = torch.logical_and(mask1, mask2).sum().item()
            union = torch.logical_or(mask1, mask2).sum().item()

            # Avoid division by zero
            if union == 0:
                return 0.0
            iou = intersection / union
            return iou

        for (label, masks), (other_label, other_masks) in product(predicted_masks.items(), predicted_masks.items()):
            if other_label <= label:
                continue

            overlapped_label = []
            overlapped_other_label = []
            for (im, mask), (jm, other_mask) in product(enumerate(masks), enumerate(other_masks)):
                _mask_iou = _calculate_mask_iou(mask, other_mask)
                if _mask_iou > threshold_iou:
                    # compare overlapped regions between different labels and filter out the lower score
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        overlapped_other_label.append(jm)
                    else:
                        overlapped_label.append(im)
                elif _mask_iou > 0:
                    # refine the slightly overlapping region
                    overlapped_coords = torch.where(torch.logical_and(mask, other_mask))
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        other_mask[overlapped_coords] = 0.0
                    else:
                        mask[overlapped_coords] = 0.0

            for im in sorted(list(set(overlapped_label)), reverse=True):
                masks.pop(im)
                used_points[label].pop(im)

            for jm in sorted(list(set(overlapped_other_label)), reverse=True):
                other_masks.pop(jm)
                used_points[other_label].pop(jm)

    def _predict_masks(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        original_size: Tensor,
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
                mask_input = torch.zeros(1, 1, *map(lambda x: x * 4, image_embeddings.shape[2:]), device=self.device)
                has_mask_input = self.has_mask_inputs[0].to(self.device)

            elif i == 1:
                # Cascaded Post-refinement-1
                mask_input, masks = self._postprocess_masks(masks, logits, scores, is_single=True)  # noqa: F821
                if masks.sum() == 0:
                    return masks

                has_mask_input = self.has_mask_inputs[1].to(self.device)

            elif i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks = self._postprocess_masks(masks, logits, scores)  # noqa: F821
                if masks.sum() == 0:
                    return masks

                has_mask_input = self.has_mask_inputs[1].to(self.device)
                coords = torch.nonzero(masks)
                y, x = coords[:, 0], coords[:, 1]
                box_coords = self._preprocess_coords(
                    torch.as_tensor(
                        [[[x.min(), y.min()], [x.max(), y.max()]]], dtype=torch.float32, device=self.device
                    ),
                    original_size,
                    self.config.model.image_size,
                )
                point_coords = torch.cat((point_coords, box_coords), dim=1)
                point_labels = torch.cat((point_labels, self.point_labels_box.to(self.device)), dim=1)

            high_res_masks, scores, logits = self(
                image_embeddings=image_embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                has_mask_input=has_mask_input,
                orig_size=original_size.unsqueeze(0),
            )
            masks = high_res_masks > self.config.model.mask_threshold
        _, masks = self._postprocess_masks(masks, logits, scores)
        return masks

    def training_step(self, batch, batch_idx) -> None:
        """Training step for `learn`."""
        self.learn(batch)

    def predict_step(self, batch, batch_idx):
        """Predict step for `infer`."""
        results = self.infer(batch, self.reference_info["reference_feats"], self.reference_info["used_indices"])
        return [result[0] for result in results]  # tmp: only mask

    def _preprocess_prompts(self, batch: Dict[str, Any]) -> Dict[Any, Any]:
        """Preprocess prompts.

        Currently, preprocessing for bounding boxes is only supported.

        Args:
            batch (Dict[str, Any]): Dictionary containing data and prompts information.

        Returns:
            (Dict[Any, Any]): Processed and arranged each single prompt
                using label information as keys. Unlike other prompts, `annotation` prompts will be aggregated
                as single annotation.
        """
        processed_prompts = defaultdict(list)
        for prompt_name in ["annotations", "polygons", "bboxes", "points"]:
            prompts = batch.get(prompt_name, None)
            labels = batch["labels"].get(prompt_name, None)
            if prompts is None or len(prompts) == 0:
                continue
            for prompt, label in zip(prompts, labels):
                if isinstance(label, ScoredLabel):
                    label = int(label.id_)
                # TODO (sungchul): revisit annotations and polygons
                if prompt_name == "annotations":
                    processed_prompts[label].append({prompt_name: torch.as_tensor(prompt, device=self.device)})
                elif prompt_name == "polygons":
                    masks = []
                    for polygon in prompt:
                        contour = [[int(point[0]), int(point[1])] for point in polygon]
                        mask_from_polygon = np.zeros(batch["original_size"], dtype=np.uint8)
                        mask_from_polygon = cv2.drawContours(mask_from_polygon, np.asarray([contour]), 0, 1, -1)
                        masks.append(mask_from_polygon)
                    processed_prompts[label].append({prompt_name: torch.tensor(prompt, device=self.device)})
                elif prompt_name == "bboxes":
                    processed_prompts[label].append(
                        {
                            prompt_name: {
                                "point_coords": torch.as_tensor(prompt.reshape(-1, 2, 2), device=self.device),
                                "point_labels": torch.tensor([[1]], device=self.device),
                            }
                        }
                    )
                elif prompt_name == "points":
                    processed_prompts[label].append(
                        {
                            prompt_name: {
                                "point_coords": torch.as_tensor(prompt.reshape(-1, 2), device=self.device),
                                "point_labels": torch.tensor([[1]], device=self.device),
                            }
                        }
                    )

        processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x))  # type: ignore[assignment]
        return processed_prompts

    def _preprocess_coords(
        self,
        coords: Tensor,
        ori_shape: Union[List[int], Tuple[int, int], Tensor],
        target_length: int,
    ) -> Tensor:
        """Expects a torch tensor of length 2 in the final dimension.

        Requires the original image size in (H, W) format.

        Args:
            coords (Tensor): Coordinates tensor.
            ori_shape (Union[List[int], Tuple[int, int], Tensor]): Original size of image.
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
    ) -> Tuple[Tensor, ...]:
        """Generate masked features.

        Args:
            feats (Tensor): Raw reference features. It will be filtered with masks.
            masks (Tensor): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.

        Returns:
            (Tensor): Masked features.
        """
        scale_factor = self.config.model.image_size / max(masks.shape)

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
        masked_feat = masked_feat / masked_feat.norm(dim=-1, keepdim=True)

        return masked_feat

    def _pad_to_square(self, x: Tensor) -> Tensor:
        """Pad to a square input.

        Args:
            x (Tensor): Mask to be padded.

        Returns:
            (Tensor): Padded mask.
        """
        h, w = x.shape[-2:]
        padh = self.config.model.image_size - h
        padw = self.config.model.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _postprocess_masks(
        self,
        masks: Tensor,
        logits: Tensor,
        scores: Tensor,
        is_single: bool = False,
    ):
        """Post-process masks for cascaded post-refinements."""
        if is_single:
            best_idx = 0
        else:
            # skip the first index components
            scores, masks, logits = map(lambda x: x[:, 1:], (scores, masks, logits))

            # filter zero masks
            while len(scores[0]) > 0 and masks[0, (best_idx := torch.argmax(scores[0]))].sum() == 0:
                scores, masks, logits = map(
                    lambda x: torch.cat((x[:, :best_idx], x[:, best_idx + 1 :]), dim=1), (scores, masks, logits)
                )

            if len(scores[0]) == 0:
                # all predicted masks were zero masks, ignore them.
                return None, torch.zeros(masks.shape[-2:], device="cpu")

            best_idx = torch.argmax(scores[0])
        return logits[:, [best_idx]], masks[0, best_idx]

    def set_metrics(self) -> None:
        """Skip set_metrics unused in zero-shot learning."""
        pass

    def configure_optimizers(self) -> None:
        """Skip configure_optimizers unused in zero-shot learning."""
        pass

    def _find_latest_reference_info(self, root: str = "vpm_zsl_reference_infos") -> Union[str, None]:
        """Find latest reference info to be used."""
        if not os.path.isdir(root):
            return None
        if len(stamps := sorted(os.listdir(root), reverse=True)) > 0:
            return stamps[0]
        return None

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""
        self.initialize_reference_info()

    def on_predict_start(self) -> None:
        """Called at the beginning of predicting."""
        if (latest_stamp := self._find_latest_reference_info()) is not None:
            latest_reference_info = self.path_reference_info.format(latest_stamp)
            self.reference_info = torch.load(latest_reference_info)
            self.reference_info.to(self.device)
            logger.info(f"reference info saved at {latest_reference_info} was successfully loaded.")

    def training_epoch_end(self, outputs) -> None:
        """Called in the training loop at the very end of the epoch."""
        self.reference_info["used_indices"] = Parameter(
            self.reference_info["used_indices"].unique(), requires_grad=False
        )
        if self.config.model.save_outputs:
            path_reference_info = self.path_reference_info.format(time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(os.path.dirname(path_reference_info), exist_ok=True)
            torch.save(self.reference_info, path_reference_info)
            pickle.dump(
                {k: v.numpy() for k, v in self.reference_info.items()},
                open(path_reference_info.replace(".pt", ".pickle"), "wb"),
            )
            json.dump(
                repr(self.trainer.datamodule.train_dataset.dataset),
                open(path_reference_info.replace("reference_info.pt", "reference_meta.json"), "w"),
            )
            logger.info(f"Saved reference info at {path_reference_info}.")
