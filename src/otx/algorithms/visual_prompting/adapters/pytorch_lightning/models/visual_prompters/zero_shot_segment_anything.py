"""SAM module for visual prompting zero-shot learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import product
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import (
    ResizeLongestSide,
)
from otx.api.entities.scored_label import ScoredLabel
from otx.utils.logger import get_logger

from .segment_anything import SegmentAnything

logger = get_logger()


class PromptGetter(nn.Module):
    """Prompt getter for zero-shot learning."""

    default_threshold_reference = 0.3
    default_threshold_target = 0.65

    def __init__(
        self,
        image_size: int,
        reference_feats: Optional[torch.Tensor] = None,
        reference_prompts: Optional[torch.Tensor] = None,
        downsizing: int = 64,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.downsizing = downsizing
        self.initialize(reference_feats, reference_prompts)

        self.zero_tensor = torch.tensor(0)

    def initialize(
        self, reference_feats: Optional[torch.Tensor] = None, reference_prompts: Optional[torch.Tensor] = None
    ) -> None:
        """Initialize reference features and prompts."""
        self.reference_feats = reference_feats
        self.reference_prompts = reference_prompts

    def set_default_thresholds(self, default_threshold_reference: float, default_threshold_target: float) -> None:
        """Set default thresholds."""
        self.default_threshold_reference = default_threshold_reference
        self.default_threshold_target = default_threshold_target

    def set_reference(self, label: ScoredLabel, reference_feats: torch.Tensor, reference_prompts: torch.Tensor) -> None:
        """Set reference features and prompts."""
        if self.reference_feats is None:
            self.reference_feats = torch.zeros_like(reference_feats).unsqueeze(0)
        if self.reference_prompts is None:
            self.reference_prompts = torch.zeros_like(reference_prompts).unsqueeze(0)

        for idx in range(int(label.id_) + 1):
            if idx == int(label.id_):
                while self.reference_feats.shape[0] - 1 < idx:
                    self.reference_feats = torch.cat(
                        (self.reference_feats, torch.zeros_like(reference_feats).unsqueeze(0)), dim=0
                    )
                    self.reference_prompts = torch.cat(
                        (self.reference_prompts, torch.zeros_like(reference_prompts).unsqueeze(0)), dim=0
                    )
                self.reference_feats[idx] = reference_feats
                self.reference_prompts[idx] = reference_prompts

    def forward(
        self,
        image_embeddings: torch.Tensor,
        original_size: torch.Tensor,
        threshold: torch.Tensor = torch.tensor([[0.0]], dtype=torch.float32),
        num_bg_points: torch.Tensor = torch.tensor([[1]], dtype=torch.int64),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prompt candidates."""
        total_points_scores: torch.Tensor
        total_bg_coords: torch.Tensor

        device = image_embeddings.device
        threshold = threshold.to(device)
        for label in torch.arange(self.reference_feats.shape[0]):
            points_scores, bg_coords = self.get_prompt_candidates(
                image_embeddings=image_embeddings,
                label=label,
                original_size=original_size,
                threshold=threshold,
                num_bg_points=num_bg_points,
                device=device,
            )
            if label == 0:
                total_points_scores = points_scores.unsqueeze(0)
                total_bg_coords = bg_coords.unsqueeze(0)
            else:
                pad_size = torch.tensor(points_scores.shape[0] - total_points_scores.shape[1])
                pad_tot = torch.max(self.zero_tensor, pad_size)
                pad_cur = torch.max(self.zero_tensor, -pad_size)

                total_points_scores = F.pad(total_points_scores, (0, 0, 0, pad_tot, 0, 0), value=-1)
                points_scores = F.pad(points_scores, (0, 0, 0, pad_cur), value=-1)

                total_points_scores = torch.cat((total_points_scores, points_scores.unsqueeze(0)), dim=0)
                total_bg_coords = torch.cat((total_bg_coords, bg_coords.unsqueeze(0)), dim=0)

        return total_points_scores, total_bg_coords

    def get_prompt_candidates(
        self,
        image_embeddings: torch.Tensor,
        label: torch.Tensor,
        original_size: torch.Tensor,
        threshold: torch.Tensor = torch.tensor([[0.0]], dtype=torch.float32),
        num_bg_points: torch.Tensor = torch.tensor([[1]], dtype=torch.int64),
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prompt candidates from given reference and target features."""
        assert original_size.dim() == 2 and threshold.dim() == 2 and num_bg_points.dim() == 2

        target_feat = image_embeddings.squeeze()
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        sim = self.reference_feats[label].to(device) @ target_feat
        sim = sim.reshape(1, 1, h_feat, w_feat)
        sim = ZeroShotSegmentAnything.mask_postprocessing(sim, self.image_size, original_size[0])

        threshold = (threshold == 0) * self.default_threshold_target + threshold
        points_scores, bg_coords = self._point_selection(
            mask_sim=sim[0, 0],
            original_size=original_size[0],
            threshold=threshold,
            num_bg_points=num_bg_points,
        )

        return points_scores, bg_coords

    def _point_selection(
        self,
        mask_sim: torch.Tensor,
        original_size: torch.Tensor,
        threshold: torch.Tensor,
        num_bg_points: torch.Tensor = torch.tensor([[1]], dtype=torch.int64),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select point used as point prompts."""
        _, w_sim = mask_sim.shape

        # Top-last point selection
        bg_indices = mask_sim.flatten().topk(num_bg_points[0, 0], largest=False)[1]
        bg_x = (bg_indices // w_sim).unsqueeze(0)
        bg_y = bg_indices - bg_x * w_sim
        bg_coords = torch.cat((bg_y, bg_x), dim=0).permute(1, 0)
        bg_coords = bg_coords.to(torch.float32)

        point_coords = torch.where(mask_sim > threshold)
        fg_coords_scores = torch.stack(point_coords[::-1] + (mask_sim[point_coords],), dim=0).T

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
        points_scores = matched_grid[matched_grid[..., -1].argsort(dim=0, descending=True)[0]].diagonal().T

        # sort by the highest score
        points_scores = points_scores[torch.argsort(points_scores[:, -1], descending=True)]

        return points_scores, bg_coords


class ZeroShotSegmentAnything(SegmentAnything):
    """Zero-shot learning module using Segment Anything."""

    def __init__(self, config: Optional[DictConfig] = None, state_dict: Optional[OrderedDict] = None) -> None:
        if config is None:
            config = self.set_default_config()

        if not config.model.freeze_image_encoder:
            logger.warning("config.model.freeze_image_encoder(=False) must be set to True, changed.")
            config.model.freeze_image_encoder = True

        if not config.model.freeze_prompt_encoder:
            logger.warning("config.model.freeze_prompt_encoder(=False) must be set to True, changed.")
            config.model.freeze_prompt_encoder = True

        if not config.model.freeze_mask_decoder:
            logger.warning("config.model.freeze_mask_decoder(=False) must be set to True, changed.")
            config.model.freeze_mask_decoder = True

        prompt_getter_reference_feats = None
        prompt_getter_reference_prompts = None
        if state_dict:
            if "prompt_getter.reference_feats" in state_dict:
                prompt_getter_reference_feats = state_dict.pop("prompt_getter.reference_feats")
            if "prompt_getter.reference_prompts" in state_dict:
                prompt_getter_reference_prompts = state_dict.pop("prompt_getter.reference_prompts")

        super().__init__(config, state_dict)

        self.prompt_getter = PromptGetter(
            image_size=config.model.image_size,
            reference_feats=prompt_getter_reference_feats,
            reference_prompts=prompt_getter_reference_prompts,
        )
        self.prompt_getter.set_default_thresholds(
            default_threshold_reference=config.model.default_threshold_reference,
            default_threshold_target=config.model.default_threshold_target,
        )

        self.point_labels_box = torch.tensor([[2, 3]], dtype=torch.float32)
        self.has_mask_inputs = [torch.tensor([[0.0]]), torch.tensor([[1.0]])]

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
                }
            }
        )

    @torch.no_grad()
    def learn(
        self,
        images: torch.Tensor,
        processed_prompts: Dict[ScoredLabel, List[Dict[str, torch.Tensor]]],
        padding: Union[Tuple[int, ...], torch.Tensor],
        original_size: torch.Tensor,
    ) -> None:
        """Get reference features.

        Using given images, get reference features and save it to PromptGetter.
        These reference features will be used for `infer` to get target results.
        Currently, single batch is only supported.

        Args:
            images (torch.Tensor): Given images for reference features.
            processed_prompts (Dict[ScoredLabel, List[Dict[str, torch.Tensor]]]): The whole class-wise prompts
                processed at _preprocess_prompts.
            padding (Union[Tuple[int, ...], torch.Tensor]): Padding size.
            original_size (torch.Tensor): Original image size.
        """
        assert images.shape[0] == 1, "Only single batch is supported."

        self.prompt_getter.initialize()

        image_embeddings = self.image_encoder(images)
        ref_feat = image_embeddings.squeeze().permute(1, 2, 0)

        for label, input_prompts in processed_prompts.items():
            if label.name.lower() == "background":
                # skip background
                # TODO (sungchul): how to skip background class
                continue

            # generate reference mask
            # TODO (sungchul): ensemble multi reference features (current : use merged masks)
            reference_prompt = torch.zeros(*map(int, original_size), dtype=torch.uint8, device=self.device)
            for input_prompt in input_prompts:
                if "annotation" in input_prompt:
                    # directly use annotation information as a mask
                    reference_prompt[input_prompt.get("annotation") == 1] += 1
                else:
                    merged_input_prompts = self._merge_prompts(label, input_prompt, processed_prompts)
                    # TODO (sungchul): they must be processed in `_merge_prompts`
                    # and it is required to be expanded to other prompts.
                    point_coords = []
                    point_labels = []
                    if "box" in merged_input_prompts:
                        for box in merged_input_prompts["box"]:
                            point_coords.append(box[:2])
                            point_labels.append(2)
                            point_coords.append(box[2:])
                            point_labels.append(3)

                    if "points" in merged_input_prompts:
                        raise NotImplementedError()

                    if "annotations" in merged_input_prompts:
                        raise NotImplementedError()

                    point_coords = torch.stack(point_coords, dim=0).unsqueeze(0)
                    point_labels = torch.tensor([point_labels], device=self.device)
                    masks = self._predict_masks(
                        image_embeddings=image_embeddings,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        original_size=original_size,
                        is_cascade=False,
                    )
                    reference_prompt[masks] += 1
            reference_prompt = torch.clip(reference_prompt, 0, 1)

            ref_mask = reference_prompt.to(torch.float32)
            reference_feat = None
            default_threshold_reference = deepcopy(self.prompt_getter.default_threshold_reference)
            while reference_feat is None:
                logger.info(f"[*] default_threshold_reference : {default_threshold_reference:.4f}")
                reference_feat = self._generate_masked_features(
                    ref_feat, ref_mask, default_threshold_reference, padding=padding
                )
                default_threshold_reference -= 0.05

            self.prompt_getter.set_reference(label, reference_feat, reference_prompt)

    @torch.no_grad()
    def infer(
        self, images: torch.Tensor, original_size: torch.Tensor
    ) -> List[List[DefaultDict[int, List[torch.Tensor]]]]:
        """Zero-shot inference with reference features.

        Get target results by using reference features and target images' features.

        Args:
            images (torch.Tensor): Given images for target results.
            original_size (torch.Tensor): Original image size.

        Returns:
            (List[List[DefaultDict[int, List[torch.Tensor]]]]): Target results.
                Lists wrapping results is following this order:
                    1. Target images
                    2. Tuple of predicted masks and used points gotten by point selection
        """
        assert images.shape[0] == 1, "Only single batch is supported."

        total_results = []
        for image in images:
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image_embeddings = self.image_encoder(images)

            total_points_scores, total_bg_coords = self.prompt_getter(
                image_embeddings=image_embeddings, original_size=original_size
            )
            predicted_masks: defaultdict = defaultdict(list)
            used_points: defaultdict = defaultdict(list)
            for label, (points_scores, bg_coords) in enumerate(zip(total_points_scores, total_bg_coords)):
                for points_score in points_scores:
                    if points_score[-1] == -1:
                        continue
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
                    point_coords = ResizeLongestSide.apply_coords(
                        point_coords, original_size[0], self.config.model.image_size
                    )
                    point_labels = torch.tensor(
                        [1] + [0] * len(bg_coords), dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    mask = self._predict_masks(
                        image_embeddings=image_embeddings,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        original_size=original_size[0],
                    )
                    predicted_masks[label].append((mask * points_score[2]).detach().cpu())
                    used_points[label].append(points_score.detach().cpu())

            # check overlapping area between different label masks
            self.__inspect_overlapping_areas(predicted_masks, used_points)
            total_results.append([predicted_masks, used_points])
        return total_results
    
    def __inspect_overlapping_areas(self, predicted_masks: Dict[int, List[torch.Tensor]], used_points: Dict[int, List[torch.Tensor]], threshold_iou: float = 0.8):
        def __calculate_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor):
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
                if __calculate_mask_iou(mask, other_mask) > threshold_iou:
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        overlapped_other_label.append(jm)
                    else:
                        overlapped_label.append(im)

            for im in overlapped_label[::-1]:
                masks.pop(im)
                used_points[label].pop(im)

            for jm in overlapped_other_label[::-1]:
                other_masks.pop(jm)
                used_points[other_label].pop(jm)

    def _predict_masks(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        original_size: torch.Tensor,
        is_cascade: bool = True,
    ) -> torch.Tensor:
        """Predict target masks."""
        logits: torch.Tensor
        scores: torch.Tensor
        for i in range(3):
            if i == 0:
                # First-step prediction
                mask_input = torch.zeros(1, 1, *map(lambda x: x * 4, image_embeddings.shape[2:]), device=self.device)
                has_mask_input = self.has_mask_inputs[0].to(self.device)

            elif is_cascade and i == 1:
                # Cascaded Post-refinement-1
                mask_input, masks = self._postprocess_masks(logits, scores, original_size, is_single=True)  # noqa: F821
                if masks.sum() == 0:
                    return masks

                has_mask_input = self.has_mask_inputs[1].to(self.device)

            elif is_cascade and i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks = self._postprocess_masks(logits, scores, original_size)  # noqa: F821
                if masks.sum() == 0:
                    return masks

                has_mask_input = self.has_mask_inputs[1].to(self.device)
                coords = torch.nonzero(masks)
                y, x = coords[:, 0], coords[:, 1]
                box_coords = ResizeLongestSide.apply_coords(
                    torch.tensor([[[x.min(), y.min()], [x.max(), y.max()]]], dtype=torch.float32, device=self.device),
                    original_size,
                    self.config.model.image_size,
                )
                point_coords = torch.cat((point_coords, box_coords), dim=1)
                point_labels = torch.cat((point_labels, self.point_labels_box.to(self.device)), dim=1)

            scores, logits = self(
                image_embeddings=image_embeddings,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                has_mask_input=has_mask_input,
            )

        _, masks = self._postprocess_masks(logits, scores, original_size)
        return masks

    def training_step(self, batch, batch_idx) -> None:
        """Training step for `learn`."""
        # TODO (sungchul): each prompt will be assigned with each label
        bboxes = batch["bboxes"]
        labels = batch["labels"]
        # TODO (sungchul): support other below prompts
        # points = batch["points"]
        # annotations = batch["annotations"]

        # organize prompts based on label
        processed_prompts = self._preprocess_prompts(bboxes=bboxes[0], labels=labels[0])

        self.learn(
            images=batch["images"],
            processed_prompts=processed_prompts,
            padding=batch.get("padding")[0],
            original_size=batch.get("original_size")[0],
        )

    def predict_step(self, batch, batch_idx):
        """Predict step for `infer`."""
        results = self.infer(images=batch["images"], original_size=batch.get("original_size")[0].unsqueeze(0))
        return [result[0] for result in results]  # tmp: only mask

    def _preprocess_prompts(
        self,
        bboxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        annotations: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[ScoredLabel, List[Dict[str, torch.Tensor]]]:
        """Preprocess prompts.

        Currently, preprocessing for bounding boxes is only supported.

        Args:
            bboxes (torch.Tensor, optional): Bounding box prompts to be preprocessed.
            points (torch.Tensor, optional): Point prompts to be preprocessed, to be supported.
            annotations (torch.Tensor, optional): annotation prompts to be preprocessed, to be supported.
            labels (torch.Tensor, optional): Assigned labels according to given prompts.
                Currently, it is only matched to bboxes, and it will be deprecated.

        Returns:
            (defaultdict[ScoredLabel, List[Dict[str, torch.Tensor]]]): Processed and arranged each single prompt
                using label information as keys. Unlike other prompts, `annotation` prompts will be aggregated
                as single annotation.
        """
        processed_prompts = defaultdict(list)
        # TODO (sungchul): will be updated
        if bboxes is not None:
            for bbox, label in zip(bboxes, labels):
                processed_prompts[label].append({"box": bbox.reshape(-1, 4)})

        if points:
            pass

        if annotations:
            pass

        processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x[0].id_))  # type: ignore[assignment]
        return processed_prompts

    def _generate_masked_features(
        self,
        feats: torch.Tensor,
        masks: torch.Tensor,
        threshold_mask: float,
        padding: Optional[Union[Tuple[int, ...], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Generate masked features.

        Args:
            feats (torch.Tensor): Raw reference features. It will be filtered with masks.
            masks (torch.Tensor): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.
            padding (Union[Tuple[int, ...], torch.Tensor], optional): Padding size.

        Returns:
            (torch.Tensor): Masked features.
        """
        if padding:
            resized_size = (
                self.config.model.image_size - padding[1] - padding[3],
                self.config.model.image_size - padding[0] - padding[2],
            )
        else:
            resized_size = (self.config.model.image_size, self.config.model.image_size)

        # Post-process masks
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=resized_size, mode="bilinear").squeeze()
        masks = self._preprocess_masks(masks)
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=feats.shape[0:2], mode="bilinear").squeeze()

        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None, None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0).unsqueeze(0)
        masked_feat = masked_feat / masked_feat.norm(dim=-1, keepdim=True)

        return masked_feat

    def _preprocess_masks(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input.

        Args:
            x (torch.Tensor): Mask to be padded.

        Returns:
            (torch.Tensor): Padded mask.
        """
        # Pad
        h, w = x.shape[-2:]
        padh = self.config.model.image_size - h
        padw = self.config.model.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _postprocess_masks(
        self,
        logits: torch.Tensor,
        scores: torch.Tensor,
        original_size: torch.Tensor,
        is_single: bool = False,
    ):
        """Post-process masks for cascaded post-refinements."""
        high_res_masks = self.mask_postprocessing(logits, self.config.model.image_size, original_size)
        masks = high_res_masks > self.config.model.mask_threshold

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
                return None, torch.zeros((self.config.model.image_size, self.config.model.image_size), device="cpu")

            best_idx = torch.argmax(scores[0])
        return logits[:, best_idx], masks[0, best_idx]

    def _update_value(self, target: Dict[str, Any], key: str, value: torch.Tensor) -> None:
        """Update tensor to target dictionary.

        Args:
            target (Dict[str, Any]): Target dictionary to be updated.
            key (str): Key to be used for update.
            value (torch.Tensor): Value to be used for update.
        """
        if key in target:
            target[key] = torch.cat((target[key], value))
        else:
            target[key] = value

    def _merge_prompts(
        self,
        label: ScoredLabel,
        input_prompts: Dict[str, torch.Tensor],
        processed_prompts: Dict[ScoredLabel, List[Dict[str, torch.Tensor]]],
        use_only_background: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Merge target prompt and other prompts.

        Merge a foreground prompt and other prompts (background or prompts with other classes).

        Args:
            label (ScoredLabel): Label information. Background is 0 and other foregrounds are >= 0.
            input_prompts (Dict[str, torch.Tensor]): A foreground prompt to be merged with other prompts.
            processed_prompts (Dict[ScoredLabel, List[Dict[str, torch.Tensor]]]): The whole class-wise prompts
                processed at _preprocess_prompts.
            use_only_background (bool): Whether merging only background prompt, defaults to True.
                It is applied to only point_coords.

        Returns:
            (Dict[str, torch.Tensor]): Merged prompts.
        """
        merged_input_prompts = deepcopy(input_prompts)
        for other_label, other_input_prompts in processed_prompts.items():
            if other_label.id_ == label.id_:
                continue
            if (use_only_background and other_label.id_ == 0) or (not use_only_background):
                # only add point (and scribble) prompts
                # use_only_background=True -> background prompts are only added as background
                # use_only_background=False -> other prompts are added as background
                for other_input_prompt in other_input_prompts:
                    if "point_coords" in other_input_prompt:
                        # point, scribble
                        self._update_value(merged_input_prompts, "point_coords", other_input_prompt.get("point_coords"))
                        self._update_value(
                            merged_input_prompts,
                            "point_labels",
                            torch.zeros_like(other_input_prompt.get("point_labels")),
                        )
        return merged_input_prompts

    def set_metrics(self) -> None:
        """Skip set_metrics unused in zero-shot learning."""
        pass

    def configure_optimizers(self) -> None:
        """Skip configure_optimizers unused in zero-shot learning."""
        pass

    def training_epoch_end(self, outputs) -> None:
        """Skip training_epoch_end unused in zero-shot learning."""
        pass
