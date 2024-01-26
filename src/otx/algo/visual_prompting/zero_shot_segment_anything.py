# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything model for the OTX zero-shot visual prompting."""

from __future__ import annotations

import logging as log
from copy import deepcopy
from typing import Tuple, Any, Dict, List, DefaultDict

import torch
from torch import nn, LongTensor, Tensor
from torch.nn import functional as F
from torchvision import tv_tensors
from collections import defaultdict
from datumaro import Polygon as dmPolygon
from itertools import product

from otx.algo.visual_prompting.segment_anything import SegmentAnything, OTXSegmentAnything, DEFAULT_CONFIG_SEGMENT_ANYTHING
from otx.core.data.entity.base import OTXBatchLossEntity, Points
from otx.core.data.entity.visual_prompting import ZeroShotVisualPromptingBatchDataEntity, ZeroShotVisualPromptingBatchPredEntity


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

    def forward(
        self,
        image_embedding: Tensor,
        reference_feats: Tensor,
        used_indices: list[int],
        ori_shape: Tensor,
        threshold: Tensor | None = None,
        num_bg_points: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Get prompt candidates."""
        if threshold is None:
            threshold = torch.tensor([[0.0]], dtype=torch.float32)
        if num_bg_points is None:
            num_bg_points = torch.tensor([[1]], dtype=torch.int64)

        device = image_embedding.device
        threshold = threshold.to(device)

        total_points_scores: Tensor = torch.zeros(max(used_indices)+1, 0, 3, device=device)
        total_bg_coords: Tensor = torch.zeros(max(used_indices)+1, num_bg_points, 2, device=device)
        for label in used_indices:
            points_scores, bg_coords = self._get_prompt_candidates(
                image_embedding=image_embedding,
                reference_feat=reference_feats[label],
                ori_shape=ori_shape,
                threshold=threshold,
                num_bg_points=num_bg_points,
                device=device,
            )
            
            pad_size = torch.tensor(points_scores.shape[0] - total_points_scores.shape[1])
            pad_tot = torch.max(self.zero_tensor, pad_size)
            pad_cur = torch.max(self.zero_tensor, -pad_size)

            total_points_scores = F.pad(total_points_scores, (0, 0, 0, pad_tot, 0, 0), value=-1)
            points_scores = F.pad(points_scores, (0, 0, 0, pad_cur), value=-1)

            total_points_scores[label] = points_scores
            total_bg_coords[label] = bg_coords

        return total_points_scores, total_bg_coords

    def _get_prompt_candidates(
        self,
        image_embedding: Tensor,
        reference_feat: Tensor,
        ori_shape: Tensor,
        threshold: Tensor,
        num_bg_points: Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Tensor, Tensor]:
        """Get prompt candidates from given reference and target features."""
        assert threshold.dim() == 2 and num_bg_points.dim() == 2

        target_feat = image_embedding.squeeze()
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        sim = reference_feat.to(device) @ target_feat
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
        threshold: Tensor,
        num_bg_points: Tensor,
    ) -> Tuple[Tensor, Tensor]:
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

        ratio = self.image_size / ori_shape.max()
        width = (ori_shape[1] * ratio).to(torch.int64)
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

    def __init__(
        self,
        default_threshold_reference: float = 0.3,
        default_threshold_target: float = 0.65,
        *args, **kwargs
    ) -> None:
        msg = ""
        if len(kwargs) == 0:
            msg += "There isn't any given argument. Default setting will be used."
        elif len(kwargs) == 1 and "backbone" in kwargs:
            msg += (
                f"There is only backbone (={kwargs.get('backbone')}) argument. "
                f"Other parameters will be set along with backbone (={kwargs.get('backbone')}).")
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

        super().__init__(*args, **kwargs)

        self.prompt_getter = PromptGetter(image_size=self.image_size)
        self.prompt_getter.set_default_thresholds(default_threshold_reference=default_threshold_reference, default_threshold_target=default_threshold_target)

        self.point_labels_box = torch.tensor([[2, 3]], dtype=torch.float32)
        self.has_mask_inputs = [torch.tensor([[0.0]]), torch.tensor([[1.0]])]
        
        self.initialize_reference_info()
        
    def set_default_config(self, **kwargs) -> dict[str, Any]:
        """Set default config when using independently."""
        backbone = kwargs.get("backbone", "tiny_vit")
        kwargs.update({
            "backbone": backbone,
            "load_from": kwargs.get("load_from", DEFAULT_CONFIG_SEGMENT_ANYTHING[backbone]["load_from"]),
            "freeze_image_encoder": kwargs.get("freeze_image_encoder", True),
            "freeze_mask_decoder": kwargs.get("freeze_mask_decoder", True),
            "freeze_prompt_encoder": kwargs.get("freeze_prompt_encoder", True),
        })
        return kwargs
        
    def initialize_reference_info(self) -> None:
        """Initialize reference information."""
        self.reference_feats: Tensor = None
        self.reference_masks: list[Tensor] = None
        self.used_indices: DefaultDict[int, list[int]] = None
        
    def check_reference_info_is_empty(self) -> None:
        """Check if reference information is empty."""
        return (
            self.reference_feats is None and
            self.reference_masks is None and
            self.used_indices is None
        )
        
    @torch.no_grad()
    def learn(
        self,
        images: list[tv_tensors.Image],
        processed_prompts: list[dict[int, list[tv_tensors.TVTensor]]],
        ori_shapes: list[Tensor],
        return_outputs: bool = False,
        is_init_ref_info: bool = True,
    ) -> tuple[Tensor, list[Tensor], defaultdict[int, list[int]]] | None:
        """Get reference features.

        Using given images, get reference features.
        These reference features will be used for `infer` to get target results.
        Currently, single batch is only supported.

        Args:
            images (list[tv_tensors.Image]): List of given images for reference features.
            processed_prompts (dict[int, list[tv_tensors.TVTensor]]): The class-wise prompts
                processed at _preprocess_prompts.
            ori_shapes (List[Tensor]): List of original shapes per image.
            return_outputs (bool): Whether return reference features and masks.
                If True, `learn` operation will output reference features and masks.
                If False, `learn` operation will insert reference features and masks to save them with the model.
                Defaults to False.
            is_init_ref_info (bool): Whether initialize reference features and masks.
        """
        assert len(images) == 1, "Only single batch is supported."

        # initialize tensors to contain reference features and prompts
        if self.check_reference_info_is_empty() or is_init_ref_info:
            largest_label = max(sum([[int(p) for p in prompt] for prompt in processed_prompts], []))
            self.reference_feats = torch.zeros(len(images), largest_label + 1, 1, self.embed_dim)
            self.reference_masks = [torch.zeros(largest_label + 1, *map(int, ori_shape)) for ori_shape in ori_shapes]
            self.used_indices = defaultdict(list)
        else:
            # TODO(sungchul): expand axis if there are new labels
            # TODO(sungchul): consider who to handle multiple reference features, currently replace it
            return

        for batch, (image, prompts, ori_shape) in enumerate(zip(images, processed_prompts, ori_shapes)):
            image_embedding = self.image_encoder(image)
            processed_embedding = image_embedding.squeeze().permute(1, 2, 0)

            for label, input_prompts in prompts.items():
                # TODO (sungchul): how to skip background class
                # TODO (sungchul): ensemble multi reference features (current : use merged masks)
                ref_mask = torch.zeros(*map(int, ori_shape), dtype=torch.uint8, device=image.device)
                for input_prompt in input_prompts:
                    if isinstance(input_prompt, tv_tensors.Mask):
                        # directly use annotation information as a mask
                        ref_mask[input_prompt == 1] += 1 # TODO(sungchul): check if the mask is bool or int
                    else:
                        if isinstance(input_prompt, tv_tensors.BoundingBoxes):
                            point_coords = input_prompt.reshape(1, 2, 2)
                            point_labels = torch.tensor([[2, 3]], device=point_coords.device)
                        elif isinstance(input_prompt, Points):
                            point_coords = input_prompt.reshape(1, 1, 2)
                            point_labels = torch.tensor([[1]], device=point_coords.device)
                        elif isinstance(input_prompt, dmPolygon): # TODO(sungchul): add other polygon types
                            # TODO(sungchul): convert polygon to mask
                            continue
                        else:
                            log.info(f"Current input prompt ({input_prompt.__class__.__name__}) is not supported.")
                            continue

                        masks = self._predict_masks(
                            mode="learn",
                            image_embedding=image_embedding,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            ori_shape=ori_shape,
                            is_cascade=False,
                        )
                        ref_mask[masks] += 1

                ref_mask = torch.clip(ref_mask, 0, 1).to(torch.float32)

                ref_feat = None
                default_threshold_reference = deepcopy(self.prompt_getter.default_threshold_reference)
                while ref_feat is None:
                    log.info(f"[*] default_threshold_reference : {default_threshold_reference:.4f}")
                    ref_feat = self._generate_masked_features(processed_embedding, ref_mask, default_threshold_reference)
                    default_threshold_reference -= 0.05

                self.reference_feats[batch][label] = ref_feat.detach().cpu()
                self.reference_masks[batch][label] = ref_mask.detach().cpu()
                self.used_indices[batch].append(label)

        if return_outputs:
            return self.reference_feats, self.reference_masks, self.used_indices

    @torch.no_grad()
    def infer(
        self,
        images: list[tv_tensors.Image],
        reference_feats: Tensor,
        used_indices: dict[int, list[int]],
        ori_shapes: list[Tensor]
    ) -> list[list[defaultdict[int, list[Tensor]]]]:
        """Zero-shot inference with reference features.

        Get target results by using reference features and target images' features.

        Args:
            images (list[tv_tensors.Image]): Given images for target results.
            reference_feats (Tensor): Reference features for target prediction.
            used_indices (dict[int, list[int]]): To check which indices of reference features are validate.
            ori_shapes (list[Tensor]): Original image size.

        Returns:
            (list[list[defaultdict[int, list[Tensor]]]]): List of predicted masks and used points.
        """
        assert len(images) == 1, "Only single batch is supported."

        total_results = []
        for batch, (image, ori_shape) in enumerate(zip(images, ori_shapes)):
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image_embedding = self.image_encoder(image)

            total_points_scores, total_bg_coords = self.prompt_getter(
                image_embedding=image_embedding,
                reference_feats=reference_feats[batch],
                used_indices=used_indices[batch],
                ori_shape=ori_shape
            )
            predicted_masks: defaultdict = defaultdict(list)
            used_points: defaultdict = defaultdict(list)
            for label in used_indices[batch]:
                points_scores, bg_coords = total_points_scores[label], total_bg_coords[label]
                for point_score in points_scores:
                    if point_score[-1] == -1:
                        continue
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
                        [1] + [0] * len(bg_coords), dtype=torch.float32, device=point_coords.device
                    ).unsqueeze(0)
                    mask = self._predict_masks(
                        mode="infer",
                        image_embedding=image_embedding,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        ori_shape=ori_shape,
                    )
                    predicted_masks[label].append((mask * point_score[2]))
                    used_points[label].append(point_score)

            # check overlapping area between different label masks
            self.__inspect_overlapping_areas(predicted_masks, used_points)
            total_results.append([predicted_masks, used_points])
        return total_results

    def __inspect_overlapping_areas(
        self,
        predicted_masks: Dict[Tensor, List[Tensor]],
        used_points: Dict[Tensor, List[Tensor]],
        threshold_iou: float = 0.8,
    ):
        def __calculate_mask_iou(mask1: Tensor, mask2: Tensor):
            assert mask1.ndim == 2 and mask2.ndim == 2
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
        mode: str,
        image_embedding: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        ori_shape: Tensor,
        is_cascade: bool = True,
    ) -> Tensor:
        """Predict target masks."""
        logits: Tensor
        scores: Tensor
        for i in range(3):
            if i == 0:
                # First-step prediction
                mask_input = torch.zeros(1, 1, *map(lambda x: x * 4, image_embedding.shape[2:]), device=image_embedding.device)
                has_mask_input = self.has_mask_inputs[0].to(mask_input.device)

            elif is_cascade and i == 1:
                # Cascaded Post-refinement-1
                mask_input, masks = self._decide_cascade_results(masks, logits, scores, is_single=True)  # noqa: F821
                if masks.sum() == 0:
                    return masks

                has_mask_input = self.has_mask_inputs[1].to(mask_input.device)

            elif is_cascade and i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks = self._decide_cascade_results(masks, logits, scores)  # noqa: F821
                if masks.sum() == 0:
                    return masks

                has_mask_input = self.has_mask_inputs[1].to(mask_input.device)
                coords = torch.nonzero(masks)
                y, x = coords[:, 0], coords[:, 1]
                box_coords = self._preprocess_coords(torch.tensor([[[x.min(), y.min()], [x.max(), y.max()]]], dtype=torch.float32, device=coords.device), ori_shape, self.image_size)
                point_coords = torch.cat((point_coords, box_coords), dim=1)
                point_labels = torch.cat((point_labels, self.point_labels_box.to(point_labels.device)), dim=1)

            high_res_masks, scores, logits = self(
                mode=mode,
                image_embeddings=image_embedding,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                has_mask_input=has_mask_input,
                ori_shape=ori_shape,
            )
            masks = high_res_masks > self.mask_threshold

        _, masks = self._decide_cascade_results(masks, logits, scores)
        return masks
    
    def _preprocess_coords(
        self,
        coords: Tensor,
        ori_shape: List[int] | Tuple[int, int] | Tensor,
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
    ) -> Tuple[Tensor, ...]:
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
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _decide_cascade_results(
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
                return None, torch.zeros((self.image_size, self.image_size), device="cpu")

            best_idx = torch.argmax(scores[0])
        return logits[:, best_idx], masks[0, best_idx]


class OTXZeroShotSegmentAnything(OTXSegmentAnything):
    """Zero-Shot Visual Prompting model."""
    
    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""
        return ZeroShotSegmentAnything(**self.config)

    def forward(self, inputs: ZeroShotVisualPromptingBatchDataEntity) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        forward_fn = self.model.learn if self.training else self.model.infer

        outputs = forward_fn(**self._customize_inputs(inputs))

        return self._customize_outputs(outputs, inputs)

    def _customize_inputs(self, inputs: ZeroShotVisualPromptingBatchDataEntity) -> dict[str, Any]:
        """Customize the inputs for the model."""
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
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ZeroShotVisualPromptingBatchDataEntity,
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
                masks.append(tv_tensors.Mask(torch.stack(predicted_mask, dim=0), dtype=torch.float32))
                prompts.append(Points(torch.stack([p[:2] for p in used_points[label]], dim=0), canvas_size=inputs.imgs_info[0].ori_shape, dtype=torch.float32))
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
        labels: list[Tensor]
    ) -> list[dict[int, list[tv_tensors.TVTensor]]]:
        """Gather prompts according to labels."""
        total_processed_prompts = []
        for prompt, label in zip(prompts, labels):
            processed_prompts = defaultdict(list)
            for p, l in zip(prompt, label):
                processed_prompts[int(l)].append(p)
            processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x))
            total_processed_prompts.append(processed_prompts)
        return total_processed_prompts
