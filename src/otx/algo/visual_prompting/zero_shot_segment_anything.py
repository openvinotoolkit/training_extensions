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

from otx.algo.visual_prompting.segment_anything import SegmentAnything, OTXSegmentAnything
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
        image_embeddings: Tensor,
        reference_feats: Tensor,
        original_size: Tensor,
        threshold: Tensor | None = None,
        num_bg_points: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Get prompt candidates."""
        if threshold is None:
            threshold = torch.tensor([[0.0]], dtype=torch.float32)
        if num_bg_points is None:
            num_bg_points = torch.tensor([[1]], dtype=torch.int64)

        total_points_scores: Tensor
        total_bg_coords: Tensor

        device = image_embeddings.device
        threshold = threshold.to(device)
        for label in torch.arange(reference_feats.shape[0]):
            points_scores, bg_coords = self._get_prompt_candidates(
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

    def _get_prompt_candidates(
        self,
        image_embeddings: Tensor,
        label: Tensor,
        original_size: Tensor,
        threshold: Tensor,
        num_bg_points: Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Tensor, Tensor]:
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
        mask_sim: Tensor,
        original_size: Tensor,
        threshold: Tensor,
        num_bg_points: Tensor = torch.tensor([[1]], dtype=torch.int64),
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

    def __init__(self, *args, **kwargs) -> None:
        default_threshold_reference = kwargs.pop("default_threshold_reference")
        default_threshold_target = kwargs.pop("default_threshold_target")
        
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
        
    @torch.no_grad()
    def learn(
        self,
        images: list[tv_tensors.Image],
        processed_prompts: list[dict[LongTensor, list[tv_tensors.TVTensor]]],
        ori_shapes: list[Tensor],
        return_outputs: bool = False,
    ) -> None:
        """Get reference features.

        Using given images, get reference features.
        These reference features will be used for `infer` to get target results.
        Currently, single batch is only supported.

        Args:
            images (list[tv_tensors.Image]): List of given images for reference features.
            processed_prompts (dict[LongTensor, list[tv_tensors.TVTensor]]): The class-wise prompts
                processed at _preprocess_prompts.
            ori_shapes (List[Tensor]): List of original shapes per image.
            return_outputs (bool): Whether return reference features and prompts.
                If True, `learn` operation will output reference features and prompts.
                If False, `learn` operation will insert reference features and prompts to save them with the model.
                Defaults to False.
        """
        assert len(images) == 1, "Only single batch is supported."

        # initialize tensors to contain reference features and prompts
        largest_label = max(sum([[int(p) for p in prompt] for prompt in processed_prompts], []))
        self.reference_feats = torch.zeros(len(images), largest_label + 1, 1, self.embed_dim)
        self.reference_masks = [torch.zeros(largest_label + 1, *map(int, ori_shape)) for ori_shape in ori_shapes]
        self.used_indices = defaultdict(list)
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

                self.reference_feats[batch][int(label)] = ref_feat.detach().cpu()
                self.reference_masks[batch][int(label)] = ref_mask.detach().cpu()
                self.used_indices[batch].append(int(label))

        if return_outputs:
            return self.reference_feats, self.reference_masks, self.used_indices

    @torch.no_grad()
    def infer(
        self, images: Tensor, original_size: Tensor
    ) -> List[List[DefaultDict[int, List[Tensor]]]]:
        """Zero-shot inference with reference features.

        Get target results by using reference features and target images' features.

        Args:
            images (Tensor): Given images for target results.
            original_size (Tensor): Original image size.

        Returns:
            (List[List[DefaultDict[int, List[Tensor]]]]): Target results.
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
                        point_coords, original_size[0], self.image_size
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

    def __inspect_overlapping_areas(
        self,
        predicted_masks: Dict[int, List[Tensor]],
        used_points: Dict[int, List[Tensor]],
        threshold_iou: float = 0.8,
    ):
        def __calculate_mask_iou(mask1: Tensor, mask2: Tensor):
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
                mask_input, masks = self._postprocess_masks(logits, scores, ori_shape, is_single=True)  # noqa: F821
                if masks.sum() == 0:
                    return masks

                has_mask_input = self.has_mask_inputs[1].to(mask_input.device)

            elif is_cascade and i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks = self._postprocess_masks(logits, scores, ori_shape)  # noqa: F821
                if masks.sum() == 0:
                    return masks

                has_mask_input = self.has_mask_inputs[1].to(mask_input.device)
                coords = torch.nonzero(masks)
                y, x = coords[:, 0], coords[:, 1]
                box_coords = ResizeLongestSide.apply_coords(
                    torch.tensor([[[x.min(), y.min()], [x.max(), y.max()]]], dtype=torch.float32, device=self.device),
                    ori_shape,
                    self.image_size,
                )
                point_coords = torch.cat((point_coords, box_coords), dim=1)
                point_labels = torch.cat((point_labels, self.point_labels_box.to(self.device)), dim=1)

            scores, logits = self(
                mode="learn",
                image_embeddings=image_embedding,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                has_mask_input=has_mask_input,
            )

        _, masks = self._postprocess_masks(logits, scores, ori_shape)
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
        bboxes: Tensor | None = None,
        points: Tensor | None = None,
        annotations: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Dict[ScoredLabel, List[Dict[str, Tensor]]]:
        """Preprocess prompts.

        Currently, preprocessing for bounding boxes is only supported.

        Args:
            bboxes (Tensor, optional): Bounding box prompts to be preprocessed.
            points (Tensor, optional): Point prompts to be preprocessed, to be supported.
            annotations (Tensor, optional): annotation prompts to be preprocessed, to be supported.
            labels (Tensor, optional): Assigned labels according to given prompts.
                Currently, it is only matched to bboxes, and it will be deprecated.

        Returns:
            (defaultdict[ScoredLabel, List[Dict[str, Tensor]]]): Processed and arranged each single prompt
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
            return None, None

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

    def _postprocess_masks(
        self,
        logits: Tensor,
        scores: Tensor,
        ori_shape: Tensor,
        is_single: bool = False,
    ):
        """Post-process masks for cascaded post-refinements."""
        high_res_masks = self.postprocess_masks(logits, self.image_size, ori_shape)
        masks = high_res_masks > self.mask_threshold

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

    def _update_value(self, target: Dict[str, Any], key: str, value: Tensor) -> None:
        """Update tensor to target dictionary.

        Args:
            target (Dict[str, Any]): Target dictionary to be updated.
            key (str): Key to be used for update.
            value (Tensor): Value to be used for update.
        """
        if key in target:
            target[key] = torch.cat((target[key], value))
        else:
            target[key] = value

    def _merge_prompts(
        self,
        label: LongTensor,
        input_prompt: tv_tensors.TVTensor,
        processed_prompts: Dict[LongTensor, List[Dict[str, Tensor]]],
        use_only_background: bool = True,
    ) -> Dict[str, Tensor]:
        """Merge target prompt and other prompts.

        Merge a foreground prompt and other prompts (background or prompts with other classes).

        Args:
            label (LongTensor): Label information.
            input_prompt (tv_tensors.TVTensor): A foreground prompt to be merged with other prompts.
            processed_prompts (Dict[LongTensor, List[Dict[str, Tensor]]]): The whole class-wise prompts
                processed at _preprocess_prompts.
            use_only_background (bool): Whether merging only background prompt, defaults to True.
                It is applied to only point_coords.

        Returns:
            (Dict[str, Tensor]): Merged prompts.
        """
        merged_input_prompt = deepcopy(input_prompt)
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
                        self._update_value(merged_input_prompt, "point_coords", other_input_prompt.get("point_coords"))
                        self._update_value(
                            merged_input_prompt,
                            "point_labels",
                            torch.zeros_like(other_input_prompt.get("point_labels")),
                        )
        return merged_input_prompt


class OTXZeroShotSegmentAnything(OTXSegmentAnything):
    """Zero-Shot Visual Prompting model."""
    
    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""
        return ZeroShotSegmentAnything(**self.config)

    def forward(self, inputs: ZeroShotVisualPromptingBatchDataEntity) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        if self.training:
            forward_fn = self.model.learn
        else:
            forward_fn = self.model.infer

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
        images = torch.stack(inputs.images, dim=0).to(dtype=torch.float32)
        return {"images": images}
            

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ZeroShotVisualPromptingBatchDataEntity,
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for you model."""
        if self.training:
            return outputs

        masks: list[tv_tensors.Mask] = []
        scores: list[Tensor] = []
        labels: list[torch.LongTensor] = inputs.labels
        for mask, score in zip(*outputs):
            masks.append(tv_tensors.Mask(mask, dtype=torch.float32))
            scores.append(score)

        return ZeroShotVisualPromptingBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=[],
            masks=masks,
            polygons=[],
            labels=labels,
        )
        
    def _gather_prompts_with_labels(
        self,
        prompts: list[list[tv_tensors.TVTensor]],
        labels: list[LongTensor]
    ) -> list[dict[LongTensor, list[tv_tensors.TVTensor]]]:
        """Gather prompts according to labels."""
        total_processed_prompts = []
        for prompt, label in zip(prompts, labels):
            processed_prompts = defaultdict(list)
            for p, l in zip(prompt, label):
                processed_prompts[l].append(p)
            processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x))
            total_processed_prompts.append(processed_prompts)
        return total_processed_prompts
            
        
