# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
"""Segment Anything model for the OTX visual prompting."""

from __future__ import annotations

import logging as log
from collections import defaultdict
from copy import deepcopy
from itertools import product
from typing import Any

import torch
from datumaro import Polygon as dmPolygon
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision import tv_tensors

from otx.algo.visual_prompting.utils.postprocess import get_prepadded_size, postprocess_masks
from otx.core.data.entity.base import Points
from otx.core.utils.mask_util import polygon_to_bitmap


class SegmentAnything(nn.Module):
    """Visual prompting model class for Segment Anything."""

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        criterion: nn.Module,
        image_size: int = 1024,
        mask_threshold: float = 0.0,
        use_stability_score: bool = False,
        return_single_mask: bool = False,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.use_stability_score = use_stability_score
        self.return_single_mask = return_single_mask
        self.return_extra_metrics = return_extra_metrics
        self.stability_score_offset = stability_score_offset

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.criterion = criterion

    def forward(
        self,
        images: tv_tensors.Image,
        ori_shapes: list[Tensor],
        bboxes: list[tv_tensors.BoundingBoxes | None],
        points: list[tuple[Points, Tensor] | None],
        gt_masks: list[tv_tensors.Mask] | None = None,
    ) -> Tensor | tuple[list[Tensor], list[Tensor]]:
        """Forward method for SAM training/validation/prediction.

        Args:
            images (tv_tensors.Image): Images with shape (B, C, H, W).
            ori_shapes (List[Tensor]): List of original shapes per image.
            bboxes (List[tv_tensors.BoundingBoxes], optional): A Nx4 array given a box prompt to the model,
                in XYXY format.
            points (List[Tuple[Points, Tensor]], optional): Point coordinates and labels to embed.
                Point coordinates are BxNx2 arrays of point prompts to the model.
                Each point is in (X,Y) in pixels. Labels are BxN arrays of labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            gt_masks (List[tv_tensors.Mask], optional): Ground truth masks for loss calculation.

        Returns:
            (Tensor): Calculated loss values.
            (Tuple[List[Tensor], List[Tensor]]): Tuple of list with predicted masks with shape (B, 1, H, W)
                and List with IoU predictions with shape (N, 1).
        """
        image_embeddings = self.image_encoder(images)
        pred_masks = []
        ious = []
        for idx, embedding in enumerate(image_embeddings):
            low_res_masks, iou_predictions = [], []
            for prompt in [bboxes[idx], points[idx]]:
                if prompt is None:
                    continue

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=prompt if isinstance(prompt[0], Points) else None,
                    boxes=prompt if isinstance(prompt, tv_tensors.BoundingBoxes) else None,
                    masks=None,
                )
                _low_res_masks, _iou_predictions = self.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,  # when given multiple prompts. if there is single prompt True would be better. # noqa: E501
                )
                low_res_masks.append(_low_res_masks)
                iou_predictions.append(_iou_predictions)

            pred_masks.append(torch.cat(low_res_masks, dim=0))
            ious.append(torch.cat(iou_predictions, dim=0))

        if self.training:
            return self.criterion(pred_masks, gt_masks, ious, ori_shapes)

        post_processed_pred_masks: list[Tensor] = []
        for pred_mask, ori_shape in zip(pred_masks, ori_shapes):
            post_processed_pred_mask = postprocess_masks(pred_mask, self.image_size, ori_shape)
            post_processed_pred_masks.append(post_processed_pred_mask.squeeze(1).sigmoid())
        return post_processed_pred_masks, ious

    @torch.no_grad()
    def forward_for_tracing(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        mask_input: Tensor,
        has_mask_input: Tensor,
        ori_shape: Tensor,
    ) -> tuple[Tensor, ...]:
        """Forward method for SAM inference (export/deploy).

        Args:
            image_embeddings (Tensor): The image embedding with a batch index of length 1.
                If it is a zero tensor, the image embedding will be computed from the image.
            point_coords (Tensor): Coordinates of sparse input prompts,
                corresponding to both point inputs and box inputs.
                Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner.
                Coordinates must already be transformed to long-side 1024. Has a batch index of length 1.
            point_labels (Tensor): Labels for the sparse input prompts.
                0 is a negative input point, 1 is a positive input point,
                2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point.
                If there is no box input, a single padding point with label -1 and
                coordinates (0.0, 0.0) should be concatenated.
            mask_input (Tensor): A mask input to the model with shape 1x1x256x256.
                This must be supplied even if there is no mask input. In this case, it can just be zeros.
            has_mask_input (Tensor): An indicator for the mask input.
                1 indicates a mask input, 0 indicates no mask input.
                This input has 1x1 shape due to supporting openvino input layout.
            ori_shape (Tensor): The size of the input image in (H,W) format, before any transformation.
                This input has 1x2 shape due to supporting openvino input layout.
        """
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = self.calculate_stability_score(
                masks,
                self.mask_threshold,
                self.stability_score_offset,
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        upscaled_masks = postprocess_masks(masks, self.image_size, ori_shape)

        if self.return_extra_metrics:
            stability_scores = self.calculate_stability_score(
                upscaled_masks,
                self.mask_threshold,
                self.stability_score_offset,
            )
            areas = (upscaled_masks > self.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, masks

    def _embed_points(self, point_coords: Tensor, point_labels: Tensor) -> Tensor:
        """Embed sparse input prompts.

        Args:
            point_coords (Tensor): Coordinates of sparse input prompts,
                corresponding to both point inputs and box inputs. Boxes are encoded using two points,
                one for the top-left corner and one for the bottom-right corner.
                Coordinates must already be transformed to long-side 1024. Has a batch index of length 1.
            point_labels (Tensor): Labels for the sparse input prompts.
                0 is a negative input point, 1 is a positive input point,
                2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point.
                If there is no box input, a single padding point with label -1 and
                coordinates (0.0, 0.0) should be concatenated.

        Returns:
            point_embedding (Tensor): The embedded sparse input prompts.
        """
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.image_size
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)  # noqa: SLF001
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: Tensor, has_mask_input: Tensor) -> Tensor:
        """Embed the mask input.

        Args:
            input_mask (Tensor): A mask input to the model with shape 1x1x256x256.
                This must be supplied even if there is no mask input. In this case, it can just be zeros.
            has_mask_input (Tensor): An indicator for the mask input.
                1 indicates a mask input, 0 indicates no mask input.

        Returns:
            mask_embedding (Tensor): The embedded mask input.
        """
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(input_mask)
        return mask_embedding + (1 - has_mask_input) * self.prompt_encoder.no_mask_embed.weight.reshape(
            1,
            -1,
            1,
            1,
        )

    def calculate_stability_score(self, masks: Tensor, mask_threshold: float, threshold_offset: float = 1.0) -> Tensor:
        """Computes the stability score for a batch of masks.

        The stability score is the IoU between the binary masks obtained
        by thresholding the predicted mask logits at high and low values.

        Args:
            masks (Tensor): A batch of predicted masks with shape BxHxW.
            mask_threshold (float): The threshold used to binarize the masks.
            threshold_offset (float, optional): The offset used to compute the stability score.

        Returns:
            stability_scores (Tensor): The stability scores for the batch of masks.
        """
        # One mask is always contained inside the other.
        # Save memory by preventing unnecessary cast to torch.int64
        intersections = (
            (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        )
        unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
        return intersections / unions

    def select_masks(self, masks: Tensor, iou_preds: Tensor, num_points: int) -> tuple[Tensor, Tensor]:
        """Selects the best mask from a batch of masks.

        Args:
            masks (Tensor): A batch of predicted masks with shape BxMxHxW.
            iou_preds (Tensor): A batch of predicted IoU scores with shape BxM.
            num_points (int): The number of points in the input.

        Returns:
            masks (Tensor): The selected masks with shape Bx1xHxW.
            iou_preds (Tensor): The selected IoU scores with shape Bx1.
        """
        # Determine if we should return the multi-click mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor([[1000] + [0] * (self.mask_decoder.num_mask_tokens - 1)]).to(
            iou_preds.device,
        )
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds


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
        threshold: float = 0.0,
        num_bg_points: int = 1,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        """Get prompt candidates."""
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
        threshold: float = 0.0,
        num_bg_points: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Get prompt candidates from given reference and target features."""
        target_feat = image_embeddings.squeeze()  # (256, 64, 64)
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        sim = reference_feat @ target_feat
        sim = sim.reshape(1, 1, h_feat, w_feat)
        sim = postprocess_masks(sim, self.image_size, ori_shape)

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
        threshold: float = 0.0,
        num_bg_points: int = 1,
    ) -> tuple[Tensor, Tensor]:
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
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        criterion: nn.Module | None = None,
        image_size: int = 1024,
        mask_threshold: float = 0.0,
        default_threshold_reference: float = 0.3,
        default_threshold_target: float = 0.65,
        use_stability_score: bool = False,
        return_single_mask: bool = False,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        super().__init__(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            criterion=criterion,
            image_size=image_size,
            mask_threshold=mask_threshold,
            use_stability_score=use_stability_score,
            return_single_mask=return_single_mask,
            return_extra_metrics=return_extra_metrics,
            stability_score_offset=stability_score_offset,
        )

        # set PromptGetter
        self.prompt_getter = PromptGetter(image_size=self.image_size)
        self.prompt_getter.set_default_thresholds(
            default_threshold_reference=default_threshold_reference,
            default_threshold_target=default_threshold_target,
        )

        # set default constants
        self.point_labels_box = torch.tensor([[2, 3]], dtype=torch.float32)
        self.has_mask_inputs = [torch.tensor([[0.0]]), torch.tensor([[1.0]])]

    def expand_reference_info(self, reference_feats: Tensor, new_largest_label: int) -> Tensor:
        """Expand reference info dimensions if newly given processed prompts have more labels."""
        if new_largest_label > (cur_largest_label := len(reference_feats) - 1):
            diff = new_largest_label - cur_largest_label
            reference_feats = F.pad(reference_feats, (0, 0, 0, 0, 0, diff), value=0.0)
        return reference_feats

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Forward method for Zero-shot SAM."""
        if self.training:
            return self.learn(*args, **kwargs)
        return self.infer(*args, **kwargs)

    @torch.no_grad()
    def learn(
        self,
        images: list[tv_tensors.Image],
        processed_prompts: list[dict[int, list[tv_tensors.BoundingBoxes | Points | dmPolygon | tv_tensors.Mask]]],
        reference_feats: Tensor,
        used_indices: Tensor,
        ori_shapes: list[Tensor],
        is_cascade: bool = False,
    ) -> tuple[dict[str, Tensor], list[Tensor]] | None:
        """Get reference features.

        Using given images, get reference features.
        These reference features will be used for `infer` to get target results.
        Currently, single batch is only supported.

        Args:
            images (list[Image]): List of given images for reference features.
            processed_prompts (dict[int, list[BoundingBoxes | Points | dmPolygon | Mask]]): The class-wise prompts
                processed at OTXZeroShotSegmentAnything._gather_prompts_with_labels.
            reference_feats (Tensor): Reference features for target prediction.
            used_indices (Tensor): To check which indices of reference features are validate.
            ori_shapes (List[Tensor]): List of original shapes per image.
            is_cascade (bool): Whether use cascade inference. Defaults to False.
        """
        # initialize tensors to contain reference features and prompts
        largest_label = max([max(prompt.keys()) for prompt in processed_prompts])
        reference_feats = self.expand_reference_info(reference_feats, largest_label)
        new_used_indices: list[Tensor] = []
        # TODO (sungchul): consider how to handle multiple reference features, currently replace it

        reference_masks: list[Tensor] = []
        for image, prompts, ori_shape in zip(images, processed_prompts, ori_shapes):
            image_embeddings = self.image_encoder(image)
            processed_embedding = image_embeddings.squeeze().permute(1, 2, 0)

            ref_masks = torch.zeros(largest_label + 1, *map(int, ori_shape))
            for label, input_prompts in prompts.items():
                # TODO (sungchul): how to skip background class
                # TODO (sungchul): ensemble multi reference features (current : use merged masks)
                ref_mask = torch.zeros(*map(int, ori_shape), dtype=torch.uint8, device=image.device)
                for input_prompt in input_prompts:
                    if isinstance(input_prompt, tv_tensors.Mask):
                        # directly use annotation information as a mask
                        ref_mask[input_prompt] += 1
                    elif isinstance(input_prompt, dmPolygon):
                        ref_mask[torch.as_tensor(polygon_to_bitmap([input_prompt], *ori_shape)[0])] += 1
                    else:
                        if isinstance(input_prompt, tv_tensors.BoundingBoxes):
                            point_coords = input_prompt.reshape(-1, 2, 2)
                            point_labels = torch.tensor([[2, 3]], device=point_coords.device)
                        elif isinstance(input_prompt, Points):
                            point_coords = input_prompt.reshape(-1, 1, 2)
                            point_labels = torch.tensor([[1]], device=point_coords.device)
                        else:
                            log.info(f"Current input prompt ({input_prompt.__class__.__name__}) is not supported.")
                            continue

                        masks = self._predict_masks(
                            image_embeddings=image_embeddings,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            ori_shape=ori_shape,
                            is_cascade=is_cascade,
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

                reference_feats[label] = ref_feat.detach().cpu()
                new_used_indices.append(torch.tensor([label]))
                ref_masks[label] = ref_mask.detach().cpu()
            reference_masks.append(ref_masks)
        used_indices = torch.cat((used_indices, *new_used_indices), dim=0).unique()
        return {"reference_feats": reference_feats, "used_indices": used_indices}, reference_masks

    @torch.no_grad()
    def infer(
        self,
        images: list[tv_tensors.Image],
        reference_feats: Tensor,
        used_indices: Tensor,
        ori_shapes: list[Tensor],
        threshold: float = 0.0,
        num_bg_points: int = 1,
        is_cascade: bool = True,
    ) -> list[list[defaultdict[int, list[Tensor]]]]:
        """Zero-shot inference with reference features.

        Get target results by using reference features and target images' features.

        Args:
            images (list[Image]): Given images for target results.
            reference_feats (Tensor): Reference features for target prediction.
            used_indices (Tensor): To check which indices of reference features are validate.
            ori_shapes (list[Tensor]): Original image size.
            threshold (float): Threshold to control masked region. Defaults to 0.0.
            num_bg_points (1): Number of background points. Defaults to 1.
            is_cascade (bool): Whether use cascade inference. Defaults to True.

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
                threshold=threshold,
                num_bg_points=num_bg_points,
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
        def _calculate_mask_iou(mask1: Tensor, mask2: Tensor) -> tuple[float, Tensor | None]:
            if (union := torch.logical_or(mask1, mask2).sum().item()) == 0:
                # Avoid division by zero
                return 0.0, None
            intersection = torch.logical_and(mask1, mask2)
            return intersection.sum().item() / union, intersection

        for (label, masks), (other_label, other_masks) in product(predicted_masks.items(), predicted_masks.items()):
            if other_label <= label:
                continue

            overlapped_label = []
            overlapped_other_label = []
            for (im, mask), (jm, other_mask) in product(enumerate(masks), enumerate(other_masks)):
                _mask_iou, _intersection = _calculate_mask_iou(mask, other_mask)
                if _mask_iou > threshold_iou:
                    # compare overlapped regions between different labels and filter out the lower score
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        overlapped_other_label.append(jm)
                    else:
                        overlapped_label.append(im)
                elif _mask_iou > 0:
                    # refine the slightly overlapping region
                    overlapped_coords = torch.where(_intersection)
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        other_mask[overlapped_coords] = 0.0
                    else:
                        mask[overlapped_coords] = 0.0

            for im in sorted(list(set(overlapped_label)), reverse=True):  # noqa: C414
                masks.pop(im)
                used_points[label].pop(im)

            for jm in sorted(list(set(overlapped_other_label)), reverse=True):  # noqa: C414
                other_masks.pop(jm)
                used_points[other_label].pop(jm)

    def _predict_masks(
        self,
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

        # First-step prediction
        mask_input = torch.zeros(
            1,
            1,
            *(x * 4 for x in image_embeddings.shape[2:]),
            device=image_embeddings.device,
        )
        has_mask_input = self.has_mask_inputs[0].to(mask_input.device)
        high_res_masks, scores, logits = self.forward_for_tracing(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_input,
            ori_shape=ori_shape,
        )
        masks = high_res_masks > self.mask_threshold

        if is_cascade:
            for i in range(2):
                if i == 0:
                    # Cascaded Post-refinement-1
                    mask_input, best_masks = self._decide_cascade_results(
                        masks,
                        logits,
                        scores,
                        is_single=True,
                    )
                    if best_masks.sum() == 0:
                        return best_masks

                    has_mask_input = self.has_mask_inputs[1].to(mask_input.device)

                else:
                    # Cascaded Post-refinement-2
                    mask_input, best_masks = self._decide_cascade_results(masks, logits, scores)
                    if best_masks.sum() == 0:
                        return best_masks

                    has_mask_input = self.has_mask_inputs[1].to(mask_input.device)
                    coords = torch.nonzero(best_masks)
                    y, x = coords[:, 0], coords[:, 1]
                    box_coords = self._preprocess_coords(
                        torch.tensor(
                            [[[x.min(), y.min()], [x.max(), y.max()]]],
                            dtype=torch.float32,
                            device=coords.device,
                        ),
                        ori_shape,
                        self.image_size,
                    )
                    point_coords = torch.cat((point_coords, box_coords), dim=1)
                    point_labels = torch.cat((point_labels, self.point_labels_box.to(point_labels.device)), dim=1)

                high_res_masks, scores, logits = self.forward_for_tracing(
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
        ori_shape: Tensor,
        target_length: int,
    ) -> Tensor:
        """Expects a torch tensor of length 2 in the final dimension.

        Requires the original image size in (H, W) format.

        Args:
            coords (Tensor): Coordinates tensor.
            ori_shape (Tensor): Original size of image.
            target_length (int): The length of the longest side of the image.

        Returns:
            (Tensor): Resized coordinates.
        """
        old_h, old_w = ori_shape
        new_h, new_w = get_prepadded_size(ori_shape, target_length)
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
        masks = self.pad_to_square(masks)
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=feats.shape[0:2], mode="bilinear").squeeze()

        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0).unsqueeze(0)
        return masked_feat / masked_feat.norm(dim=-1, keepdim=True)

    def pad_to_square(self, x: Tensor) -> Tensor:
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
        return logits[:, [best_idx]], masks[0, best_idx]
