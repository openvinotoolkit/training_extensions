# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything model for the OTX zero-shot visual prompting."""

from __future__ import annotations

import logging as log
import pickle  # nosec  B403   used pickle for dumping object
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torchvision.transforms.v2 as tvt_v2
from datumaro import Polygon as dmPolygon
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision import tv_tensors
from torchvision.tv_tensors import BoundingBoxes, Image, Mask

from otx.algo.visual_prompting.segment_anything import DEFAULT_CONFIG_SEGMENT_ANYTHING, SegmentAnything
from otx.core.data.entity.base import OTXBatchLossEntity, Points
from otx.core.data.entity.visual_prompting import (
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.metrics.visual_prompting import VisualPromptingMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.visual_prompting import OTXZeroShotVisualPromptingModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes, NullLabelInfo
from otx.core.utils.mask_util import polygon_to_bitmap

if TYPE_CHECKING:
    import numpy as np
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


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

        super().__init__(*args, **kwargs)

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

    def expand_reference_info(self, reference_feats: Tensor, new_largest_label: int) -> Tensor:
        """Expand reference info dimensions if newly given processed prompts have more lables."""
        if new_largest_label > (cur_largest_label := len(reference_feats) - 1):
            diff = new_largest_label - cur_largest_label
            reference_feats = F.pad(reference_feats, (0, 0, 0, 0, 0, diff), value=0.0)
        return reference_feats

    @torch.no_grad()
    def learn(
        self,
        images: list[Image],
        processed_prompts: list[dict[int, list[BoundingBoxes | Points | dmPolygon | Mask]]],
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
                    if isinstance(input_prompt, Mask):
                        # directly use annotation information as a mask
                        ref_mask[input_prompt] += 1
                    elif isinstance(input_prompt, dmPolygon):
                        ref_mask[torch.as_tensor(polygon_to_bitmap([input_prompt], *ori_shape)[0])] += 1
                    else:
                        if isinstance(input_prompt, BoundingBoxes):
                            point_coords = input_prompt.reshape(-1, 2, 2)
                            point_labels = torch.tensor([[2, 3]], device=point_coords.device)
                        elif isinstance(input_prompt, Points):
                            point_coords = input_prompt.reshape(-1, 1, 2)
                            point_labels = torch.tensor([[1]], device=point_coords.device)
                        else:
                            log.info(f"Current input prompt ({input_prompt.__class__.__name__}) is not supported.")
                            continue

                        masks = self._predict_masks(
                            mode="learn",
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
        images: list[Image],
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

        # First-step prediction
        mask_input = torch.zeros(
            1,
            1,
            *(x * 4 for x in image_embeddings.shape[2:]),
            device=image_embeddings.device,
        )
        has_mask_input = self.has_mask_inputs[0].to(mask_input.device)
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


class OTXZeroShotSegmentAnything(OTXZeroShotVisualPromptingModel):
    """Zero-Shot Visual Prompting model."""

    def __init__(  # noqa: PLR0913
        self,
        backbone: Literal["tiny_vit", "vit_b"],
        label_info: LabelInfoTypes = NullLabelInfo(),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = VisualPromptingMetricCallable,
        torch_compile: bool = False,
        reference_info_dir: Path | str = "reference_infos",
        infer_reference_info_root: Path | str = "../.latest/train",
        save_outputs: bool = True,
        pixel_mean: list[float] | None = [123.675, 116.28, 103.53],  # noqa: B006
        pixel_std: list[float] | None = [58.395, 57.12, 57.375],  # noqa: B006
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = True,
        default_threshold_reference: float = 0.3,
        default_threshold_target: float = 0.65,
        use_stability_score: bool = False,
        return_single_mask: bool = False,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        self.config = {
            "backbone": backbone,
            "freeze_image_encoder": freeze_image_encoder,
            "freeze_prompt_encoder": freeze_prompt_encoder,
            "freeze_mask_decoder": freeze_mask_decoder,
            "default_threshold_reference": default_threshold_reference,
            "default_threshold_target": default_threshold_target,
            "use_stability_score": use_stability_score,
            "return_single_mask": return_single_mask,
            "return_extra_metrics": return_extra_metrics,
            "stability_score_offset": stability_score_offset,
            **DEFAULT_CONFIG_SEGMENT_ANYTHING[backbone],
        }
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

        self.save_outputs = save_outputs
        self.reference_info_dir: Path = Path(reference_info_dir)
        self.infer_reference_info_root: Path = Path(infer_reference_info_root)

        self.register_buffer("pixel_mean", Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", Tensor(pixel_std).view(-1, 1, 1), False)

        self.initialize_reference_info()

    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""
        return ZeroShotSegmentAnything(**self.config)

    def forward(  # type: ignore[override]
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        forward_fn = self.learn if self.training else self.infer
        return forward_fn(inputs)  # type: ignore[operator]

    def learn(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reference_feats: Tensor | None = None,
        used_indices: Tensor | None = None,
        reset_feat: bool = False,
        is_cascade: bool = False,
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Learn to directly connect to the model."""
        self.training = True
        if reset_feat:
            self.initialize_reference_info()

        outputs = self.model.learn(
            **self._customize_inputs(inputs, reference_feats=reference_feats, used_indices=used_indices),
            is_cascade=is_cascade,
        )
        return self._customize_outputs(outputs, inputs)

    def infer(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reference_feats: Tensor | None = None,
        used_indices: Tensor | None = None,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        is_cascade: bool = True,
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Infer to directly connect to the model."""
        self.training = False
        outputs = self.model.infer(
            **self._customize_inputs(inputs, reference_feats=reference_feats, used_indices=used_indices),
            threshold=threshold,
            num_bg_points=num_bg_points,
            is_cascade=is_cascade,
        )
        return self._customize_outputs(outputs, inputs)

    def _customize_inputs(  # type: ignore[override]
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reference_feats: Tensor | None = None,
        used_indices: Tensor | None = None,
    ) -> dict[str, Any]:  # type: ignore[override]
        """Customize the inputs for the model."""
        inputs = self.transforms(inputs)
        forward_inputs = {
            "images": [tv_tensors.wrap(image.unsqueeze(0), like=image) for image in inputs.images],
            "reference_feats": reference_feats if reference_feats is not None else self.reference_feats,
            "used_indices": used_indices if used_indices is not None else self.used_indices,
            "ori_shapes": [torch.tensor(info.ori_shape) for info in inputs.imgs_info],
        }
        if self.training:
            # learn
            forward_inputs.update({"processed_prompts": self._gather_prompts_with_labels(inputs)})

        return forward_inputs

    def _customize_outputs(  # type: ignore[override]
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for you model."""
        if self.training:
            self.reference_feats = outputs[0].get("reference_feats")
            self.used_indices = outputs[0].get("used_indices")
            return outputs

        masks: list[Mask] = []
        prompts: list[Points] = []
        scores: list[Tensor] = []
        labels: list[Tensor] = []
        for idx, (predicted_masks, used_points) in enumerate(outputs):
            _masks: list[Tensor] = []
            _prompts: list[Tensor] = []
            _scores: list[Tensor] = []
            _labels: list[Tensor] = []
            for label, predicted_mask in predicted_masks.items():
                if len(predicted_mask) == 0:
                    continue
                _masks.append(torch.stack(predicted_mask, dim=0))
                _used_points_scores = torch.stack(used_points[label], dim=0)
                _prompts.append(_used_points_scores[:, :2])
                _scores.append(_used_points_scores[:, 2])
                _labels.append(torch.tensor([label] * len(_used_points_scores), dtype=torch.int64, device=self.device))

            if len(_masks) == 0:
                masks.append(
                    tv_tensors.Mask(
                        torch.zeros((1, *inputs.imgs_info[idx].ori_shape), dtype=torch.float32, device=self.device),
                    ),
                )
                prompts.append(
                    Points([], canvas_size=inputs.imgs_info[idx].ori_shape, dtype=torch.float32, device=self.device),
                )
                scores.append(torch.tensor([-1.0], dtype=torch.float32, device=self.device))
                labels.append(torch.tensor([-1], dtype=torch.int64, device=self.device))
                continue

            masks.append(tv_tensors.Mask(torch.cat(_masks, dim=0)))
            prompts.append(Points(torch.cat(_prompts, dim=0), canvas_size=inputs.imgs_info[idx].ori_shape))
            scores.append(torch.cat(_scores, dim=0))
            labels.append(torch.cat(_labels, dim=0))

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
        inputs: ZeroShotVisualPromptingBatchDataEntity,
    ) -> list[dict[int, list[BoundingBoxes | Points | dmPolygon | Mask]]]:
        """Gather prompts according to labels."""
        total_processed_prompts: list[dict[int, list[BoundingBoxes | Points | dmPolygon | Mask]]] = []
        for batch, batch_labels in enumerate(inputs.labels):
            processed_prompts = defaultdict(list)
            for prompt_type in ["prompts", "polygons", "masks"]:
                _prompts = getattr(inputs, prompt_type, None)
                prompt_labels = getattr(batch_labels, prompt_type, None)
                if _prompts is None or prompt_labels is None:
                    continue

                for idx, _label in enumerate(prompt_labels):
                    if prompt_type in ("prompts", "polygons"):
                        processed_prompts[int(_label)].append(_prompts[batch][idx])
                    else:
                        # for mask
                        processed_prompts[int(_label)].append(Mask(_prompts[batch][idx]))

            sorted_processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x))
            total_processed_prompts.append(sorted_processed_prompts)

        return total_processed_prompts

    def apply_image(self, image: Image | np.ndarray, target_length: int = 1024) -> Image:
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
            canvas_size=(target_length, target_length),
        )

    def apply_prompts(
        self,
        prompts: list[Points | BoundingBoxes],
        ori_shape: tuple[int, ...],
        target_length: int = 1024,
    ) -> list[Points | BoundingBoxes]:
        """Preprocess prompts to be used in the model."""
        transformed_prompts: list[Points | BoundingBoxes] = []
        for prompt in prompts:
            if isinstance(prompt, Points):
                transformed_prompts.append(self.apply_points(prompt, ori_shape, target_length))
            elif isinstance(prompt, BoundingBoxes):
                transformed_prompts.append(self.apply_boxes(prompt, ori_shape, target_length))
            else:
                log.info(f"Current prompt ({prompt.__class__.__name__}) is not supported, saved as it is.")
                transformed_prompts.append(prompt)
        return transformed_prompts

    def get_preprocess_shape(self, oldh: int, oldw: int, target_length: int) -> tuple[int, int]:
        """Get preprocess shape."""
        scale = target_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def preprocess(self, x: Image) -> Image:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        x = self.model.pad_to_square(x)
        return Image(x)

    def transforms(self, entity: ZeroShotVisualPromptingBatchDataEntity) -> ZeroShotVisualPromptingBatchDataEntity:
        """Transforms for ZeroShotVisualPromptingBatchDataEntity."""
        return entity.wrap(
            images=[self.preprocess(self.apply_image(image)) for image in entity.images],
            prompts=[
                self.apply_prompts(prompt, info.ori_shape, self.model.image_size)
                for prompt, info in zip(entity.prompts, entity.imgs_info)
            ],
            masks=entity.masks,
            polygons=entity.polygons,
            labels=entity.labels,
        )

    def initialize_reference_info(self) -> None:
        """Initialize reference information."""
        self.register_buffer("reference_feats", torch.zeros(0, 1, self.model.embed_dim), False)
        self.register_buffer("used_indices", torch.tensor([], dtype=torch.int64), False)

    def save_reference_info(self, default_root_dir: Path | str) -> None:
        """Save reference info."""
        reference_info = {
            "reference_feats": self.reference_feats,
            "used_indices": self.used_indices,
        }
        # save reference info
        self.saved_reference_info_path: Path = Path(default_root_dir) / self.reference_info_dir / "reference_info.pt"
        self.saved_reference_info_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO (sungchul): ticket no. 139210
        torch.save(reference_info, self.saved_reference_info_path)
        pickle.dump(
            {k: v.numpy() for k, v in reference_info.items()},
            self.saved_reference_info_path.with_suffix(".pickle").open("wb"),
        )
        log.info(f"Saved reference info at {self.saved_reference_info_path}.")

    def load_reference_info(
        self,
        default_root_dir: Path | str,
        device: str | torch.device = "cpu",
        path_to_directly_load: Path | None = None,
    ) -> bool:
        """Load latest reference info to be used.

        Args:
            default_root_dir (Path | str): Default root directory to be used
                when inappropriate infer_reference_info_root is given.
            device (str | torch.device): Device that reference infos will be attached.
            path_to_directly_load (Path | None): Reference info path to directly be loaded.
                Normally, it is obtained after `learn` which is executed when trying to do `infer`
                without reference features in `on_test_start` or `on_predict_start`.

        Returns:
            (bool): Whether normally loading checkpoint or not.
        """
        if path_to_directly_load is not None:
            # if `path_to_directly_load` is given, forcely load
            reference_info = torch.load(path_to_directly_load)
            retval = True
            log.info(f"reference info saved at {path_to_directly_load} was successfully loaded.")

        else:
            if str(self.infer_reference_info_root) == "../.latest/train":
                # for default setting
                path_reference_info = (
                    Path(default_root_dir)
                    / self.infer_reference_info_root
                    / self.reference_info_dir
                    / "reference_info.pt"
                )
            else:
                # for user input
                path_reference_info = self.infer_reference_info_root / self.reference_info_dir / "reference_info.pt"

            if path_reference_info.is_file():
                reference_info = torch.load(path_reference_info)
                retval = True
                log.info(f"reference info saved at {path_reference_info} was successfully loaded.")
            else:
                reference_info = {}
                retval = False

        self.register_buffer(
            "reference_feats",
            reference_info.get("reference_feats", torch.zeros(0, 1, self.model.embed_dim)).to(device),
            False,
        )
        self.register_buffer(
            "used_indices",
            reference_info.get("used_indices", torch.tensor([], dtype=torch.int64)).to(device),
            False,
        )
        return retval
