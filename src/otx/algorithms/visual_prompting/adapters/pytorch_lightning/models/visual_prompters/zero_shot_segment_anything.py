"""SAM module for visual prompting zero-shot learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import ResizeLongestSide
from otx.api.entities.scored_label import ScoredLabel
from otx.utils.logger import get_logger

from .segment_anything import SegmentAnything

logger = get_logger()


class PromptGetter(nn.Module):
    """Prompt getter for zero-shot learning."""

    default_threshold_reference = 0.3
    default_threshold_target = 0.65

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.initialize()

    def initialize(self) -> None:
        """Initialize reference features and prompts."""
        self.reference_feats: Dict[int, torch.Tensor] = {}
        self.reference_prompts: Dict[int, torch.Tensor] = {}

    def set_default_thresholds(self, default_threshold_reference: float, default_threshold_target: float) -> None:
        """Set default thresholds."""
        self.default_threshold_reference = default_threshold_reference
        self.default_threshold_target = default_threshold_target

    def set_reference(self, label: ScoredLabel, reference_feats: torch.Tensor, reference_prompts: torch.Tensor) -> None:
        """Set reference features and prompts."""
        self.reference_feats[int(label.id_)] = reference_feats
        self.reference_prompts[int(label.id_)] = reference_prompts

    def forward(
        self,
        image_embeddings: torch.Tensor,
        padding: Tuple[int, ...],
        original_size: Tuple[int, int],
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Get prompt candidates."""
        target_feat = image_embeddings.squeeze()
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = self._preprocess_target_feat(target_feat, c_feat, h_feat, w_feat)

        prompts = {}
        for label, reference_feat in self.reference_feats.items():
            sim = reference_feat.to(target_feat.device) @ target_feat
            sim = sim.reshape(1, 1, h_feat, w_feat)
            sim = ZeroShotSegmentAnything.postprocess_masks(
                sim, (self.image_size, self.image_size), padding, original_size
            ).squeeze()

            # threshold = 0.85 * sim.max() if num_classes > 1 else self.default_threshold_target
            threshold = self.default_threshold_target
            points_scores, bg_coords = self._point_selection(sim, original_size, threshold)
            if points_scores is None:
                # skip if there is no point with score > threshold
                continue
            prompts[label] = (points_scores, bg_coords)
        return prompts

    def _preprocess_target_feat(self, target_feat: torch.Tensor, c_feat: int, h_feat: int, w_feat: int) -> torch.Tensor:
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)
        return target_feat

    def _point_selection(
        self,
        mask_sim: torch.Tensor,
        original_size: Tuple[int, int],
        threshold: float,
        num_bg_points: int = 1,
        downsizing: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select point used as point prompts."""
        _, w_sim = mask_sim.shape

        # Top-last point selection
        bg_indices = mask_sim.flatten().topk(num_bg_points, largest=False)[1]
        bg_x = (bg_indices // w_sim).unsqueeze(0)
        bg_y = bg_indices - bg_x * w_sim
        bg_coords = torch.cat((bg_y, bg_x), dim=0).permute(1, 0)
        bg_coords = bg_coords

        point_coords = torch.where(mask_sim > threshold)
        if len(point_coords[0]) == 0:
            return None, None

        fg_coords_scores = torch.stack(point_coords[::-1] + (mask_sim[point_coords],), dim=0).T

        max_len = max(original_size)
        ratio = self.image_size / max_len
        _, width = map(lambda x: int(x * ratio), original_size)
        n_w = width // downsizing

        res = (fg_coords_scores[:, 1] * ratio // downsizing * n_w + fg_coords_scores[:, 0] * ratio // downsizing).to(
            torch.int32
        )
        points_scores = torch.stack([fg_coords_scores[res == r][0] for r in torch.unique(res)], dim=0)
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

        self.prompt_getter = PromptGetter(image_size=config.model.image_size)
        self.prompt_getter.initialize()
        self.prompt_getter.set_default_thresholds(
            config.model.default_threshold_reference, config.model.default_threshold_target
        )

        if prompt_getter_reference_feats:
            self.prompt_getter.reference_feats = prompt_getter_reference_feats
        if prompt_getter_reference_prompts:
            self.prompt_getter.reference_prompts = prompt_getter_reference_prompts

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
        padding: Tuple[int, ...],
        original_size: Tuple[int, int],
    ) -> None:
        """Get reference features.

        Using given images, get reference features and save it to PromptGetter.
        These reference features will be used for `infer` to get target results.
        Currently, single batch is only supported.

        Args:
            images (torch.Tensor): Given images for reference features.
            processed_prompts (Dict[ScoredLabel, List[Dict[str, torch.Tensor]]]): The whole class-wise prompts
                processed at _preprocess_prompts.
            padding (Tuple[int, ...]): Padding size.
            original_size (Tuple[int, int]): Original image size.
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
            reference_prompt = torch.zeros(original_size, dtype=torch.uint8, device=images.device)
            for input_prompt in input_prompts:
                if "annotation" in input_prompt:
                    # directly use annotation information as a mask
                    reference_prompt[input_prompt.get("annotation") == 1] += 1
                else:
                    merged_input_prompts = self._merge_prompts(label, input_prompt, processed_prompts)
                    masks, scores, logits = self._predict_mask(
                        image_embeddings=image_embeddings,
                        input_prompts=merged_input_prompts,
                        padding=padding,
                        original_size=original_size,
                        multimask_output=True,
                    )
                    best_idx = torch.argmax(scores)
                    reference_prompt[masks[0, best_idx]] += 1
            reference_prompt = torch.clip(reference_prompt, 0, 1)

            ref_mask = torch.tensor(reference_prompt, dtype=torch.float32)
            reference_feat = None
            default_threshold_reference = deepcopy(self.prompt_getter.default_threshold_reference)
            while reference_feat is None:
                logger.info(f"[*] default_threshold_reference : {default_threshold_reference:.4f}")
                reference_feat = self._generate_masked_features(
                    ref_feat, ref_mask, default_threshold_reference, padding=padding
                )
                default_threshold_reference -= 0.05

            self.prompt_getter.set_reference(label, reference_feat.detach().cpu(), reference_prompt.detach().cpu())

    @torch.no_grad()
    def infer(
        self, images: torch.Tensor, padding: Tuple[int, ...], original_size: Tuple[int, int]
    ) -> List[List[DefaultDict[int, List[torch.Tensor]]]]:
        """Zero-shot inference with reference features.

        Get target results by using reference features and target images' features.

        Args:
            images (torch.Tensor): Given images for target results.
            padding (Tuple[int, ...]): Padding size.
            original_size (Tuple[int, int]): Original image size.

        Returns:
            (List[List[DefaultDict[int, List[torch.Tensor]]]]): Target results.
                Lists wrapping results is following this order:
                    1. Target images
                    2. Tuple of predicted masks and used points gotten by point selection
        """
        assert images.shape[0] == 1, "Only single batch is supported."

        total_results = []
        # num_classes = len(self.reference_feats.keys())
        for image in images:
            if image.ndim == 3:
                image = image.unsqueeze(0)

            image_embeddings = self.image_encoder(images)

            prompts = self.prompt_getter(
                image_embeddings=image_embeddings, padding=padding, original_size=original_size
            )
            predicted_masks: defaultdict = defaultdict(list)
            used_points: defaultdict = defaultdict(list)
            for label, (points_scores, bg_coords) in prompts.items():
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

                    mask, used_point_score = self(
                        image_embeddings=image_embeddings,
                        points_score=points_score,
                        bg_coords=bg_coords,
                        padding=padding,
                        original_size=original_size,
                    )
                    predicted_masks[label].append(mask)
                    used_points[label].append(used_point_score)

            total_results.append([predicted_masks, used_points])
        return total_results

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        points_score: torch.Tensor,
        bg_coords: torch.Tensor,
        padding: Tuple[int, ...],
        original_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict point prompts and predicted masks.

        Args:
            image_embeddings (torch.Tensor): The image embedding with a batch index of length 1.
            points_score (torch.Tensor): Foreground point prompts from point selection algorithm.
            bg_coords (torch.Tensor): Background point prompts from point selection algorithm.
            padding (Tuple[int, ...]): Padding size.
            original_size (Tuple[int, int]): Original image size.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): Predicted masks and used points with corresponding score.
        """
        point_coords = torch.cat((points_score[:2].unsqueeze(0), bg_coords), dim=0).unsqueeze(0)
        point_coords = ResizeLongestSide.apply_coords(point_coords, original_size, self.config.model.image_size)
        point_labels = torch.tensor([1] + [0] * len(bg_coords), dtype=torch.int32).unsqueeze(0)
        mask = self._predict_target_mask(
            image_embeddings=image_embeddings,
            input_prompts={"points": (point_coords, point_labels)},
            padding=padding,
            original_size=original_size,
        )

        return mask.detach().cpu().to(torch.uint8), points_score.detach().cpu()

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
        results = self.infer(
            images=batch["images"], padding=batch.get("padding")[0], original_size=batch.get("original_size")[0]
        )
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
        self, feats: torch.Tensor, masks: torch.Tensor, threshold_mask: float, padding: Optional[Tuple[int, ...]] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Generate masked features.

        Args:
            feats (torch.Tensor): Raw reference features. It will be filtered with masks.
            masks (torch.Tensor): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.
            padding (Tuple[int, ...], optional): Padding size.

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
        masks = self._preprocess_mask(masks)
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=feats.shape[0:2], mode="bilinear").squeeze()

        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None, None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0).unsqueeze(0)
        masked_feat = masked_feat / masked_feat.norm(dim=-1, keepdim=True)

        return masked_feat

    def _preprocess_mask(self, x: torch.Tensor) -> torch.Tensor:
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

    def _predict_target_mask(
        self,
        image_embeddings: torch.Tensor,
        input_prompts: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        padding: Tuple[int, ...],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Predict target masks.

        Args:
            image_embeddings (torch.Tensor): The image embedding with a batch index of length 1.
            input_prompts (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): Dictionary including point, box,
                and mask prompts. index=1 of tuple is point labels which indicate whether foreground or background.
            padding (Tuple[int, ...]): Padding size.
            original_size (Tuple[int, int]): Original image size.

        Return:
            (torch.Tensor): Predicted mask.
        """
        # First-step prediction
        _, _, logits = self._predict_mask(
            image_embeddings, input_prompts, padding, original_size, multimask_output=False
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        input_prompts.update({"masks": logits[:, best_idx : best_idx + 1, :, :]})
        masks, scores, logits = self._predict_mask(
            image_embeddings, input_prompts, padding, original_size, multimask_output=True
        )
        best_idx = torch.argmax(scores)

        # Cascaded Post-refinement-2
        coords = torch.nonzero(masks[0, best_idx])
        y, x = coords[:, 0], coords[:, 1]
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_prompts.update(
            {
                "masks": logits[:, best_idx : best_idx + 1, :, :],
                "box": torch.tensor([x_min, y_min, x_max, y_max], device=logits.device),
            }
        )
        masks, scores, _ = self._predict_mask(
            image_embeddings, input_prompts, padding, original_size, multimask_output=True
        )
        best_idx = torch.argmax(scores)

        return masks[0, best_idx]

    def _predict_mask(
        self,
        image_embeddings: torch.Tensor,
        input_prompts: Dict[str, torch.Tensor],
        padding: Tuple[int, ...],
        original_size: Tuple[int, int],
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """Predict target masks.

        Args:
            image_embeddings (torch.Tensor): The image embedding with a batch index of length 1.
            input_prompts (Dict[str, torch.Tensor]): Dictionary including point, box, and mask prompts.
            padding (Tuple[int, ...]): Padding size.
            original_size (Tuple[int, int]): Original image size.
            multimask_output (bool): Whether getting multi mask outputs or not. Defaults to True.

        Return:
            (Tuple[torch.Tensor, ...]): Predicted mask, score, and logit.
        """
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=input_prompts.get("points", None),
            boxes=input_prompts.get("box", None),  # TODO (sungchul): change key box -> boxes to use **input_prompts
            masks=input_prompts.get("masks", None),
        )

        low_res_masks, scores = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        high_res_masks = self.postprocess_masks(
            low_res_masks, (self.config.model.image_size, self.config.model.image_size), padding, original_size
        )
        masks = high_res_masks > self.config.model.mask_threshold

        return masks, scores, low_res_masks

    def set_metrics(self) -> None:
        """Skip set_metrics unused in zero-shot learning."""
        pass

    def configure_optimizers(self) -> None:
        """Skip configure_optimizers unused in zero-shot learning."""
        pass

    def training_epoch_end(self, outputs) -> None:
        """Skip training_epoch_end unused in zero-shot learning."""
        pass
