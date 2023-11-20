"""SAM module for visual prompting zero-shot learning."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import DictConfig
from .segment_anything import SegmentAnything
from otx.algorithms.common.utils.logger import get_logger
from copy import deepcopy
from collections import defaultdict
from otx.api.entities.scored_label import ScoredLabel
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.pipelines import ResizeLongestSide

logger = get_logger()


class ZeroShotSegmentAnything(SegmentAnything):
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
            
        self.default_threshold_reference = config.model.default_threshold_reference
        self.default_threshold_target = config.model.default_threshold_target
        self._initialize_reference()
        if state_dict:
            if "reference_feats" in state_dict:
                self.reference_feats = state_dict.pop("reference_feats")
                
            if "reference_prompts" in state_dict:
                self.reference_prompts = state_dict.pop("reference_prompts")

        super().__init__(config, state_dict)
        
    def set_default_config(self) -> DictConfig:
        return DictConfig({"model": {
            "backbone": "tiny_vit",
            "checkpoint": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
            "default_threshold_reference": 0.3,
            "default_threshold_target": 0.65,
            "freeze_image_encoder": True,
            "freeze_mask_decoder": True,
            "freeze_prompt_encoder": True,
            "image_size": 1024,
            "mask_threshold": 0.0,
        }})
    
    def _initialize_reference(self) -> None:
        """Initialize reference information."""
        self.reference_feats = {}
        self.reference_prompts = {}
    
    @torch.no_grad()
    def learn(self, images: torch.Tensor, processed_prompts: Dict[ScoredLabel, List[Dict[str, torch.Tensor]]], **kwargs): # TODO (sungchul): fix type information
        """Get reference features."""
        assert images.shape[0] == 1, "Only single batch is supported."
        
        self._initialize_reference()

        image_embeddings = self.image_encoder(images)
        ref_feat = image_embeddings.squeeze().permute(1, 2, 0)
        
        for label, input_prompts in processed_prompts.items():
            if label.name.lower() == "background":
                # skip background
                # TODO (sungchul): how to skip background class
                continue
                
            # generate reference mask
            # TODO (sungchul): ensemble multi reference features (current : use merged masks)
            results_prompt = torch.zeros(kwargs["original_size"][0], dtype=torch.uint8, device=images.device)
            for input_prompt in input_prompts:
                if "annotation" in input_prompt:
                    # directly use annotation information as a mask
                    results_prompt[input_prompt.get("annotation") == 1] += 1
                else:
                    merged_input_prompts = self._merge_prompts(label, input_prompt, processed_prompts)
                    # masks, scores = self.mask_decoder.predict_masks(**merged_input_prompts, multimask_output=True)
                    
                    masks, scores, logits = self._predict_mask(image_embeddings, merged_input_prompts, images.shape[2:], multimask_output=True, **kwargs)
                    best_idx = torch.argmax(scores)
                    results_prompt[masks[0,best_idx]] += 1
            results_prompt = torch.clip(results_prompt, 0, 1)
            
            ref_mask = torch.tensor(results_prompt, dtype=torch.float32)
            reference_feat = None
            default_threshold_reference = deepcopy(self.default_threshold_reference)
            while reference_feat is None:
                logger.info(f"[*] default_threshold_reference : {default_threshold_reference:.4f}")
                reference_feat = self._generate_masked_features(ref_feat, ref_mask, default_threshold_reference, **kwargs)
                default_threshold_reference -= 0.05

            self.reference_feats[int(label.id)] = reference_feat.detach().cpu()
            self.reference_prompts[int(label.id)] = results_prompt.detach().cpu()
        
        # TODO (sungchul): predict reference input results
            
        # self.reference_feats.append(reference_feats)
        # self.reference_embeddings.append(reference_embeddings)
        # reference_results["results_reference"].append(
        #     np.concatenate((
        #         np.zeros((height, width, 1)), np.stack(results_reference, axis=2)
        #     ), axis=2)
        # )

        # for image, manual_ref_feats in zip(images, self.reference_feats):
        #     total_predicted_masks, total_used_points = self._infer_target_segmentation(
        #         [image], params.get("target_params", DEFAULT_SETTINGS["reference"]["target_params"]), manual_ref_feats)
        #     reference_results["results_target"] += total_predicted_masks
        #     reference_results["results_target_points"] += total_used_points
    
    @torch.no_grad()
    def infer(self, images: torch.Tensor, **kwargs):
        """Zero-shot inference with reference features."""
        total_results = []
        num_classes = len(self.reference_feats.keys())
        for image in images:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            
            _, _, h_img, w_img = image.shape
            target_embeddings = self.image_encoder(images)
            target_feat = target_embeddings.squeeze()
            c_feat, h_feat, w_feat = target_feat.shape
            target_feat = self._preprocess_target_feat(target_feat, c_feat, h_feat, w_feat)
            
            predicted_masks = defaultdict(list)
            used_points = defaultdict(list)
            for label, reference_feat in self.reference_feats.items():
                sim = reference_feat.to(target_feat.device) @ target_feat
                sim = sim.reshape(1, 1, h_feat, w_feat)
                sim = self.postprocess_masks(sim, image.shape[2:], kwargs["padding"][0], kwargs["original_size"][0]).squeeze()
                
                threshold = 0.85 * sim.max() if num_classes > 1 else self.default_threshold_target
                points_scores, bg_coords = self._point_selection(sim, kwargs["original_size"][0], threshold)
                if points_scores is None:
                    # skip if there is no point with score > threshold
                    continue

                for x, y, score in points_scores:
                    is_done = False
                    for pm in predicted_masks.get(label, []):
                        # check if that point is already assigned
                        if pm[int(y), int(x)] > 0:
                            is_done = True
                            break
                    if is_done:
                        continue
                    
                    point_coords = torch.cat((torch.tensor([[x, y]], device=bg_coords.device), bg_coords), dim=0).unsqueeze(0)
                    point_coords = ResizeLongestSide.apply_coords(point_coords, kwargs["original_size"][0], h_img)
                    point_labels = torch.tensor([1] + [0] * len(bg_coords), dtype=torch.int32).unsqueeze(0)
                    mask = self(
                        image_embeddings=target_embeddings,
                        input_prompts={"points": (point_coords, point_labels)},
                        image_shape=(h_img, w_img),
                        **kwargs
                    )

                    mask = mask.detach().cpu().to(torch.uint8)
                    # set bbox based on predicted mask
                    predicted_masks[label].append(mask)
                    used_points[label].append((float(x), float(y), float(score)))
            total_results.append([predicted_masks, used_points])
        return total_results

    @torch.no_grad()
    def forward(self, image_embeddings: torch.Tensor, input_prompts: Dict[str, Any], image_shape: Tuple[int], **kwargs) -> torch.Tensor:
        """Predict target masks.
        
        Args:
            point_coords (torch.Tensor): Selected points as point prompts from similarity map.
            point_labels (torch.Tensor): Labels that are set in foreground or background.

        Return:
            (torch.Tensor): Predicted mask.
        """
        # First-step prediction
        _, _, logits = self._predict_mask(image_embeddings, input_prompts, image_shape, multimask_output=False, **kwargs)
        best_idx = 0

        # Cascaded Post-refinement-1
        input_prompts.update({"masks": logits[:, best_idx: best_idx + 1, :, :]})
        masks, scores, logits = self._predict_mask(image_embeddings, input_prompts, image_shape, multimask_output=True, **kwargs)
        best_idx = torch.argmax(scores)

        # Cascaded Post-refinement-2
        coords = torch.nonzero(masks[0, best_idx])
        y, x = coords[:,0], coords[:,1]
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_prompts.update({
            "masks": logits[:, best_idx: best_idx + 1, :, :],
            "box": torch.tensor([x_min, y_min, x_max, y_max], device=logits.device)
        })
        masks, scores, _ = self._predict_mask(image_embeddings, input_prompts, image_shape, multimask_output=True, **kwargs)
        best_idx = torch.argmax(scores)
        
        return masks[0, best_idx]
    
    def training_step(self, batch, batch_idx) -> None:
        """Learn"""
        images = batch["images"]
        
        # TODO (sungchul): each prompt will be assigned with each label
        bboxes = batch["bboxes"]
        labels = batch["labels"]
        # TODO (sungchul): support other below prompts
        # points = batch["points"]
        # annotations = batch["annotations"]
        
        # organize prompts based on label
        processed_prompts = self._preprocess_prompts(bboxes=bboxes, labels=labels)
        
        kwargs = {
            "original_size": batch.get("original_size"),
            "padding": batch.get("padding"),
        }
        self.learn(images, processed_prompts, **kwargs)
        
    def predict_step(self, batch, batch_idx):
        """Predict step."""
        images = batch["images"]
        
        kwargs = {
            "original_size": batch.get("original_size"),
            "padding": batch.get("padding"),
        }
        results = self.infer(images, **kwargs)
        return [result[0] for result in results] # tmp: only mask
    
    def _preprocess_prompts(
        self,
        bboxes: Optional[List[torch.Tensor]] = None,
        points: Optional[List[torch.Tensor]] = None,
        annotations: Optional[List[torch.Tensor]] = None,
        labels: Optional[List[torch.Tensor]] = None,
        # height: int = 1024,
        # width: int = 1024,
    ) -> Dict[str, Dict[str, Any]]:
        """Preprocess prompts.

        This function proceeds such below thigs:
            1. Gather prompts which have the same labels
            3. If there are box prompts, the key `point_coords` is changed to `box`
        
        Args:
            prompts (list): Given prompts to be processed.
            height (int): Image height.
            width (int): Image width.
            
        Returns:
            (dict): Processed and arranged each single prompt using label information as keys.
                Unlike other prompts, `annotation` prompts will be aggregated as single annotation.

                [Example]
                processed_prompts = {
                    0: [ # background
                        {
                            "point_coords": torch.Tensor,
                            "point_labels": torch.Tensor,
                        },
                        {
                            "box": torch.Tensor,
                        },
                        {
                            "annotation": torch.Tensor, # there is only single processed annotation prompt
                        },
                        ...
                    ],
                    1: [
                        {
                            "point_coords": torch.Tensor,
                            "point_labels": torch.Tensor,
                        },
                        {
                            "point_coords": torch.Tensor,
                            "point_labels": torch.Tensor,
                        },
                        {
                            "annotation": torch.Tensor, # there is only single processed annotation prompt
                        },
                        ...
                    ],
                    2: [
                        {
                            "box": torch.Tensor,
                        },
                        {
                            "box": torch.Tensor,
                        },
                        ...
                    ]
                }
        """
        processed_prompts = defaultdict(list)
        # TODO (sungchul): will be updated
        if bboxes:
            for bbox, label in zip(bboxes[0], labels[0]):
                processed_prompts[label].append({"box": bbox.reshape(-1, 4)})
                
        if points:
            pass
        
        if annotations:
            pass

        processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x[0]))
        return processed_prompts
    
    def _generate_masked_features(self, feats: torch.Tensor, masks: torch.Tensor, threshold_mask: float, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Generate masked features.
        
        Args:
            feats (torch.Tensor): Raw reference features. It will be filtered with masks.
            masks (torch.Tensor): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.

        Returns:
            (torch.Tensor): Masked features.
            (torch.Tensor): Masked embeddings used for semantic prompting.
        """
        if kwargs:
            image_padding = kwargs["padding"][0]
            resized_size = (self.config.model.image_size - image_padding[1] - image_padding[3], self.config.model.image_size - image_padding[0] - image_padding[2])
        else:
            resized_size = (self.config.model.image_size, self.config.model.image_size)

        # Post-process masks
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=resized_size, mode="bilinear").squeeze()
        masks = self._preprocess_mask(masks)
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=feats.shape[0: 2], mode="bilinear").squeeze()
        
        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None, None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0).unsqueeze(0)    
        masked_feat = masked_feat / masked_feat.norm(dim=-1, keepdim=True)
        
        return masked_feat
    
    def _preprocess_target_feat(self, target_feat: torch.Tensor, c_feat: int, h_feat: int, w_feat: int) -> torch.Tensor:
        target_feat = target_feat / target_feat.norm(dim=0, keepdim=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)
        return target_feat
    
    def _preprocess_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.config.model.image_size - h
        padw = self.config.model.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def _update_value(self, target: Dict[str, Any], key: str, value: torch.Tensor) -> None:
        """Update tensor to target dictionary."""
        if key in target:
            target[key] = torch.cat((target[key], value))
        else:
            target[key] = value

    def _merge_prompts(self, label: int, input_prompts: Dict[str, Any], processed_prompts: Dict[str, Any], use_only_background: bool = True) -> Dict[str, Any]:
        """Merge target prompt and other prompts.

        Merge a foreground prompt and other prompts (background or prompts with other classes).
        
        Args:
            label (int): Label information. Background is 0 and other foregrounds are >= 0.
            input_prompts (dict): A foreground prompt to be merged with other prompts.
            processed_prompts (dict): The whole class-wise prompts processed at _preprocess_prompts.
            use_only_background (bool): Whether merging only background prompt, defaults to True. It is applied to only point_coords.
        """
        merged_input_prompts = deepcopy(input_prompts)
        for other_label, other_input_prompts in processed_prompts.items():
            if other_label == label:
                continue
            if (use_only_background and other_label == 0) or (not use_only_background):
                # only add point (and scribble) prompts
                # use_only_background=True -> background prompts are only added as background
                # use_only_background=False -> other prompts are added as background
                for other_input_prompt in other_input_prompts:
                    if "point_coords" in other_input_prompt:
                        # point, scribble
                        self._update_value(merged_input_prompts, "point_coords", other_input_prompt.get("point_coords"))
                        self._update_value(merged_input_prompts, "point_labels", torch.zeros_like(other_input_prompt.get("point_labels")))
        return merged_input_prompts
    
    def _point_selection(
        self,
        mask_sim: torch.Tensor,
        original_size: Tuple[int],
        threshold: float,
        num_bg_points: int = 1,
        downsizing: int = 16
    ) -> Tuple[Dict, ...]:
        """Select point used as point prompts."""
        _, w_sim = mask_sim.shape

        # Top-last point selection
        bg_indices = mask_sim.flatten().topk(num_bg_points, largest=False)[1]
        bg_x = (bg_indices // w_sim).unsqueeze(0)
        bg_y = (bg_indices - bg_x * w_sim)
        bg_coords = torch.cat((bg_y, bg_x), dim=0).permute(1, 0)
        bg_coords = bg_coords

        point_coords = torch.where(mask_sim > threshold)
        if len(point_coords[0]) == 0:
            return None, None
        
        fg_coords_scores = torch.stack(point_coords[::-1] + (mask_sim[point_coords],), dim=0).T
        
        max_len = max(original_size)
        ratio = self.config.model.image_size / max_len
        _, width = map(lambda x: int(x * ratio), original_size)
        n_w = width // downsizing
        
        res = (fg_coords_scores[:,1] * ratio // downsizing * n_w + fg_coords_scores[:,0] * ratio // downsizing).to(torch.int32)
        points_scores = torch.stack([fg_coords_scores[res == r][0] for r in torch.unique(res)], dim=0)
        points_scores = points_scores[torch.argsort(points_scores[:,-1], descending=True)]

        return points_scores, bg_coords
    
    def _predict_mask(self, image_embeddings: torch.Tensor, input_prompts: Dict[str, torch.Tensor], image_shape: Tuple[int], multimask_output: bool = True, **kwargs) -> torch.Tensor:
        """Predict target masks.
        
        Args:
            point_coords (torch.Tensor): Selected points as point prompts from similarity map.
            point_labels (torch.Tensor): Labels that are set in foreground or background.

        Return:
            (torch.Tensor): Predicted mask.
        """
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=input_prompts.get("points", None),
            boxes=input_prompts.get("box", None), # TODO (sungchul): change key box -> boxes to use **input_prompts
            masks=input_prompts.get("masks", None),
        )

        low_res_masks, scores = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        high_res_masks = self.postprocess_masks(low_res_masks, image_shape, kwargs["padding"][0], kwargs["original_size"][0])
        masks = high_res_masks > self.config.model.mask_threshold
        
        return masks, scores, low_res_masks
        
    def set_metrics(self) -> None:
        pass
    
    def configure_optimizers(self) -> None:
        pass

    def training_epoch_end(self, outputs) -> None:
        pass
