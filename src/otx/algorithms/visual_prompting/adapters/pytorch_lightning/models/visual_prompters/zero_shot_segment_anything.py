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

logger = get_logger()


class ZeroShotSegmentAnything(SegmentAnything):
    def __init__(self, config: DictConfig, state_dict: Optional[OrderedDict] = None) -> None:
        if not config.model.freeze_image_encoder:
            logger.warning("config.model.freeze_image_encoder(=False) must be set to True, changed.")
            config.model.freeze_image_encoder = True

        if not config.model.freeze_prompt_encoder:
            logger.warning("config.model.freeze_prompt_encoder(=False) must be set to True, changed.")
            config.model.freeze_prompt_encoder = True

        if not config.model.freeze_mask_decoder:
            logger.warning("config.model.freeze_mask_decoder(=False) must be set to True, changed.")
            config.model.freeze_mask_decoder = True
            
        self.default_threshold_reference = config.default_threshold_reference
        super().__init__(config, state_dict)
        
        
    def set_metrics(self) -> None:
        pass
    
    def configure_optimizers(self) -> None:
        pass
    
    def _initialize_reference(self) -> None:
        """Initialize reference information."""
        self.reference_feats: List[torch.Tensor] = []
        self.reference_embeddings: List[torch.Tensor] = []
        # self.reference_logit = None
    
    @torch.no_grad()
    def learn(self, images: torch.Tensor, processed_prompts: Dict[Any]): # TODO (sungchul): fix type information
        """Get reference features."""
        self._initialize_reference()
        
        _, _, height, width = images.shape

        reference_feats = []
        reference_embeddings = []
        results_prompts = []
        for label in range(2): # TODO (sungchul): support multi class
            if label == 0:
                # background
                continue
            
            if label not in processed_prompts:
                # for empty class
                reference_feats.append(None)
                reference_embeddings.append(None)
                results_prompts.append(torch.zeros((height, width)))
                continue
                
            # generate reference mask
            results_prompt = torch.zeros((height, width), dtype=torch.uint8)
            input_prompts = processed_prompts.get(label)
            for input_prompt in input_prompts:
                if "annotation" in input_prompt:
                    # directly use annotation information as a mask
                    results_prompt[input_prompt.get("annotation") == 1] += 1
                else:
                    merged_input_prompts = self._merge_prompts(label, input_prompt, processed_prompts)
                    # merged_input_prompts.update({"mask_input": self.reference_logit})
                    masks, scores, logits, _ = self.model.predict(**merged_input_prompts, multimask_output=True)
                    # if params.get("use_logit", DEFAULT_SETTINGS["reference"]["use_logit"]):
                    #     self.reference_logit = logits[best_idx][None]
                    best_idx = torch.argmax(scores)
                    results_prompt[masks[best_idx]] += 1
            results_prompt = torch.clip(results_prompt, 0, 1)
            
            ref_feat = self.image_encoder(images).squeeze().permute(1, 2, 0)
            ref_mask = torch.tensor(results_prompt, dtype=torch.float32)
            reference_feat = None
            default_threshold_reference = deepcopy(self.default_threshold_reference)
            while reference_feat is None:
                logger.info(f"[*] default_threshold_reference : {default_threshold_reference:.4f}")
                reference_feat, reference_embedding = self._generate_masked_features(ref_feat, ref_mask, default_threshold_reference)
                default_threshold_reference -= 0.05

            reference_feats.append(reference_feat)
            reference_embeddings.append(reference_embedding)
            results_prompts.append(results_prompt)
            
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
        
        return reference_feats, reference_embeddings, results_prompts
    
    @torch.no_grad()
    def infer(self):
        """Get results from given images & prompts"""

    @torch.no_grad()
    def forward(self):
        """Zero-shot inference with reference features."""
        return self.infer()
    
    def training_step(self, batch, batch_idx):
        """Learn"""
        images = batch["images"]
        bboxes = batch["bboxes"]
        points = batch["points"]
        annotations = batch["annotations"]
        
        return self.learn(images, ...)
    
    def validation_step(self, batch, batch_idx):
        """No validation step."""
        pass
    
    def _generate_masked_features(self, feats: torch.Tensor, masks: torch.Tensor, threshold_mask: float) -> Tuple[torch.Tensor, ...]:
        """Generate masked features.
        
        Args:
            feats (torch.Tensor): Raw reference features. It will be filtered with masks.
            masks (torch.Tensor): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.

        Returns:
            (torch.Tensor): Masked features.
            (torch.Tensor): Masked embeddings used for semantic prompting.
        """
        # Post-process masks
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=self.model.input_size, mode="bilinear").squeeze()
        masks = self.model.preprocess_mask(masks)
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=feats.shape[0: 2], mode="bilinear").squeeze()
        
        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None, None

        masked_feat = feats[masks > threshold_mask]
        masked_embedding = masked_feat.mean(0).unsqueeze(0)    
        masked_feat = masked_embedding / masked_embedding.norm(dim=-1, keepdim=True)
        masked_embedding = masked_embedding.unsqueeze(0)
        
        return masked_feat, masked_embedding
    
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
