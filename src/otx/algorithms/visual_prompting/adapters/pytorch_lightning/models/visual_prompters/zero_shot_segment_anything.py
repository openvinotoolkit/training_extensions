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
            
        self.default_threshold_reference = config.model.default_threshold_reference
        super().__init__(config, state_dict)
    
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
                    sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        boxes=merged_input_prompts["box"], # TODO (sungchul): change key box -> boxes
                        # TODO (sungchul): support other prompts
                        points=None,
                        masks=None,
                    )

                    low_res_masks, scores = self.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                    )
                    high_res_masks = self.postprocess_masks(low_res_masks, images.shape[2:], kwargs["padding"][0], kwargs["original_size"][0])
                    masks = high_res_masks > self.config.model.mask_threshold
                    best_idx = torch.argmax(scores)
                    results_prompt[masks[0,best_idx]] += 1
            results_prompt = torch.clip(results_prompt, 0, 1)
            
            ref_mask = torch.tensor(results_prompt, dtype=torch.float32)
            reference_feat = None
            default_threshold_reference = deepcopy(self.default_threshold_reference)
            while reference_feat is None:
                logger.info(f"[*] default_threshold_reference : {default_threshold_reference:.4f}")
                reference_feat = self._generate_masked_features(ref_feat, ref_mask, default_threshold_reference)
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
    def infer(self, images: torch.Tensor, processed_prompts: Dict[ScoredLabel, List[Dict[str, torch.Tensor]]]):
        """Zero-shot inference with reference features."""

    @torch.no_grad()
    def forward(self):
        """Zero-shot inference with reference features."""
        return self.infer()
    
    def training_step(self, batch, batch_idx):
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
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=self.config.model.image_size, mode="bilinear").squeeze()
        masks = self.preprocess_mask(masks)
        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0), size=feats.shape[0: 2], mode="bilinear").squeeze()
        
        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None, None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0).unsqueeze(0)    
        masked_feat = masked_feat / masked_feat.norm(dim=-1, keepdim=True)
        
        return masked_feat
    
    def preprocess_mask(self, x: torch.Tensor) -> torch.Tensor:
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
        
    def set_metrics(self) -> None:
        pass
    
    def configure_optimizers(self) -> None:
        pass

    def training_epoch_end(self, outputs) -> None:
        pass
