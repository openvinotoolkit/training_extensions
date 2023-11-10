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
    def learn(self, images: torch.Tensor, processed_prompts: Dict[str, Any], **kwargs): # TODO (sungchul): fix type information
        """Get reference features."""
        self._initialize_reference()
        
        _, _, height, width = images.shape

        image_embeddings = self.image_encoder(images)
        ref_feat = image_embeddings.squeeze().permute(1, 2, 0)
        
        reference_feats = []
        reference_embeddings = []
        results_prompts = []
        
        num_classes = max(processed_prompts.keys()) + 1
        for label in range(num_classes): # TODO (sungchul): support multi class
            if label == 0:
                # skip background
                continue
            
            if label not in processed_prompts:
                # for empty class
                reference_feats.append(None)
                reference_embeddings.append(None)
                results_prompts.append(torch.zeros((height, width)))
                continue
                
            # generate reference mask
            results_prompt = torch.zeros(kwargs["original_size"][0], dtype=torch.uint8, device=images.device)
            input_prompts = processed_prompts.get(label)
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
        processed_prompts, label_dict = self._preprocess_prompts(bboxes=bboxes, labels=labels)
        
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
        label_dict = {}
        # TODO (sungchul): will be updated
        if bboxes:
            for bbox, label in zip(bboxes[0], labels[0]):
                label_id = int(label.id)
                processed_prompts[label_id].append({"box": bbox.reshape(-1, 4)})
                if label_id not in label_dict:
                    label_dict[label_id] = label
                
        if points:
            pass
        
        if annotations:
            pass
            
        # for prompt in prompts:
        #     processed_prompt = {}
        #     if prompt.get("type") == "point":
        #         processed_prompt.update({
        #             "point_coords": prompt.get("point_coords"),
        #             "point_labels": prompt.get("point_labels"),
        #         })

        #     elif prompt.get("type") == "polygon":
        #         polygon = prompt.get("point_coords")
        #         if convert_mask:
        #             # convert polygon to mask
        #             contour = [[int(point[0]), int(point[1])] for point in polygon]
        #             gt_mask = np.zeros((height, width), dtype=np.uint8)
        #             gt_mask = cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, -1)

        #             # randomly sample points from generated mask
        #             ys, xs, _ = np.nonzero(gt_mask)
        #         else:
        #             ys, xs = polygon[:,1], polygon[:,0]

        #         rand_idx = np.random.permutation(len(ys))[:num_sample]
        #         _point_coords = []
        #         _point_labels = []
        #         for x, y in zip(xs[rand_idx], ys[rand_idx]):
        #             _point_coords.append([x, y])
        #             _point_labels.append(prompt.get("point_labels")[0])

        #         processed_prompt.update({
        #             "point_coords": np.array(_point_coords),
        #             "point_labels": np.array(_point_labels),
        #         })

        #     elif prompt.get("type") == "box":
        #         processed_prompt.update({
        #             "box": prompt.get("point_coords").reshape(-1, 4),
        #         })

        #     elif prompt.get("type") == "annotation":
        #         polygon = prompt.get("point_coords")
        #         contour = [[int(point[0]), int(point[1])] for point in polygon]
        #         gt_mask = np.zeros((height, width), dtype=np.uint8)
        #         gt_mask = cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, -1)
        #         processed_prompt.update({"annotation": gt_mask})

        #     processed_prompts[prompt.get("label", 0)].append(processed_prompt)

        # # aggregate annotations
        # for _, prompts in processed_prompts.items():
        #     annotations = []
        #     pop_idx = []
        #     for idx, prompt in enumerate(prompts):
        #         if "annotation" in prompt:
        #             annotations.append(prompt.get("annotation"))
        #             pop_idx.append(idx)

        #     if len(pop_idx) > 0:
        #         for idx in pop_idx[::-1]:
        #             prompts.pop(idx)

        #     if len(annotations) > 0:
        #         dtype = annotations[0].dtype
        #         prompts.append({"annotation": np.logical_or.reduce(annotations).astype(dtype)})

        processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x[0]))
        return processed_prompts, label_dict
    
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
