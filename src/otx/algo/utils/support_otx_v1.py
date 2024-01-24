# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions to guarantee the OTX1.x models."""
from __future__ import annotations

from typing import Any


class OTXv1Helper:
    @classmethod 
    def load_cls_effnet_v2_ckpt(cls, state_dict: dict[str, Any], label_type: str, add_prefix: str = "") -> dict[str, Any]:
        """Load the OTX1.x efficientnet v2 classification checkpoints."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if key.startswith("model.classifier."):
                key = key.replace("model.classifier", "head.fc")
                if not label_type == "hierarchical":
                    val = val.t()
                elif key.startswith("model"):
                    key = "backbone." + key
            state_dict[add_prefix + key] = val
        return state_dict 
        
    @classmethod
    def load_cls_effnet_b0_ckpt(cls, state_dict: dict[str, Any], label_type: str, add_prefix: str = "") -> dict[str, Any]:
        """Load the OTX1.x efficientnet b0 classification checkpoints."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if key.startswith("features.") and "activ" not in key:
                key = "backbone." + key
            elif key.startswith("output."):
                key = key.replace("output", "head")
                if not label_type == "hlabel":
                    key = key.replace("asl", "fc")
                val = val.t()
            state_dict[add_prefix + key] = val
        return state_dict 

    @classmethod
    def load_cls_mobilenet_v3_ckpt(cls, state_dict: dict[str, Any], label_type: str, add_prefix: str = "") -> dict[str, Any]:
        """Load the OTX1.x mobilenet v3 classification checkpoints."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if key.startswith("classifier."):
                if "4" in key:
                    key = "head." + key.replace("4", "3")
                    if label_type == "multilabel":
                        val = val.t()
                else:
                    key = "head." + key
            elif key.startswith("act"):
                key = "head." + key
            elif not key.startswith("backbone."):
                key = "backbone." + key
            state_dict[add_prefix + key] = val
        return state_dict

    @classmethod
    def load_det_ckpt(cls, state_dict: dict[str, Any], add_prefix: str = "") -> dict[str, Any]:
        """Load the OTX1.x detection model checkpoints."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if not key.startswith("ema_"):
                state_dict[add_prefix + key] = val
        return state_dict 
    
    @classmethod
    def load_seg_segnext_ckpt(cls, state_dict: dict[str, Any], add_prefix: str = ""):
        """Load the OTX1.x segnext segmentation checkpoints."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if not "ham.bases" in key:
                state_dict[add_prefix + key] = val
        return state_dict
    
    @classmethod
    def load_seg_lite_hrnet_ckpt(cls, state_dict: dict[str, Any], add_prefix: str = ""):
        """Load the OTX1.x lite hrnet segmentation checkpoints."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if not "decode_head.aggregator.projects" in key:
                state_dict[add_prefix + key] = val
        return state_dict
    
    @classmethod
    def load_iseg_ckpt(cls, state_dict: dict[str, Any], add_prefix: str = "") -> dict[str, Any]:
        """Load the instance segmentation model checkpoints."""
        return cls.load_det_ckpt(state_dict, add_prefix)
    
    @classmethod
    def load_action_ckpt(cls, state_dict: dict[str, Any], add_prefix: str = "") -> dict[str, Any]:
        """Load the action cls/det model checkpoints."""
        for key in list(state_dict.keys()) :
            val = state_dict.pop(key)
            state_dict[add_prefix + key] = val
        return state_dict