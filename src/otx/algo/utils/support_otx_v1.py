# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions to guarantee the OTX1.x models."""
from __future__ import annotations


class OTXv1Helper:
    """Helper class to support the backward compatibility of OTX v1."""

    @staticmethod
    def load_common_ckpt(state_dict: dict, add_prefix: str = "") -> dict:
        """Load the OTX1.x model checkpoints that don't need special handling."""
        state_dict = state_dict["model"]["state_dict"]
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            state_dict[add_prefix + key] = val
        return state_dict

    @staticmethod
    def load_cls_effnet_b0_ckpt(state_dict: dict, label_type: str, add_prefix: str = "") -> dict:
        """Load the OTX1.x efficientnet b0 classification checkpoints."""
        state_dict = state_dict["model"]["state_dict"]
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if key.startswith("features."):
                new_key = "backbone." + key if "activ" not in key else key
            elif key.startswith("output."):
                new_key = key.replace("output", "head")
                if label_type != "hlabel":
                    new_key = new_key.replace("asl", "fc")
                    val = val.t()
            state_dict[add_prefix + new_key] = val
        return state_dict

    @staticmethod
    def load_cls_effnet_v2_ckpt(state_dict: dict, label_type: str, add_prefix: str = "") -> dict:
        """Load the OTX1.x efficientnet v2 classification checkpoints."""
        state_dict = state_dict["model"]["state_dict"]
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if key.startswith("model.classifier."):
                new_key = key.replace("model.classifier", "head.fc")
                if label_type != "hlabel":
                    val = val.t()
            elif key.startswith("model"):
                new_key = "backbone." + key
            state_dict[add_prefix + new_key] = val
        return state_dict

    @staticmethod
    def load_cls_mobilenet_v3_ckpt(state_dict: dict, label_type: str, add_prefix: str = "") -> dict:
        """Load the OTX1.x mobilenet v3 classification checkpoints."""
        state_dict = state_dict["model"]["state_dict"]
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if key.startswith("classifier."):
                if "4" in key:
                    new_key = "head." + key.replace("4", "3")
                    if label_type == "multilabel":
                        val = val.t()
                else:
                    new_key = "head." + key
            elif key.startswith("act"):
                new_key = "head." + key
            elif not key.startswith("backbone."):
                new_key = "backbone." + key
            state_dict[add_prefix + new_key] = val
        return state_dict

    @staticmethod
    def load_cls_deit_ckpt(state_dict: dict, add_prefix: str = "") -> dict:
        """Load the OTX1.x deit-tiny classification checkpoints."""
        return OTXv1Helper.load_common_ckpt(state_dict, add_prefix)

    @staticmethod
    def load_det_ckpt(state_dict: dict, add_prefix: str = "") -> dict:
        """Load the OTX1.x detection model checkpoints."""
        state_dict = state_dict["model"]["state_dict"]
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if not key.startswith("ema_"):
                state_dict[add_prefix + key] = val
        return state_dict

    @staticmethod
    def load_ssd_ckpt(state_dict: dict, add_prefix: str = "") -> dict:
        """Load OTX1.x SSD model checkpoints."""
        state_dict["model"]["state_dict"]["anchors"] = state_dict.pop("anchors", None)
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)

    @staticmethod
    def load_iseg_ckpt(state_dict: dict, add_prefix: str = "") -> dict:
        """Load the instance segmentation model checkpoints."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)

    @staticmethod
    def load_seg_segnext_ckpt(state_dict: dict, add_prefix: str = "") -> dict:
        """Load the OTX1.x segnext segmentation checkpoints."""
        state_dict = state_dict["model"]["state_dict"]
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if "ham.bases" not in key:
                state_dict[add_prefix + key] = val
        return state_dict

    @staticmethod
    def load_seg_lite_hrnet_ckpt(state_dict: dict, add_prefix: str = "") -> dict:
        """Load the OTX1.x lite hrnet segmentation checkpoints."""
        state_dict = state_dict["model"]["state_dict"]
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            state_dict[add_prefix + key] = val
        return state_dict

    @staticmethod
    def load_action_ckpt(state_dict: dict, add_prefix: str = "") -> dict:
        """Load the OTX1.x action cls/det model checkpoints."""
        return OTXv1Helper.load_common_ckpt(state_dict, add_prefix)
