# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import pytest
import torch
from otx.algo.utils.support_otx_v1 import OTXv1Helper


class TestOTXv1Helper:
    @pytest.fixture()
    def fxt_random_tensor(self) -> torch.Tensor:
        return torch.randn(3, 10)

    def _check_ckpt_pairs(self, src_state_dict: dict, dst_state_dict: dict) -> None:
        for (src_key, src_value), (dst_key, dst_value) in zip(src_state_dict.items(), dst_state_dict.items()):
            assert src_key == dst_key
            assert src_value.shape == dst_value.shape

    @pytest.mark.parametrize("label_type", ["multiclass", "multilabel", "hlabel"])
    def test_load_cls_effnet_b0_ckpt(self, label_type: str, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "features.weights": fxt_random_tensor,
                    "features.activ.weights": fxt_random_tensor,
                    "output.asl.weights": fxt_random_tensor,
                },
            },
        }

        if label_type != "hlabel":
            dst_state_dict = {
                "model.model.backbone.features.weights": fxt_random_tensor,
                "model.model.features.activ.weights": fxt_random_tensor,
                "model.model.head.fc.weights": fxt_random_tensor.t(),
            }
        else:
            dst_state_dict = {
                "model.model.backbone.features.weights": fxt_random_tensor,
                "model.model.features.activ.weights": fxt_random_tensor,
                "model.model.head.asl.weights": fxt_random_tensor,
            }

        converted_state_dict = OTXv1Helper.load_cls_effnet_b0_ckpt(
            src_state_dict,
            label_type,
            add_prefix="model.model.",
        )
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)

    @pytest.mark.parametrize("label_type", ["multiclass", "multilabel", "hlabel"])
    def test_load_cls_effnet_v2_ckpt(self, label_type: str, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "model.weights": fxt_random_tensor,
                    "model.classifier.weights": fxt_random_tensor,
                },
            },
        }

        if label_type != "hlabel":
            dst_state_dict = {
                "model.model.backbone.model.weights": fxt_random_tensor,
                "model.model.head.fc.weights": fxt_random_tensor.t(),
            }
        else:
            dst_state_dict = {
                "model.model.backbone.model.weights": fxt_random_tensor,
                "model.model.head.fc.weights": fxt_random_tensor,
            }

        converted_state_dict = OTXv1Helper.load_cls_effnet_v2_ckpt(
            src_state_dict,
            label_type,
            add_prefix="model.model.",
        )
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)

    @pytest.mark.parametrize("label_type", ["multiclass", "multilabel", "hlabel"])
    def test_load_cls_mobilenet_v3_ckpt(self, label_type: str, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "model.weights": fxt_random_tensor,
                    "classifier.2.weights": fxt_random_tensor,
                    "classifier.4.weights": fxt_random_tensor,
                    "act.weights": fxt_random_tensor,
                },
            },
        }

        if label_type == "multilabel":
            dst_state_dict = {
                "model.model.backbone.model.weights": fxt_random_tensor,
                "model.model.head.classifier.2.weights": fxt_random_tensor,
                "model.model.head.classifier.3.weights": fxt_random_tensor.t(),
                "model.model.head.act.weights": fxt_random_tensor,
            }
        else:
            dst_state_dict = {
                "model.model.backbone.model.weights": fxt_random_tensor,
                "model.model.head.classifier.2.weights": fxt_random_tensor,
                "model.model.head.classifier.3.weights": fxt_random_tensor,
                "model.model.head.act.weights": fxt_random_tensor,
            }

        converted_state_dict = OTXv1Helper.load_cls_mobilenet_v3_ckpt(
            src_state_dict,
            label_type,
            add_prefix="model.model.",
        )
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)

    def test_load_det_ckpt(self, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "model.weights": fxt_random_tensor,
                    "head.weights": fxt_random_tensor,
                    "ema_model.weights": fxt_random_tensor,
                },
            },
        }

        dst_state_dict = {
            "model.model.model.weights": fxt_random_tensor,
            "model.model.head.weights": fxt_random_tensor,
        }

        converted_state_dict = OTXv1Helper.load_det_ckpt(src_state_dict, add_prefix="model.model.")
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)

    def test_load_ssd_ckpt(self, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "model.weights": fxt_random_tensor,
                    "head.weights": fxt_random_tensor,
                    "ema_model.weights": fxt_random_tensor,
                },
            },
            "anchors": fxt_random_tensor,
        }
        dst_state_dict = {
            "model.model.model.weights": fxt_random_tensor,
            "model.model.head.weights": fxt_random_tensor,
            "model.model.anchors": fxt_random_tensor,
        }
        converted_state_dict = OTXv1Helper.load_det_ckpt(src_state_dict, add_prefix="model.model.")
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)

    def test_load_iseg_ckpt(self, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "model.weights": fxt_random_tensor,
                    "head.weights": fxt_random_tensor,
                    "ema_model.weights": fxt_random_tensor,
                },
            },
        }

        dst_state_dict = {
            "model.model.model.weights": fxt_random_tensor,
            "model.model.head.weights": fxt_random_tensor,
        }

        converted_state_dict = OTXv1Helper.load_iseg_ckpt(src_state_dict, add_prefix="model.model.")
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)

    def test_load_seg_segnext_ckpt(self, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "model.weights": fxt_random_tensor,
                    "head.weights": fxt_random_tensor,
                    "ham.bases.weights": fxt_random_tensor,
                },
            },
        }

        dst_state_dict = {
            "model.model.model.weights": fxt_random_tensor,
            "model.model.head.weights": fxt_random_tensor,
        }

        converted_state_dict = OTXv1Helper.load_seg_segnext_ckpt(src_state_dict, add_prefix="model.model.")
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)

    def test_load_seg_lite_hrnet_ckpt(self, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "model.weights": fxt_random_tensor,
                    "head.weights": fxt_random_tensor,
                    "decode_head.aggregator.projects.weights": fxt_random_tensor,
                },
            },
        }

        dst_state_dict = {
            "model.model.model.weights": fxt_random_tensor,
            "model.model.head.weights": fxt_random_tensor,
        }

        converted_state_dict = OTXv1Helper.load_seg_lite_hrnet_ckpt(src_state_dict, add_prefix="model.model.")
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)

    def test_load_action_ckpt(self, fxt_random_tensor: torch.Tensor) -> None:
        src_state_dict = {
            "model": {
                "state_dict": {
                    "model.weights": fxt_random_tensor,
                    "head.weights": fxt_random_tensor,
                },
            },
        }

        dst_state_dict = {
            "model.model.model.weights": fxt_random_tensor,
            "model.model.head.weights": fxt_random_tensor,
        }

        converted_state_dict = OTXv1Helper.load_iseg_ckpt(src_state_dict, add_prefix="model.model.")
        self._check_ckpt_pairs(converted_state_dict, dst_state_dict)
