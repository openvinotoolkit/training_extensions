# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict
import torch
from mmdet.models.builder import build_detector

from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_deformable_detr_detector import (
    CustomDeformableDETR,
)
from otx.algorithms.detection.adapters.mmdet.models.detectors.custom_dino_detector import (
    CustomDINO,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomDINO:
    @e2e_pytest_unit
    def test_custom_dino_build(self, fxt_cfg_custom_dino: Dict):
        model = build_detector(fxt_cfg_custom_dino)
        assert isinstance(model, CustomDINO)

    @e2e_pytest_unit
    def test_custom_dino_load_state_pre_hook(self, fxt_cfg_custom_dino: Dict, mocker):
        mocker.patch.object(CustomDeformableDETR, "load_state_dict_pre_hook", return_value=True)
        model = build_detector(fxt_cfg_custom_dino)
        ckpt_dict = {
            "level_embed": "level_embed",
            "encoder.self_attn": "encoder.self_attn",
            "encoder.cross_attn": "encoder.cross_attn",
            "encoder.ffn": "encoder.ffn",
            "level_embed": "level_embed",
            "decoder.self_attn": "decoder.self_attn",
            "decoder.cross_attn": "decoder.cross_attn",
            "decoder.ffn": "decoder.ffn",
            "query_embedding.weight": "query_embedding.weight",
            "dn_query_generator.label_embedding.weight": "dn_query_generator.label_embedding.weight",
            "memory_trans_fc": "memory_trans_fc",
            "memory_trans_norm": "memory_trans_norm",
        }
        model.load_state_dict_pre_hook([], [], ckpt_dict)

        assert ckpt_dict["bbox_head.transformer.level_embeds"] == "level_embed"
        assert ckpt_dict["bbox_head.transformer.encoder.attentions.0"] == "encoder.self_attn"
        assert ckpt_dict["bbox_head.transformer.encoder.attentions.1"] == "encoder.cross_attn"
        assert ckpt_dict["bbox_head.transformer.encoder.ffns.0"] == "encoder.ffn"
        assert ckpt_dict["bbox_head.transformer.decoder.attentions.0"] == "decoder.self_attn"
        assert ckpt_dict["bbox_head.transformer.decoder.attentions.1"] == "decoder.cross_attn"
        assert ckpt_dict["bbox_head.transformer.decoder.ffns.0"] == "decoder.ffn"
        assert ckpt_dict["bbox_head.query_embedding.weight"] == "query_embedding.weight"
        assert (
            ckpt_dict["bbox_head.transformer.dn_query_generator.label_embedding.weight"]
            == "dn_query_generator.label_embedding.weight"
        )
        assert ckpt_dict["bbox_head.transformer.enc_output"] == "memory_trans_fc"
        assert ckpt_dict["bbox_head.transformer.enc_output_norm"] == "memory_trans_norm"
