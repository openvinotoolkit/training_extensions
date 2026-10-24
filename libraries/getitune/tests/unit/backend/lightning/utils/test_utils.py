# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from getitune.backend.lightning.utils.utils import (
    is_ckpt_for_finetuning,
    remove_state_dict_prefix,
)


def test_is_ckpt_for_finetuning():
    ckpt = {"state_dict": {"param1": 1, "param2": 2}}
    assert is_ckpt_for_finetuning(ckpt)

    ckpt = {"other_key": "value"}
    assert not is_ckpt_for_finetuning(ckpt)

    ckpt = {}
    assert not is_ckpt_for_finetuning(ckpt)


def test_remove_state_dict_prefix():
    state_dict = {
        "model._orig_mod.backbone.0.weight": 1,
        "model._orig_mod.backbone.0.bias": 2,
        "model._orig_mod.backbone.1.weight": 3,
        "model._orig_mod.backbone.1.bias": 4,
    }
    new_state_dict = remove_state_dict_prefix(state_dict=state_dict, prefix="_orig_mod.")
    expected = {
        "model.backbone.0.weight": 1,
        "model.backbone.0.bias": 2,
        "model.backbone.1.weight": 3,
        "model.backbone.1.bias": 4,
    }
    assert new_state_dict == expected
