# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
from otx.algo.utils.xai_utils import process_saliency_maps, process_saliency_maps_in_pred_entity
from otx.core.config.explain import ExplainConfig
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import MulticlassClsBatchPredEntity, MultilabelClsBatchPredEntity
from otx.core.types.explain import TargetExplainGroup

NUM_CLASSES = 5
BATCH_SIZE = 3
RAW_SIZE = 7
OUT_SIZE = 224

PRED_LABELS = [[0, 2, 3], [1], []]
PRED_LABELS_TOP_ONE = [[1], [0], [4]]
SALIENCY_MAPS = [np.ones((NUM_CLASSES, RAW_SIZE, RAW_SIZE), dtype=np.uint8) for _ in range(BATCH_SIZE)]
SALIENCY_MAPS_IMAGE = [np.ones((RAW_SIZE, RAW_SIZE), dtype=np.uint8) for _ in range(BATCH_SIZE)]
ORI_IMG_SHAPES = [(OUT_SIZE, OUT_SIZE)] * BATCH_SIZE
IMGS_INFO = [ImageInfo(img_idx=i, img_shape=None, ori_shape=(OUT_SIZE, OUT_SIZE)) for i in range(BATCH_SIZE)]


@pytest.mark.parametrize("postprocess", [False, True])
def test_process_all(postprocess) -> None:
    explain_config = ExplainConfig(target_explain_group=TargetExplainGroup.ALL, postprocess=postprocess)

    with pytest.raises(ValueError, match="Shape mismatch."):
        processed_saliency_maps = process_saliency_maps(
            SALIENCY_MAPS_IMAGE,
            explain_config,
            PRED_LABELS,
            ORI_IMG_SHAPES,
        )

    processed_saliency_maps = process_saliency_maps(SALIENCY_MAPS, explain_config, PRED_LABELS, ORI_IMG_SHAPES)

    assert len(processed_saliency_maps) == BATCH_SIZE
    assert all(len(s_map_dict) == NUM_CLASSES for s_map_dict in processed_saliency_maps)

    if postprocess:
        assert all(
            next(iter(s_map_dict.values())).shape == (OUT_SIZE, OUT_SIZE, 3) for s_map_dict in processed_saliency_maps
        )
    else:
        assert all(
            next(iter(s_map_dict.values())).shape == (RAW_SIZE, RAW_SIZE) for s_map_dict in processed_saliency_maps
        )


@pytest.mark.parametrize("postprocess", [False, True])
def test_process_predictions(postprocess) -> None:
    explain_config = ExplainConfig(target_explain_group=TargetExplainGroup.PREDICTIONS, postprocess=postprocess)

    with pytest.raises(ValueError, match="Shape mismatch."):
        processed_saliency_maps = process_saliency_maps(
            SALIENCY_MAPS_IMAGE,
            explain_config,
            PRED_LABELS,
            ORI_IMG_SHAPES,
        )

    processed_saliency_maps = process_saliency_maps(SALIENCY_MAPS, explain_config, PRED_LABELS, ORI_IMG_SHAPES)

    assert len(processed_saliency_maps) == BATCH_SIZE
    assert all(len(s_map_dict) == len(PRED_LABELS[i]) for (i, s_map_dict) in enumerate(processed_saliency_maps))

    if postprocess:
        assert all(
            next(iter(s_map_dict.values())).shape == (OUT_SIZE, OUT_SIZE, 3)
            for s_map_dict in processed_saliency_maps
            if s_map_dict
        )
    else:
        assert all(
            next(iter(s_map_dict.values())).shape == (RAW_SIZE, RAW_SIZE)
            for s_map_dict in processed_saliency_maps
            if s_map_dict
        )


@pytest.mark.parametrize("postprocess", [False, True])
def test_process_image(postprocess) -> None:
    explain_config = ExplainConfig(target_explain_group=TargetExplainGroup.IMAGE, postprocess=postprocess)

    with pytest.raises(ValueError, match="Shape mismatch."):
        processed_saliency_maps = process_saliency_maps(SALIENCY_MAPS, explain_config, PRED_LABELS, ORI_IMG_SHAPES)

    processed_saliency_maps = process_saliency_maps(SALIENCY_MAPS_IMAGE, explain_config, PRED_LABELS, ORI_IMG_SHAPES)

    assert len(processed_saliency_maps) == BATCH_SIZE
    assert all(len(s_map_dict) == 1 for s_map_dict in processed_saliency_maps)

    if postprocess:
        assert all(
            s_map_dict["map_per_image"].shape == (OUT_SIZE, OUT_SIZE, 3) for s_map_dict in processed_saliency_maps
        )
    else:
        assert all(s_map_dict["map_per_image"].shape == (RAW_SIZE, RAW_SIZE) for s_map_dict in processed_saliency_maps)


def _get_pred_result_multiclass(pred_labels) -> MulticlassClsBatchPredEntity:
    return MulticlassClsBatchPredEntity(
        batch_size=BATCH_SIZE,
        images=None,
        imgs_info=IMGS_INFO,
        scores=None,
        labels=pred_labels,
        saliency_map=SALIENCY_MAPS,
        feature_vector=None,
    )


def _get_pred_result_multilabel(pred_labels) -> MultilabelClsBatchPredEntity:
    return MultilabelClsBatchPredEntity(
        batch_size=BATCH_SIZE,
        images=None,
        imgs_info=IMGS_INFO,
        scores=None,
        labels=pred_labels,
        saliency_map=SALIENCY_MAPS,
        feature_vector=None,
    )


def test_process_saliency_maps_in_pred_entity_multiclass() -> None:
    explain_config = ExplainConfig(target_explain_group=TargetExplainGroup.PREDICTIONS)

    pred_labels = [torch.tensor(labels) for labels in PRED_LABELS_TOP_ONE]
    predict_result_batch1 = _get_pred_result_multiclass(pred_labels)
    predict_result_batch2 = _get_pred_result_multiclass(pred_labels)

    predict_result = process_saliency_maps_in_pred_entity(
        [predict_result_batch1, predict_result_batch2],
        explain_config,
    )

    for i in range(len(predict_result)):
        assert isinstance(predict_result[i].saliency_map, list)
        assert isinstance(predict_result[i].saliency_map[0], dict)
        processed_saliency_maps = predict_result[i].saliency_map
        assert all(len(s_map_dict) == 1 for s_map_dict in processed_saliency_maps)


def test_process_saliency_maps_in_pred_entity_multilabel() -> None:
    explain_config = ExplainConfig(target_explain_group=TargetExplainGroup.PREDICTIONS)

    pred_labels = [torch.tensor(labels) for labels in PRED_LABELS]
    predict_result_batch1 = _get_pred_result_multilabel(pred_labels)
    predict_result_batch2 = _get_pred_result_multilabel(pred_labels)

    predict_result = process_saliency_maps_in_pred_entity(
        [predict_result_batch1, predict_result_batch2],
        explain_config,
    )

    for i in range(len(predict_result)):
        assert isinstance(predict_result[i].saliency_map, list)
        assert isinstance(predict_result[i].saliency_map[0], dict)
        processed_saliency_maps = predict_result[i].saliency_map
        assert all(len(s_map_dict) == len(PRED_LABELS[i]) for (i, s_map_dict) in enumerate(processed_saliency_maps))
