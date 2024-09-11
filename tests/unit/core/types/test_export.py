# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.core.config.data import TileConfig
from otx.core.types.export import TaskLevelExportParameters


@pytest.mark.parametrize("task_type", ["instance_segmentation", "classification"])
def test_wrap(fxt_label_info, task_type):
    params = TaskLevelExportParameters(
        model_type="dummy model",
        task_type=task_type,
        label_info=fxt_label_info,
        optimization_config={},
    )

    multilabel = False
    hierarchical = False
    output_raw_scores = True
    confidence_threshold = 0.0
    iou_threshold = 0.0
    return_soft_prediction = False
    soft_threshold = 0.0
    blur_strength = 0
    tile_config = TileConfig()

    params = params.wrap(
        multilabel=multilabel,
        hierarchical=hierarchical,
        output_raw_scores=output_raw_scores,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        return_soft_prediction=return_soft_prediction,
        soft_threshold=soft_threshold,
        blur_strength=blur_strength,
        tile_config=tile_config,
    )

    metadata = params.to_metadata()

    assert metadata[("model_info", "multilabel")] == str(multilabel)
    assert metadata[("model_info", "hierarchical")] == str(hierarchical)
    assert metadata[("model_info", "confidence_threshold")] == str(confidence_threshold)
    assert metadata[("model_info", "iou_threshold")] == str(iou_threshold)
    assert metadata[("model_info", "return_soft_prediction")] == str(return_soft_prediction)
    assert metadata[("model_info", "soft_threshold")] == str(soft_threshold)
    assert metadata[("model_info", "blur_strength")] == str(blur_strength)
    assert metadata[("model_info", "output_raw_scores")] == str(output_raw_scores)

    # Tile config
    assert ("model_info", "tile_size") in metadata
    assert ("model_info", "tiles_overlap") in metadata
    assert ("model_info", "max_pred_number") in metadata
