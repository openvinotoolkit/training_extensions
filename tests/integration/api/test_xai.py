# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import openvino.runtime as ov
import pytest
from otx.core.data.entity.classification import (
    MulticlassClsBatchPredEntity,
    MulticlassClsBatchPredEntityWithXAI,
    MultilabelClsBatchPredEntityWithXAI,
)
from otx.engine import Engine

RECIPE_LIST_ALL = pytest.RECIPE_LIST
MULTI_CLASS_CLS = [recipe for recipe in RECIPE_LIST_ALL if "multi_class_cls" in recipe]
MULTI_LABEL_CLS = [recipe for recipe in RECIPE_LIST_ALL if "multi_label_cls" in recipe]
MC_ML_CLS = MULTI_CLASS_CLS + MULTI_LABEL_CLS


@pytest.mark.parametrize(
    "recipe",
    MULTI_CLASS_CLS,
    ids=lambda x: "/".join(Path(x).parts[-2:]),
)
def test_forward_explain(
    recipe: str,
    fxt_target_dataset_per_task: dict,
) -> None:
    """
    Test forward == forward_explain.

    Args:
        recipe (str): The recipe to use for predicting. (eg. 'classification/otx_mobilenet_v3_large.yaml')

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if "dino" in model_name:
        pytest.skip("DINO is not supported.")

    engine = Engine.from_config(
        config_path=recipe,
        data_root=fxt_target_dataset_per_task[task],
    )

    predict_result = engine.predict()
    assert isinstance(predict_result[0], MulticlassClsBatchPredEntity)

    predict_result_explain = engine.predict(explain=True)
    assert isinstance(predict_result_explain[0], MulticlassClsBatchPredEntityWithXAI)

    for i in range(len(predict_result[0].scores)):
        assert all(predict_result[0].labels[i] == predict_result_explain[0].labels[i])
        assert all(predict_result[0].scores[i] == predict_result_explain[0].scores[i])


@pytest.mark.parametrize(
    "recipe",
    MC_ML_CLS,
    ids=lambda x: "/".join(Path(x).parts[-2:]),
)
def test_predict_with_explain(
    recipe: str,
    tmp_path: Path,
    fxt_target_dataset_per_task: dict,
) -> None:
    """
    Test XAI.

    Args:
        recipe (str): The recipe to use for predicting. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if "mobilenet_v3_large_light" in model_name:
        pytest.skip("Dataloader failure during train.")
    if "dino" in model_name:
        pytest.skip("DINO is not supported.")

    tmp_path = tmp_path / f"otx_xai_{model_name}"
    engine = Engine.from_config(
        config_path=recipe,
        data_root=fxt_target_dataset_per_task[task],
        work_dir=tmp_path,
    )

    # Train
    engine.train(
        max_epochs=2,
        seed=0,
        deterministic=True,
    )

    # Predict with explain torch
    predict_result_explain_torch = engine.predict(explain=True)
    assert isinstance(
        predict_result_explain_torch[0],
        (MulticlassClsBatchPredEntityWithXAI, MultilabelClsBatchPredEntityWithXAI),
    )
    assert predict_result_explain_torch[0].saliency_maps is not None
    assert isinstance(predict_result_explain_torch[0].saliency_maps[0], dict)

    # Export with explain
    exported_model_path = engine.export(explain=True)

    model = ov.Core().read_model(exported_model_path)
    saliency_map_output = None
    for output in model.outputs:
        if "saliency_map" in output.get_names():
            saliency_map_output = output
            break
    assert saliency_map_output is not None
    assert len(saliency_map_output.get_shape()) in [3, 4]

    # Predict OV model with xai
    predict_result_explain_ov = engine.predict(checkpoint=exported_model_path, explain=True)
    assert isinstance(
        predict_result_explain_ov[0],
        (MulticlassClsBatchPredEntityWithXAI, MultilabelClsBatchPredEntityWithXAI),
    )
    assert predict_result_explain_ov[0].saliency_maps is not None
    assert isinstance(predict_result_explain_ov[0].saliency_maps[0], dict)

    maps_torch = predict_result_explain_torch[0].saliency_maps
    maps_ov = predict_result_explain_ov[0].saliency_maps

    assert len(maps_torch) == len(maps_ov)

    for i in range(len(maps_torch)):
        class_id = 0
        for class_id in maps_torch[i]:
            assert class_id in maps_ov[i]
            assert (
                np.mean(abs(maps_torch[i][class_id].astype(np.float32) - maps_ov[i][class_id].astype(np.float32))) < 150
            )
