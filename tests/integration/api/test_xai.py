# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

import openvino.runtime as ov

from otx.core.data.entity.classification import MulticlassClsBatchPredEntity, MulticlassClsBatchPredEntityWithXAI, MultilabelClsBatchPredEntityWithXAI
from otx.engine import Engine


@pytest.mark.parametrize(
    "recipe",
    pytest.RECIPE_LIST,
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

    if "_cls" not in task:
        pytest.skip("Supported only for classification.")
    if "multi_class_cls" not in task:
        pytest.skip("Required only for multiclass classification (not required for multilabel and h-label).")
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
    pytest.RECIPE_LIST,
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

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if "_cls" not in task:
        pytest.skip("Supported only for classification.")
    if "h_label_cls" in task:
        pytest.skip("H-label is not supported.")
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
    assert isinstance(predict_result_explain_torch[0], (MulticlassClsBatchPredEntityWithXAI, MultilabelClsBatchPredEntityWithXAI))

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

    # Predict OV of the model with xai
    predict_result_explain_ov = engine.predict(checkpoint=exported_model_path, explain=True)
    assert isinstance(predict_result_explain_ov[0], (MulticlassClsBatchPredEntityWithXAI, MultilabelClsBatchPredEntityWithXAI))

    assert len(predict_result_explain_torch[0].saliency_maps) == len(predict_result_explain_ov[0].saliency_maps)

    import numpy as np
    res_torch = predict_result_explain_torch[0]
    res_ov = predict_result_explain_ov[0]
    for i in range(len(predict_result_explain_torch[0].saliency_maps)):
        class_id = 0
        print(np.max(abs(res_torch.saliency_maps[i][class_id].astype(np.float32) - res_ov.saliency_maps[i][class_id].astype(np.float32))))
