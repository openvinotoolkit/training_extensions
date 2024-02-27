# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from otx.core.data.entity.classification import MulticlassClsBatchPredEntity, MulticlassClsBatchPredEntityWithXAI
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
