# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import openvino.runtime as ov
import pytest
from otx.core.data.entity.base import OTXBatchPredEntity
from otx.engine import Engine

RECIPE_LIST_ALL = pytest.RECIPE_LIST
MULTI_CLASS_CLS = [recipe for recipe in RECIPE_LIST_ALL if "multi_class_cls" in recipe]
MULTI_LABEL_CLS = [recipe for recipe in RECIPE_LIST_ALL if "multi_label_cls" in recipe]
MC_ML_CLS = MULTI_CLASS_CLS + MULTI_LABEL_CLS

DETECTION_LIST = [recipe for recipe in RECIPE_LIST_ALL if "/detection" in recipe and "tile" not in recipe]
INST_SEG_LIST = [recipe for recipe in RECIPE_LIST_ALL if "instance_segmentation" in recipe and "tile" not in recipe]
EXPLAIN_MODEL_LIST = MC_ML_CLS + DETECTION_LIST + INST_SEG_LIST

MEAN_TORCH_OV_DIFF = 150


@pytest.mark.parametrize(
    "recipe",
    EXPLAIN_MODEL_LIST,
)
def test_forward_explain(
    recipe: str,
    fxt_target_dataset_per_task: dict,
    fxt_accelerator: str,
) -> None:
    """
    Test forward == forward_explain.

    Args:
        recipe (str): The recipe to use for predicting. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        fxt_target_dataset_per_task (dict): A dictionary mapping tasks to target datasets.
        fxt_accelerator (str): The accelerator used for predict.

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
        device=fxt_accelerator,
    )

    predict_result = engine.predict()
    assert isinstance(predict_result[0], OTXBatchPredEntity)
    assert not predict_result[0].has_xai_outputs

    predict_result_explain = engine.predict(explain=True)
    assert isinstance(predict_result_explain[0], OTXBatchPredEntity)
    assert predict_result_explain[0].has_xai_outputs

    batch_size = len(predict_result[0].scores)
    for i in range(batch_size):
        assert all(predict_result[0].labels[i] == predict_result_explain[0].labels[i])
        assert all(predict_result[0].scores[i] == predict_result_explain[0].scores[i])


@pytest.mark.parametrize(
    "recipe",
    EXPLAIN_MODEL_LIST,
)
def test_predict_with_explain(
    recipe: str,
    tmp_path: Path,
    fxt_target_dataset_per_task: dict,
    fxt_accelerator: str,
) -> None:
    """
    Test XAI.

    Args:
        recipe (str): The recipe to use for predicting. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the outputs.
        fxt_target_dataset_per_task (dict): A dictionary mapping tasks to target datasets.
        fxt_accelerator (str): The accelerator used for predict.

    Returns:
        None
    """
    task = recipe.split("/")[-2]
    model_name = recipe.split("/")[-1].split(".")[0]

    if "dino" in model_name:
        pytest.skip("DINO is not supported.")

    if "ssd_mobilenetv2" in model_name:
        pytest.skip("There's issue with SSD model. Skip for now.")

    if "atss" in model_name or "yolox" in model_name:
        # TODO(Jaeguk, sungchul): ATSS and YOLOX returns dynamic output for saliency map
        pytest.skip(f"There's issue with {model_name} model. Skip for now.")

    if "instance_segmentation" in recipe:
        # TODO(Eugene): figure out why instance segmentation model fails after decoupling.
        pytest.skip("There's issue with instance segmentation model. Skip for now.")

    tmp_path = tmp_path / f"otx_xai_{model_name}"
    engine = Engine.from_config(
        config_path=recipe,
        data_root=fxt_target_dataset_per_task[task],
        device=fxt_accelerator,
        work_dir=tmp_path,
    )

    # Predict with explain torch & process maps
    predict_result_explain_torch = engine.predict(explain=True)
    assert isinstance(predict_result_explain_torch[0], OTXBatchPredEntity)
    assert predict_result_explain_torch[0].has_xai_outputs
    assert predict_result_explain_torch[0].saliency_map is not None
    assert isinstance(predict_result_explain_torch[0].saliency_map[0], dict)

    # Export with explain
    ckpt_path = tmp_path / "checkpoint.ckpt"
    engine.trainer.save_checkpoint(ckpt_path)
    exported_model_path = engine.export(checkpoint=ckpt_path, explain=True)

    model = ov.Core().read_model(exported_model_path)
    feature_vector_output = None
    saliency_map_output = None
    for output in model.outputs:
        if "feature_vector" in output.get_names():
            feature_vector_output = output
        if "saliency_map" in output.get_names():
            saliency_map_output = output
    assert saliency_map_output is not None
    if "instance_segmentation" in recipe:
        assert len(saliency_map_output.get_shape()) == 1
    else:
        assert len(saliency_map_output.get_shape()) in [3, 4]

    assert feature_vector_output is not None
    assert len(feature_vector_output.get_shape()) == 2

    # Predict OV model with xai & process maps
    predict_result_explain_ov = engine.predict(checkpoint=exported_model_path, explain=True)
    assert isinstance(predict_result_explain_ov[0], OTXBatchPredEntity)
    assert predict_result_explain_ov[0].has_xai_outputs
    assert predict_result_explain_ov[0].saliency_map is not None
    assert isinstance(predict_result_explain_ov[0].saliency_map[0], dict)
    assert predict_result_explain_ov[0].feature_vector is not None
    assert isinstance(predict_result_explain_ov[0].feature_vector[0], np.ndarray)

    if task == "instance_segmentation" or "atss_r50_fpn" in recipe:
        # For instance segmentation and atss_r50_fpn batch_size for Torch task 1, for OV 2.
        # That why the predict_results have different format and we can't compare them.

        # The OV saliency maps are different from Torch and incorrect, possible root cause can be on MAPI side
        # TODO(gzalessk): remove this if statement when the issue is resolved
        return

    maps_torch = predict_result_explain_torch[0].saliency_map
    maps_ov = predict_result_explain_ov[0].saliency_map

    assert len(maps_torch) == len(maps_ov)

    if "tv_efficientnet_b3" in recipe:
        # There is the issue with different predict results for Pytorch and OpenVINO tasks.
        # Probably because of the different preprocessed images passed as an input. Skip the rest of the checks for now.
        return

    for i in range(len(maps_torch)):
        for class_id in maps_torch[i]:
            assert class_id in maps_ov[i]
            assert (
                np.mean(abs(maps_torch[i][class_id].astype(np.float32) - maps_ov[i][class_id].astype(np.float32)))
                < MEAN_TORCH_OV_DIFF
            )
