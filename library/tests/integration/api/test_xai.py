# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
from openvino import Core

from getitune.backend.lightning.engine import LightningEngine
from getitune.data.entity.sample import PredictionBatch
from getitune.engine import create_engine

RECIPE_LIST_ALL = pytest.RECIPE_LIST
MULTI_CLASS_CLS = [recipe for recipe in RECIPE_LIST_ALL if "multi_class_cls" in recipe]
MULTI_LABEL_CLS = [recipe for recipe in RECIPE_LIST_ALL if "multi_label_cls" in recipe]
MC_ML_CLS = MULTI_CLASS_CLS + MULTI_LABEL_CLS

DETECTION_LIST = [recipe for recipe in RECIPE_LIST_ALL if "/detection" in recipe]
INST_SEG_LIST = [recipe for recipe in RECIPE_LIST_ALL if "instance_segmentation" in recipe]
EXPLAIN_MODEL_LIST = MC_ML_CLS + DETECTION_LIST + INST_SEG_LIST

MEAN_TORCH_OV_DIFF = 150
UNSUPPORTED_MODEL_SUBSTRS = ("dino", "rfdetr")


@pytest.mark.parametrize(
    "recipe",
    EXPLAIN_MODEL_LIST,
)
def test_forward_explain(
    recipe: str,
    tmp_path: Path,
    fxt_target_dataset_per_task: dict,
    fxt_accelerator: str,
) -> None:
    """
    Test forward == forward_explain.

    Args:
        recipe (str): The recipe to use for predicting. (eg. 'classification/mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the outputs.
        fxt_target_dataset_per_task (dict): A dictionary mapping tasks to target datasets.
        fxt_accelerator (str): The accelerator used for predict.

    Returns:
        None
    """

    recipe_split = recipe.split("/")
    model_name = recipe_split[-1].split(".")[0]
    task = recipe_split[-2]

    if any(sub in model_name for sub in UNSUPPORTED_MODEL_SUBSTRS):
        pytest.skip(f"{model_name} is not supported.")

    engine = LightningEngine.from_config(
        config_path=recipe,
        data=fxt_target_dataset_per_task[task],
        device=fxt_accelerator,
        work_dir=tmp_path,
    )

    predict_result = engine.predict()
    assert isinstance(predict_result[0], PredictionBatch)
    assert predict_result[0].saliency_map is None or len(predict_result[0].saliency_map) == 0

    predict_result_explain = engine.predict(explain=True)
    assert isinstance(predict_result_explain[0], PredictionBatch)
    assert predict_result_explain[0].saliency_map is not None
    assert len(predict_result_explain[0].saliency_map) > 0

    plain_scores = predict_result[0].scores
    explain_scores = predict_result_explain[0].scores
    plain_labels = predict_result[0].labels
    explain_labels = predict_result_explain[0].labels
    assert plain_scores is not None
    assert explain_scores is not None
    assert plain_labels is not None
    assert explain_labels is not None
    for plain_batch_scores, explain_batch_scores, plain_batch_labels, explain_batch_labels in zip(
        plain_scores, explain_scores, plain_labels, explain_labels
    ):
        # The explain run recomputes the forward pass in a separate GPU call,
        # which can differ from the plain run by tiny rounding noise. With a
        # randomly initialized head this can flip the argmax when the top-2
        # scores are essentially tied, so always compare scores approximately
        # and require label agreement only when the top-2 gap is clear.
        assert torch.allclose(plain_batch_scores, explain_batch_scores, rtol=1e-4, atol=1e-5)
        sorted_scores = torch.sort(plain_batch_scores, descending=True).values
        if sorted_scores[0] - sorted_scores[1] > 2e-4:
            assert (plain_batch_labels == explain_batch_labels).all()


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
        recipe (str): The recipe to use for predicting. (eg. 'classification/mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the outputs.
        fxt_target_dataset_per_task (dict): A dictionary mapping tasks to target datasets.
        fxt_accelerator (str): The accelerator used for predict.

    Returns:
        None
    """
    recipe_split = recipe.split("/")
    model_name = recipe_split[-1].split(".")[0]
    task = recipe_split[-2]

    if any(sub in model_name for sub in UNSUPPORTED_MODEL_SUBSTRS):
        pytest.skip(f"{model_name} is not supported.")

    tmp_path = tmp_path / f"xai_{model_name}"
    engine = LightningEngine.from_config(
        config_path=recipe,
        data=fxt_target_dataset_per_task[task],
        device=fxt_accelerator,
        work_dir=tmp_path,
    )

    # Predict with explain torch & process maps
    predict_result_explain_torch = engine.predict(explain=True)
    assert isinstance(predict_result_explain_torch[0], PredictionBatch)
    assert predict_result_explain_torch[0].saliency_map is not None
    assert len(predict_result_explain_torch[0].saliency_map) > 0
    assert predict_result_explain_torch[0].saliency_map is not None
    assert isinstance(predict_result_explain_torch[0].saliency_map[0], dict)

    # Export with explain
    ckpt_path = tmp_path / "checkpoint.ckpt"
    engine.trainer.save_checkpoint(ckpt_path)
    exported_model_path = engine.export(checkpoint=ckpt_path, explain=True)

    model = Core().read_model(exported_model_path)
    feature_vector_output = None
    saliency_map_output = None
    for output in model.outputs:
        if "feature_vector" in output.get_names():
            feature_vector_output = output
        if "saliency_map" in output.get_names():
            saliency_map_output = output
    assert saliency_map_output is not None
    saliency_map_output_rank = saliency_map_output.get_partial_shape().rank.get_length()
    if "instance_segmentation" in recipe:
        assert saliency_map_output_rank == 1
    else:
        assert saliency_map_output_rank in [3, 4]

    assert feature_vector_output is not None
    assert feature_vector_output.get_partial_shape().rank.get_length() == 2

    # Predict OV model with xai & process maps
    ov_engine = create_engine(model=exported_model_path, data=engine.datamodule, work_dir=engine.work_dir)
    predict_result_explain_ov = ov_engine.predict(checkpoint=exported_model_path, explain=True)
    assert isinstance(predict_result_explain_ov[0], PredictionBatch)
    assert predict_result_explain_ov[0].saliency_map is not None
    assert len(predict_result_explain_ov[0].saliency_map) > 0
    assert predict_result_explain_ov[0].saliency_map is not None
    assert isinstance(predict_result_explain_ov[0].saliency_map[0], dict)
    assert predict_result_explain_ov[0].feature_vector is not None
    assert isinstance(predict_result_explain_ov[0].feature_vector[0], np.ndarray)

    if task == "instance_segmentation":
        # For instance segmentation batch_size for Torch task 1, for OV 2.
        # That why the predict_results have different format and we can't compare them.

        # The OV saliency maps are different from Torch and incorrect, possible root cause can be on MAPI side
        # TODO(gzalessk): remove this if statement when the issue is resolved
        return

    maps_torch = predict_result_explain_torch[0].saliency_map
    maps_ov = predict_result_explain_ov[0].saliency_map

    assert len(maps_torch) == len(maps_ov)

    if "efficientnet_b3" in recipe or "efficientnet_b0" in recipe or "vit_tiny" in recipe:
        # There is the issue with different predict results for Pytorch and OpenVINO tasks.
        # Probably because of the different preprocessed images passed as an input. Skip the rest of the checks for now.
        # Tickets: 142087, 141639
        return

    if "yolox" in recipe:
        # The cropping of the padded saliency maps is not implemented for OV (Model API) yet,
        # so the saliency maps for PyTorch and OV are different.
        # TODO(gzalessk): Implement cropping saliency maps in Model API (Ticket 144296).
        return

    for i in range(len(maps_torch)):
        for class_id in maps_torch[i]:
            assert class_id in maps_ov[i]
            assert (
                np.mean(abs(maps_torch[i][class_id].astype(np.float32) - maps_ov[i][class_id].astype(np.float32)))
                < MEAN_TORCH_OV_DIFF
            )
