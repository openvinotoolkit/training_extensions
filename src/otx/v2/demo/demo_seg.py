# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from otx.v2.adapters.torch.mmengine.mmseg import Engine, get_model, list_models

from otx.v2.adapters.torch.mmengine.mmseg.dataset import Dataset

from tests.v2.integration.test_helper import TASK_CONFIGURATION


dataset = Dataset(
    train_data_roots=TASK_CONFIGURATION["segmentation"]["train_data_roots"],
    val_data_roots=TASK_CONFIGURATION["segmentation"]["val_data_roots"],
    test_data_roots=TASK_CONFIGURATION["segmentation"]["test_data_roots"],
)

tmp_dir_path = "tmp_dir_path"
model = "ocr_lite_hrnet_s_mod2"

# Setup Engine
engine = Engine(work_dir=tmp_dir_path)
built_model = get_model(model=model, num_classes=dataset.num_classes)

# Train (1 epochs)
results = engine.train(
    model=built_model,
    train_dataloader=dataset.train_dataloader(),
    val_dataloader=dataset.val_dataloader(),
    max_epochs=1,
)
assert "model" in results
assert "checkpoint" in results
assert isinstance(results["model"], torch.nn.Module)
assert isinstance(results["checkpoint"], str)
assert Path(results["checkpoint"]).exists()

# Validation
val_score = engine.validate()
assert "mDice" in val_score
assert val_score["mDice"] > 0.0

# Test
test_score = engine.test(test_dataloader=dataset.test_dataloader())
assert "mDice" in test_score
assert test_score["mDice"] > 0.0

# Prediction with single image
pred_result = engine.predict(
    model=results["model"],
    checkpoint=results["checkpoint"],
    img=TASK_CONFIGURATION["segmentation"]["sample"],
)
assert isinstance(pred_result, list)
assert len(pred_result) == 1
assert "num_classes" in pred_result[0]
assert pred_result[0].num_classes == dataset.num_classes

# Export Openvino IR Model
export_output = engine.export(
    model=results["model"],
    checkpoint=results["checkpoint"],
)
assert isinstance(export_output, dict)
assert "outputs" in export_output
assert isinstance(export_output["outputs"], dict)
assert "bin" in export_output["outputs"]
assert "xml" in export_output["outputs"]
assert Path(export_output["outputs"]["bin"]).exists()
assert Path(export_output["outputs"]["xml"]).exists()
