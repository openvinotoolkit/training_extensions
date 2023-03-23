import os
import tempfile

import cv2
import numpy as np
import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa.det.inferrer import DetectionInferrer, replace_ImageToTensor
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_RECIPE_CONFIG_PATH,
    DEFAULT_DET_TEMPLATE_DIR,
    create_dummy_coco_json,
)


class TestDetectionInferrer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_DET_RECIPE_CONFIG_PATH)
        self.inferrer = DetectionInferrer(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_run_train_mode(self, mocker):
        fake_output = {"classes": [1, 2], "eval_predictions": None, "feature_vectors": None}
        mock_infer = mocker.patch.object(DetectionInferrer, "infer", return_value=fake_output)
        returned_value = self.inferrer.run(self.model_cfg, "", self.data_cfg, mode="train")
        mock_infer.assert_called_once()
        assert returned_value == {"outputs": fake_output}

    @e2e_pytest_unit
    def test_run_test_mode(self, mocker):
        fake_output = {"classes": [1, 2], "eval_predictions": None, "feature_vectors": None}
        mock_infer = mocker.patch.object(DetectionInferrer, "infer", return_value=fake_output)
        self.inferrer.run(self.model_cfg, "", self.data_cfg, mode="test")
        mock_infer.assert_not_called()

    @e2e_pytest_unit
    @pytest.mark.parametrize("input_source", ["train", "test"])
    @pytest.mark.parametrize("dump_saliency_map", [True, False])
    def test_infer(self, input_source, dump_saliency_map, mocker):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_json_path = os.path.join(tmp_dir, "fake_data.json")
            create_dummy_coco_json(fake_json_path)

            self.data_cfg.data.train.ann_file = fake_json_path
            self.data_cfg.data.val.ann_file = fake_json_path
            self.data_cfg.data.test.ann_file = fake_json_path

            if input_source == "train":
                self.data_cfg.data.train.img_prefix = tmp_dir
                self.data_cfg.data.val.img_prefix = tmp_dir
            elif input_source == "test":
                self.data_cfg.data.test.img_prefix = tmp_dir

            self.data_cfg.data.train.data_classes = ["red", "green"]
            self.data_cfg.data.val_dataloader = dict()

            arr = np.ones((10, 10), dtype=np.uint8)
            cv2.imwrite(os.path.join(tmp_dir, "fake_name.jpg"), arr)
            cfg = self.inferrer.configure(self.model_cfg, "", self.data_cfg, training=False)
            cfg.input_source = input_source

            mocker.patch.object(DetectionInferrer, "configure_samples_per_gpu")
            mocker.patch.object(DetectionInferrer, "configure_compat_cfg")
            mock_infer_callback = mocker.patch.object(DetectionInferrer, "set_inference_progress_callback")
            returned_value = self.inferrer.infer(cfg, dump_saliency_map=dump_saliency_map)
            mock_infer_callback.assert_called_once()

        assert "classes" in returned_value
        assert "detections" in returned_value
        assert "metric" in returned_value
        assert "feature_vectors" in returned_value
        assert "saliency_maps" in returned_value
        assert len(returned_value["detections"]) >= 0


@e2e_pytest_unit
def test_replace_ImageToTensor():
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(
            type="MultiScaleFlipAug",
            transforms=[
                dict(type="Resize", keep_ratio=False),
                dict(type="ImageToTensor", keys=["img"]),
            ],
        ),
        dict(type="ImageToTensor", keys=["img"]),
    ]
    returned_value = replace_ImageToTensor(test_pipeline)

    for pipeline in returned_value:
        if "transforms" in pipeline:
            values = [p["type"] for p in pipeline["transforms"]]
        else:
            values = [pipeline["type"]]
        assert "ImageToTensor" not in values
