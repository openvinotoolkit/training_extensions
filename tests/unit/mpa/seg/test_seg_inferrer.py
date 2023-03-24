import os

import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa.seg.inferrer import SegInferrer, replace_ImageToTensor
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_RECIPE_CONFIG_PATH,
    DEFAULT_SEG_TEMPLATE_DIR,
)


class TestOTXSegTrainer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_RECIPE_CONFIG_PATH)
        self.inferrer = SegInferrer(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_run(self, mocker):
        fake_output = {"classes": [1, 2], "eval_predictions": None, "feature_vectors": None}
        mock_infer = mocker.patch.object(SegInferrer, "infer", return_value=fake_output)

        returned_value = self.inferrer.run(self.model_cfg, "", self.data_cfg)
        mock_infer.assert_called_once()
        assert returned_value == {"outputs": fake_output}

    @e2e_pytest_unit
    def test_infer(self, mocker):
        cfg = self.inferrer.configure(self.model_cfg, "", self.data_cfg, training=False)
        mocker.patch.object(SegInferrer, "configure_samples_per_gpu")
        mocker.patch.object(SegInferrer, "configure_compat_cfg")
        mock_infer_callback = mocker.patch.object(SegInferrer, "set_inference_progress_callback")

        returned_value = self.inferrer.infer(cfg)
        mock_infer_callback.assert_called_once()

        assert "classes" in returned_value
        assert "eval_predictions" in returned_value
        assert "feature_vectors" in returned_value
        assert len(returned_value["eval_predictions"]) >= 0


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
