"""Tests for Utils for common OTX algorithms."""
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from otx.algorithms.common.utils.utils import embed_onnx_model_data

import onnx


class Meta:
    def __init__(self):
        self.key = None
        self.value = None


class MockModel:
    def __init__(self):
        self.metadata_props = self

    def add(self):
        return Meta()


@e2e_pytest_unit
def test_embed_onnx_model_data(mocker):
    mocker.patch.object(onnx, "load", return_value=MockModel())
    mocker.patch.object(onnx, "save")
    data = {(str("model_info"),): "info"}

    embed_onnx_model_data("", data)
