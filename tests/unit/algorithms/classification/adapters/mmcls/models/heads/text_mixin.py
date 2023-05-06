import pytest

from otx.algorithms.classification.adapters.mmcls.models.heads.mixin import OTXHeadMixin
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOTXHeadMixin:
    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "input,expected_output",
        [
            ([1, 2, 3], 3),
            ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
            (42, 42),
            ("test_string", "test_string"),
        ],
    )
    def test_pre_logits(self, input, expected_output):
        """Verifies pre-logits logic for list output from ViT backbones."""
        output = OTXHeadMixin.pre_logits(input)
        assert output == expected_output, f"Expected {expected_output}, but got {output}"
