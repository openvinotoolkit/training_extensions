# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch

from otx.core.ov.ops.type_conversions import ConvertV0
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestConvertV0:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)

    @e2e_pytest_unit
    def test_convert_ov_type(self):
        with pytest.raises(NotImplementedError):
            ConvertV0.convert_ov_type("error")

        assert torch.uint8 == ConvertV0.convert_ov_type("u1")
        assert torch.uint8 == ConvertV0.convert_ov_type("u4")
        assert torch.uint8 == ConvertV0.convert_ov_type("u8")
        assert torch.int32 == ConvertV0.convert_ov_type("u32")
        assert torch.int64 == ConvertV0.convert_ov_type("u64")
        assert torch.int8 == ConvertV0.convert_ov_type("i4")
        assert torch.int8 == ConvertV0.convert_ov_type("i8")
        assert torch.int16 == ConvertV0.convert_ov_type("i16")
        assert torch.int32 == ConvertV0.convert_ov_type("i32")
        assert torch.int64 == ConvertV0.convert_ov_type("i64")
        assert torch.float16 == ConvertV0.convert_ov_type("f16")
        assert torch.float32 == ConvertV0.convert_ov_type("f32")
        assert torch.bool == ConvertV0.convert_ov_type("boolean")

    @e2e_pytest_unit
    def test_convert_torch_type(self):
        with pytest.raises(NotImplementedError):
            ConvertV0.convert_torch_type("error")

        assert "u8" == ConvertV0.convert_torch_type(torch.uint8)
        assert "i8" == ConvertV0.convert_torch_type(torch.int8)
        assert "i16" == ConvertV0.convert_torch_type(torch.int16)
        assert "i32" == ConvertV0.convert_torch_type(torch.int32)
        assert "i64" == ConvertV0.convert_torch_type(torch.int64)
        assert "f16" == ConvertV0.convert_torch_type(torch.float16)
        assert "f32" == ConvertV0.convert_torch_type(torch.float32)
        assert "boolean" == ConvertV0.convert_torch_type(torch.bool)

    @e2e_pytest_unit
    def test_forward(self):
        op = ConvertV0("dummy", shape=(1,), destination_type="f16")
        output = op(torch.randn(5, dtype=torch.float32))
        assert output.dtype == torch.float16
