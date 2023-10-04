# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.v2.adapters.openvino.ops.type_conversions import ConvertV0


class TestConvertV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)


    def test_convert_ov_type(self) -> None:
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


    def test_convert_torch_type(self) -> None:
        with pytest.raises(NotImplementedError):
            ConvertV0.convert_torch_type("error")

        assert ConvertV0.convert_torch_type(torch.uint8) == "u8"
        assert ConvertV0.convert_torch_type(torch.int8) == "i8"
        assert ConvertV0.convert_torch_type(torch.int16) == "i16"
        assert ConvertV0.convert_torch_type(torch.int32) == "i32"
        assert ConvertV0.convert_torch_type(torch.int64) == "i64"
        assert ConvertV0.convert_torch_type(torch.float16) == "f16"
        assert ConvertV0.convert_torch_type(torch.float32) == "f32"
        assert ConvertV0.convert_torch_type(torch.bool) == "boolean"


    def test_forward(self) -> None:
        op = ConvertV0("dummy", shape=(1,), destination_type="f16")
        output = op(torch.randn(5, dtype=torch.float32))
        assert output.dtype == torch.float16
