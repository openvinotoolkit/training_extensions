# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.v2.adapters.openvino.ops.movements import (
    BroadcastV3,
    ConcatV0,
    GatherV0,
    GatherV1,
    PadV1,
    ScatterNDUpdateV3,
    ScatterUpdateV3,
    ShuffleChannelsV0,
    SplitV1,
    StridedSliceV1,
    TileV0,
    TransposeV1,
    VariadicSplitV1,
)


class TestPadV1:

    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid pad_mode error."):
            PadV1("dummy", shape=(1,), pad_mode="error")


    def test_get_torch_pad_mode(self) -> None:
        assert PadV1.get_torch_pad_mode("constant") == "constant"
        assert PadV1.get_torch_pad_mode("edge") == "replicate"
        assert PadV1.get_torch_pad_mode("reflect") == "reflect"
        with pytest.raises(NotImplementedError):
            PadV1.get_torch_pad_mode("symmetric")
        with pytest.raises(NotImplementedError):
            PadV1.get_torch_pad_mode("error")


    def test_get_torch_pad_dim(self) -> None:
        assert [10, 10, 10, 10] == PadV1.get_torch_pad_dim([10, 10], [10, 10])
        assert [9, 4, 8, 3, 7, 2, 6, 1] == PadV1.get_torch_pad_dim(range(6, 10), range(0, 5))


    def test_forward(self) -> None:
        for mode in ("constant", "edge", "reflect"):
            op = PadV1("dummy", shape=(1,), pad_mode=mode)
            _input = torch.empty(3, 3, 4, 2)
            output = op(_input, [1, 1], [1, 1])
            assert output.shape == (3, 3, 6, 4)

            output = op(_input, [2, 2, 1], [1, 2, 1])
            assert output.shape == (3, 6, 8, 4)


class TestConcatV0:

    def test_forward(self) -> None:
        op = ConcatV0("dummy", shape=(1,), axis=1)
        input_1 = torch.rand(4, 4, 32, 32)
        input_2 = torch.rand(4, 8, 32, 32)
        output = op(input_1, input_2)
        ref = torch.cat((input_1, input_2), 1)
        assert torch.equal(output, ref)


class TestTransposeV1:

    def test_forward(self) -> None:
        op = TransposeV1("dummy", shape=(1,))
        _input = torch.randn(1, 2, 3, 4, 5)

        assert op(_input, torch.tensor([])).shape == (5, 4, 3, 2, 1)
        assert op(_input, torch.tensor([0, 2, 1, 4, 3])).shape == (1, 3, 2, 5, 4)


class TestGatherV0:

    def test_forward(self) -> None:
        op = GatherV0("dummy", shape=(1,), batch_dims=0)
        output = op(torch.tensor([1, 2, 3, 4, 5]), torch.tensor([0, 0, 4]), torch.tensor(0))
        ref = torch.tensor([1, 1, 5])
        assert torch.equal(output, ref)

        op = GatherV0("dummy", shape=(1,), batch_dims=0)
        output = op(torch.tensor([1, 2, 3, 4, 5]), torch.tensor(4), torch.tensor(0))
        ref = torch.tensor(5)
        assert torch.equal(output, ref)

        op = GatherV0("dummy", shape=(1,), batch_dims=-1)
        output = op(
            torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            torch.tensor([[0, 0, 4], [4, 0, 0]]),
            torch.tensor(1),
        )
        ref = torch.tensor([[1, 1, 5], [10, 6, 6]])
        assert torch.equal(output, ref)

        op = GatherV0("dummy", shape=(1,), batch_dims=1)
        output = op(
            torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            torch.tensor([[0, 0, 4], [4, 0, 0]]),
            torch.tensor(1),
        )
        ref = torch.tensor([[1, 1, 5], [10, 6, 6]])
        assert torch.equal(output, ref)

        op = GatherV0("dummy", shape=(1,), batch_dims=2)
        output = op(
            torch.tensor(
                [
                    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                    [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                ],
            ),
            torch.tensor([[[0, 0, 4], [4, 0, 0]], [[1, 2, 4], [4, 3, 2]]]),
            torch.tensor(2),
        )
        ref = torch.tensor([[[1, 1, 5], [10, 6, 6]], [[12, 13, 15], [20, 19, 18]]])
        assert torch.equal(output, ref)

        op = GatherV0("dummy", shape=(1,), batch_dims=1)
        output = op(
            torch.tensor(
                [
                    [
                        [
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16],
                            [17, 18, 19, 20],
                        ],
                    ],
                    [
                        [
                            [21, 22, 23, 24],
                            [25, 26, 27, 28],
                            [29, 30, 31, 32],
                            [33, 34, 35, 36],
                            [37, 38, 39, 40],
                        ],
                    ],
                ],
            ),
            torch.tensor([[1, 2, 4], [4, 3, 2]]),
            torch.tensor(2),
        )
        ref = torch.tensor(
            [
                [[[5, 6, 7, 8], [9, 10, 11, 12], [17, 18, 19, 20]]],
                [[[37, 38, 39, 40], [33, 34, 35, 36], [29, 30, 31, 32]]],
            ],
        )
        assert torch.equal(output, ref)


class TestGatherV1:

    def test_forward(self) -> None:
        op = GatherV1("dummy", shape=(1,))
        _input = torch.tensor([[1, 2], [3, 4]])
        output = op(_input, torch.tensor([[0, 0], [1, 0]]), 1)
        ref = torch.gather(_input, 1, torch.tensor([[0, 0], [1, 0]]))
        assert torch.equal(output, ref)


class TestStridedSliceV1:

    def test_forward(self) -> None:
        op = StridedSliceV1(
            "dummy",
            shape=(1,),
            begin_mask=[0, 1, 1],
            end_mask=[1, 1, 0],
            new_axis_mask=[0, 0, 0],
            shrink_axis_mask=[0, 0, 0],
        )
        output = op(
            torch.randn(2, 3, 4),
            torch.tensor([1, 0, 0]),
            torch.tensor([0, 0, 2]),
            torch.tensor([1, 1, 1]),
        )
        assert output.shape == (1, 3, 2)

        op = StridedSliceV1(
            "dummy",
            shape=(1,),
            begin_mask=[0, 1, 1],
            end_mask=[1, 1, 0],
            new_axis_mask=[1, 0, 0],
            shrink_axis_mask=[0, 0, 0],
        )
        output = op(
            torch.randn(2, 3, 4),
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 0, 4]),
            torch.tensor([1, 1, 1]),
        )
        assert output.shape == (1, 2, 3, 4)


class TestSplitV1:

    def test_forward(self) -> None:
        op = SplitV1("dummy", shape=(1,), num_splits=2)
        _input = torch.randn(4, 8, 32)
        outputs = op(_input, 1)
        refs = torch.split(_input, 4, 1)
        for output, ref in zip(outputs, refs):
            assert torch.equal(output, ref)


class TestVariadicSplitV1:

    def test_forward(self) -> None:
        op = VariadicSplitV1("dummy", shape=(1,))
        outputs = op(torch.randn(6, 12, 10, 24), 0, torch.tensor([1, 2, 3]))
        ref_shapes = ((1, 12, 10, 24), (2, 12, 10, 24), (3, 12, 10, 24))
        for output, ref_shape in zip(outputs, ref_shapes):
            assert output.shape == ref_shape

        outputs = op(torch.randn(6, 12, 10, 24), 0, torch.tensor([-1, 2]))
        ref_shapes = ((4, 12, 10, 24), (2, 12, 10, 24))
        for output, ref_shape in zip(outputs, ref_shapes):
            assert output.shape == ref_shape


class TestShuffleChannelsV0:

    def test_forward(self) -> None:
        op = ShuffleChannelsV0("dummy", shape=(1,))
        _input = torch.randn(4, 8, 32, 32)
        output = op(_input)
        assert torch.equal(output, _input)

        op = ShuffleChannelsV0("dummy", shape=(1,), group=2)
        _input = torch.randn(4, 8, 32, 32)
        output = op(_input)
        assert not torch.equal(output, _input)


class TestBroadcastV3:

    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode error."):
            BroadcastV3("dummy", shape=(1,), mode="error")


    def test_forward(self) -> None:
        op = BroadcastV3("dummy", shape=(1,), mode="numpy")
        output = op(torch.randn(16, 1, 1), torch.tensor([1, 16, 50, 50]))
        assert output.shape == (1, 16, 50, 50)

        op = BroadcastV3("dummy", shape=(1,), mode="explicit")
        output = op(torch.randn(16), torch.tensor([1, 16, 50, 50]), torch.tensor([1]))
        assert output.shape == (1, 16, 50, 50)


class TestScatterNDUpdateV3:

    def test_forward(self) -> None:
        ScatterNDUpdateV3("dummy", shape=(1,))
        # TODO
        pass


class TestScatterUpdateV3:

    def test_forward(self) -> None:
        op = ScatterUpdateV3("dummy", shape=(1,))
        updates = torch.arange(1, 11).reshape((2, 5))
        _input = torch.zeros(3, 5, dtype=updates.dtype)
        indices = torch.tensor([[0, 1, 2, 0]])

        output = op(_input, indices, updates, torch.tensor(0))
        ref = torch.tensor([[1, 0, 0, 4, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0]])
        assert torch.equal(output, ref)


class TestTileV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 3, 256, 256)
        self._input = torch.randn(self.shape)


    def test_forward(self) -> None:
        op = TileV0("dummy", shape=(self.shape,))
        output = op(self._input, torch.tensor([2, 2, 2, 2]))
        assert torch.equal(output, torch.tile(self._input, [2, 2, 2, 2]))
