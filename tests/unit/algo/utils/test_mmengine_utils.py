# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmengine/blob/main/tests/test_structures/test_instance_data.py

import itertools
import random

import numpy as np
import pytest
import torch
from otx.algo.utils.mmengine_utils import InstanceData, is_seq_of, is_tuple_of


def test_is_seq_of():
    # Test case 1: Valid sequence of integers
    seq1 = [1, 2, 3, 4, 5]
    assert is_seq_of(seq1, int)

    # Test case 2: Valid sequence of strings
    seq2 = ["a", "b", "c", "d"]
    assert is_seq_of(seq2, str)

    # Test case 3: Invalid sequence of integers and strings
    seq3 = [1, "a", 2, "b"]
    assert not is_seq_of(seq3, int)

    # Test case 4: Empty sequence
    seq4 = []
    assert is_seq_of(seq4, int)

    # Test case 5: Valid sequence of tuples
    seq5 = [(1, "a"), (2, "b"), (3, "c")]
    assert is_seq_of(seq5, tuple)

    # Test case 6: Invalid sequence of tuples and lists
    seq6 = [(1, "a"), [2, "b"], (3, "c")]
    assert not is_seq_of(seq6, tuple)


def test_is_tuple_of():
    seq = [(1, "a"), (2, "b"), (3, "c")]
    assert not is_tuple_of(seq, tuple)
    seq2 = (1, "a", 2, "b")
    assert not is_tuple_of(seq2, tuple)
    seq3 = ((1, "a"), (2, "b"), (3, "c"))
    assert is_tuple_of(seq3, tuple)


class TmpObject:
    def __init__(self, tmp) -> None:
        assert isinstance(tmp, list)
        if len(tmp) > 0:
            for t in tmp:
                assert isinstance(t, list)
        self.tmp = tmp

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                msg = f"Index {item} out of range!"
                raise IndexError(msg)
            # keep the dimension
            item = slice(item, None, len(self))
        return TmpObject(self.tmp[item])

    @staticmethod
    def cat(tmp_objs) -> "TmpObject":
        assert all(isinstance(results, TmpObject) for results in tmp_objs)
        if len(tmp_objs) == 1:
            return tmp_objs[0]
        tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        tmp_list = list(itertools.chain(*tmp_list))
        return TmpObject(tmp_list)

    def __repr__(self):
        return str(self.tmp)


class TmpObjectWithoutCat:
    def __init__(self, tmp) -> None:
        assert isinstance(tmp, list)
        if len(tmp) > 0:
            for t in tmp:
                assert isinstance(t, list)
        self.tmp = tmp

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                msg = f"Index {item} out of range!"
                raise IndexError(msg)
            # keep the dimension
            item = slice(item, None, len(self))
        return TmpObjectWithoutCat(self.tmp[item])

    def __repr__(self):
        return str(self.tmp)


class TestInstanceData:
    @pytest.fixture()
    def instance_data(self) -> InstanceData:
        metainfo = {
            "img_id": random.randint(0, 100),  # noqa: S311
            "img_shape": (random.randint(400, 600), random.randint(400, 600)),  # noqa: S311
        }
        instances_infos = [1] * 5
        bboxes = torch.rand((5, 4))
        labels = np.random.rand(5)
        kps = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        ids = (1, 2, 3, 4, 5)
        name_ids = "12345"
        polygons = TmpObject(np.arange(25).reshape((5, -1)).tolist())
        return InstanceData(
            metainfo=metainfo,
            bboxes=bboxes,
            labels=labels,
            polygons=polygons,
            kps=kps,
            ids=ids,
            name_ids=name_ids,
            instances_infos=instances_infos,
        )

    def test_set_data(self, instance_data):
        # test set '_metainfo_fields' or '_data_fields'
        with pytest.raises(AttributeError):
            instance_data._metainfo_fields = 1
        with pytest.raises(AttributeError):
            instance_data._data_fields = 1

        instance_data.keypoints = torch.rand((5, 2))
        assert "keypoints" in instance_data

    def test_getitem(self, instance_data):
        _instance_data = InstanceData()
        # length must be greater than 0
        with pytest.raises(IndexError):
            _instance_data[1]

        assert len(instance_data) == 5
        slice_instance_data = instance_data[:2]
        assert len(slice_instance_data) == 2
        slice_instance_data = instance_data[1]
        assert len(slice_instance_data) == 1
        # assert the index should in 0 ~ len(instance_data) -1
        with pytest.raises(IndexError):
            instance_data[5]

        item = torch.Tensor([1, 2, 3, 4])  # float
        with pytest.raises((IndexError, TypeError)):
            instance_data[item]

        # when input is a bool tensor, the shape of
        # the input at index 0 should equal to
        # the value length in instance_data_field
        with pytest.raises(IndexError):
            instance_data[item.bool()]

        # test LongTensor
        long_tensor = torch.randint(5, (2,))
        long_index_instance_data = instance_data[long_tensor]
        assert len(long_index_instance_data) == len(long_tensor)

        # test BoolTensor
        bool_tensor = torch.rand(5) > 0.5
        bool_index_instance_data = instance_data[bool_tensor]
        assert len(bool_index_instance_data) == bool_tensor.sum()
        bool_tensor = torch.rand(5) > 1
        empty_instance_data = instance_data[bool_tensor]
        assert len(empty_instance_data) == bool_tensor.sum()

        # test list index
        list_index = [1, 2]
        list_index_instance_data = instance_data[list_index]
        assert len(list_index_instance_data) == len(list_index)

        # test list bool
        list_bool = [True, False, True, False, False]
        list_bool_instance_data = instance_data[list_bool]
        assert len(list_bool_instance_data) == 2

        # test numpy
        long_numpy = np.random.randint(5, size=2)
        long_numpy_instance_data = instance_data[long_numpy]
        assert len(long_numpy_instance_data) == len(long_numpy)

        bool_numpy = np.random.rand(5) > 0.5
        bool_numpy_instance_data = instance_data[bool_numpy]
        assert len(bool_numpy_instance_data) == bool_numpy.sum()

        # without cat
        instance_data.polygons = TmpObjectWithoutCat(np.arange(25).reshape((5, -1)).tolist())
        bool_numpy = np.random.rand(5) > 0.5
        with pytest.raises(
            ValueError,
            match=(
                "The type of `polygons` is "
                f"`{type(instance_data.polygons)}`, "
                "which has no attribute of `cat`, so it does not "
                f"support slice with `bool`"
            ),
        ):
            bool_numpy_instance_data = instance_data[bool_numpy]

    def test_len(self, instance_data):
        assert len(instance_data) == 5
        instance_data = InstanceData()
        assert len(instance_data) == 0
