# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test NMS related class and functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from otx.algo.common.utils import nms as target_file
from otx.algo.common.utils.nms import NMSop


class TestNMSop:
    @pytest.fixture()
    def mock_torch_nms(self, mocker) -> MagicMock:
        def func(bboxes: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return torch.tensor(range(min(bboxes.size(0), 2)))

        return mocker.patch.object(target_file, "torch_nms", side_effect=func)

    @pytest.fixture()
    def mock_torch_autocast(self, mocker) -> MagicMock:
        return mocker.patch("otx.algo.common.utils.nms.torch.autocast")

    @pytest.fixture()
    def mock_bboxes(self) -> torch.Tensor:
        bboxes = torch.tensor(
            [
                [1.0, 1.0, 2.0, 2.0],
                [2.0, 2.0, 3.0, 3.0],
                [3.0, 3.0, 4.0, 4.0],
                [4.0, 4.0, 5.0, 5.0],
                [5.0, 5.0, 6.0, 6.0],
                [6.0, 6.0, 7.0, 7.0],
            ],
            dtype=torch.float32,
        )
        bboxes.get_device = MagicMock(return_value=0)
        return bboxes

    @pytest.fixture()
    def mock_bboxes_bfp16_cpu(self, mock_bboxes: torch.Tensor) -> torch.Tensor:
        bboxes = mock_bboxes.type(torch.bfloat16)
        bboxes.get_device = MagicMock(return_value=-1)
        return bboxes

    @pytest.fixture()
    def mock_scores(self) -> torch.Tensor:
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=torch.float32)
        scores.get_device = MagicMock(return_value=0)
        return scores

    @pytest.fixture()
    def mock_scores_bfp16_cpu(self, mock_scores: torch.Tensor) -> torch.Tensor:
        scores = mock_scores.type(torch.bfloat16)
        scores.get_device = MagicMock(return_value=-1)
        return scores

    @pytest.fixture()
    def mock_iou_threshold(self) -> MagicMock:
        return MagicMock()

    def test_forward(
        self,
        mock_torch_nms: MagicMock,
        mock_torch_autocast: MagicMock,
        mock_iou_threshold: MagicMock,
        mock_bboxes: target_file.Tensor,
        mock_scores: target_file.Tensor,
    ):
        nmsop = NMSop()
        nms = nmsop.forward(
            ctx=MagicMock(),
            bboxes=mock_bboxes,
            scores=mock_scores,
            iou_threshold=mock_iou_threshold,
            offset=MagicMock(),
            score_threshold=0.0,
            max_num=0,
        )

        mock_torch_autocast.assert_not_called()
        mock_torch_nms.assert_called_once_with(mock_bboxes, mock_scores, mock_iou_threshold)
        assert list(nms) == [0, 1]

    def test_forward_w_max_num(
        self,
        mock_torch_nms: MagicMock,
        mock_torch_autocast: MagicMock,
        mock_iou_threshold: MagicMock,
        mock_bboxes: target_file.Tensor,
        mock_scores: target_file.Tensor,
    ):
        nmsop = NMSop()
        nms = nmsop.forward(
            ctx=MagicMock(),
            bboxes=mock_bboxes,
            scores=mock_scores,
            iou_threshold=mock_iou_threshold,
            offset=MagicMock(),
            score_threshold=0.0,
            max_num=1,
        )

        mock_torch_autocast.assert_not_called()
        mock_torch_nms.assert_called_once_with(mock_bboxes, mock_scores, mock_iou_threshold)
        assert list(nms) == [0]

    def test_forward_w_score_threshold(
        self,
        mock_torch_nms: MagicMock,
        mock_torch_autocast: MagicMock,
        mock_iou_threshold: MagicMock,
        mock_bboxes: target_file.Tensor,
        mock_scores: target_file.Tensor,
    ):
        nmsop = NMSop()
        nms = nmsop.forward(
            ctx=MagicMock(),
            bboxes=mock_bboxes,
            scores=mock_scores,
            iou_threshold=mock_iou_threshold,
            offset=MagicMock(),
            score_threshold=0.85,
            max_num=0,
        )

        mock_torch_autocast.assert_not_called()
        mock_torch_nms.assert_called()
        assert (mock_torch_nms.call_args[0][0] == mock_bboxes[0]).all()
        assert (mock_torch_nms.call_args[0][1] == mock_scores[0]).all()
        assert list(nms) == [0]

    def test_forward_cpu_bf16(
        self,
        mock_torch_nms: MagicMock,
        mock_torch_autocast: MagicMock,
        mock_iou_threshold: MagicMock,
        mock_bboxes_bfp16_cpu: target_file.Tensor,
        mock_scores_bfp16_cpu: target_file.Tensor,
    ):
        nmsop = NMSop()
        nms = nmsop.forward(
            ctx=MagicMock(),
            bboxes=mock_bboxes_bfp16_cpu,
            scores=mock_scores_bfp16_cpu,
            iou_threshold=mock_iou_threshold,
            offset=MagicMock(),
            score_threshold=0.0,
            max_num=0,
        )

        mock_torch_autocast.assert_called()
        assert mock_torch_autocast.call_args.kwargs["enabled"] is False
        mock_torch_nms.assert_called()
        assert mock_torch_nms.call_args.args[0][0].dtype == torch.float32
        assert mock_torch_nms.call_args.args[0][1].dtype == torch.float32
        assert list(nms) == [0, 1]
