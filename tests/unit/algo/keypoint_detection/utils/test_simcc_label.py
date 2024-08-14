# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of SimCCLabel."""

import numpy as np
import pytest
from otx.algo.keypoint_detection.utils.simcc_label import SimCCLabel


class TestSimCCLabel:
    @pytest.fixture()
    def fxt_keypoints(self):
        return (0.1 + 0.8 * np.random.rand(1, 17, 2)) * [192, 256]

    @pytest.fixture()
    def fxt_keypoints_visible(self):
        return np.ones((1, 17), dtype=np.float32)

    @pytest.fixture()
    def fxt_codec_gaussian(self):
        return SimCCLabel(input_size=(256, 192), smoothing_type="gaussian", sigma=6.0, simcc_split_ratio=2.0)

    @pytest.fixture()
    def fxt_codec_smoothing(self):
        return SimCCLabel(
            input_size=(256, 192),
            smoothing_type="standard",
            sigma=5.0,
            simcc_split_ratio=3.0,
            label_smooth_weight=0.1,
        )

    @pytest.fixture()
    def fxt_codec_dark(self):
        return SimCCLabel(input_size=(256, 192), smoothing_type="gaussian", sigma=(4.9, 5.66), simcc_split_ratio=2.0)

    @pytest.fixture()
    def fxt_codec_separated_sigma(self):
        return SimCCLabel(input_size=(256, 192), smoothing_type="gaussian", sigma=(4.9, 5.66), simcc_split_ratio=2.0)

    @pytest.mark.parametrize(
        "fxt_codec",
        ["fxt_codec_gaussian", "fxt_codec_smoothing", "fxt_codec_dark", "fxt_codec_separated_sigma"],
    )
    def test_encode(self, fxt_codec, request, fxt_keypoints, fxt_keypoints_visible):
        codec = request.getfixturevalue(fxt_codec)
        encoded = codec.encode(fxt_keypoints, fxt_keypoints_visible)

        assert encoded["keypoint_x_labels"].shape == (1, 17, int(192 * codec.simcc_split_ratio))
        assert encoded["keypoint_y_labels"].shape == (1, 17, int(256 * codec.simcc_split_ratio))
        assert encoded["keypoint_weights"].shape == (1, 17)

    @pytest.mark.parametrize(
        "fxt_codec",
        ["fxt_codec_gaussian", "fxt_codec_smoothing", "fxt_codec_dark", "fxt_codec_separated_sigma"],
    )
    def test_decode(self, fxt_codec, request):
        codec = request.getfixturevalue(fxt_codec)
        simcc_x = np.random.rand(1, 17, int(192 * codec.simcc_split_ratio))
        simcc_y = np.random.rand(1, 17, int(256 * codec.simcc_split_ratio))

        keypoints, scores = codec.decode(simcc_x, simcc_y)
        assert keypoints.shape == (1, 17, 2)
        assert scores.shape == (1, 17)

        codec.decode_visibility = True

        simcc_x = np.random.rand(1, 17, int(192 * codec.simcc_split_ratio)) * 10
        simcc_y = np.random.rand(1, 17, int(256 * codec.simcc_split_ratio)) * 10
        keypoints, scores = codec.decode(simcc_x, simcc_y)

        assert len(scores) == 2
        assert scores[0].shape == (1, 17)
        assert scores[1].shape == (1, 17)
        assert scores[1].min() >= 0.0
        assert scores[1].max() <= 1.0

    @pytest.mark.parametrize(
        "fxt_codec",
        ["fxt_codec_gaussian", "fxt_codec_smoothing", "fxt_codec_dark", "fxt_codec_separated_sigma"],
    )
    def test_encode_decode(self, fxt_codec, request, fxt_keypoints, fxt_keypoints_visible):
        codec = request.getfixturevalue(fxt_codec)
        encoded = codec.encode(fxt_keypoints, fxt_keypoints_visible)
        keypoint_x_labels = encoded["keypoint_x_labels"]
        keypoint_y_labels = encoded["keypoint_y_labels"]

        _keypoints, _ = codec.decode(keypoint_x_labels, keypoint_y_labels)

        assert np.allclose(fxt_keypoints, _keypoints, atol=5.0)
