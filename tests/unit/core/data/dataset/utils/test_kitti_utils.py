# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for KITTI utils."""

import numpy as np
from otx.core.data.dataset.utils.kitti_utils import angle2class, class2angle, get_affine_transform, rect_to_img


def test_get_affine_transform():
    center = np.array([100, 100], dtype=np.float32)
    scale = np.array([1, 1], dtype=np.float32)
    rot = 90
    output_size = (400, 400)
    shift = np.array([0, 0], dtype=np.float32)

    # Calculate the expected affine transformation matrix
    expected_transform = np.array(
        [[0, 400, -39800], [-400, 0, 40200]],
        dtype=np.float32,
    )

    # Get the actual affine transformation matrix
    actual_transform = get_affine_transform(center, scale, rot, output_size, shift)

    # Compare the expected and actual transformation matrices
    assert np.allclose(actual_transform, expected_transform)


def test_rect_to_img():
    p2 = np.array([[1000, 0, 500, 0], [0, 1000, 500, 0], [0, 0, 1, 0]], dtype=np.float32)
    pts_rect = np.array([[0, 0, 10], [0, 0, 20], [0, 0, 30]], dtype=np.float32)
    expected_pts_img = np.array([[500, 500], [500, 500], [500, 500]], dtype=np.float32)
    expected_pts_rect_depth = np.array([10, 20, 30], dtype=np.float32)

    pts_img, pts_rect_depth = rect_to_img(p2, pts_rect)

    assert np.allclose(pts_img, expected_pts_img)
    assert np.allclose(pts_rect_depth, expected_pts_rect_depth)


def test_angle2class():
    # Test case 1
    angle1 = 0.0
    expected_class_id1 = 0
    expected_residual_angle1 = 0.0
    class_id1, residual_angle1 = angle2class(angle1)
    assert class_id1 == expected_class_id1
    assert np.isclose(residual_angle1, expected_residual_angle1)

    # Test case 2
    angle2 = np.pi / 2
    expected_class_id2 = 3
    expected_residual_angle2 = 0.0
    class_id2, residual_angle2 = angle2class(angle2)
    assert class_id2 == expected_class_id2
    assert np.isclose(residual_angle2, expected_residual_angle2)

    # Test case 3
    angle3 = 5 * np.pi / 4
    expected_class_id3 = 8
    expected_residual_angle3 = -np.pi / 12
    class_id3, residual_angle3 = angle2class(angle3)
    assert class_id3 == expected_class_id3
    assert np.isclose(residual_angle3, expected_residual_angle3)

    # Test case 4
    angle4 = -3 * np.pi / 2
    expected_class_id4 = 3
    expected_residual_angle4 = 0.0
    class_id4, residual_angle4 = angle2class(angle4)
    assert class_id4 == expected_class_id4
    assert np.isclose(residual_angle4, expected_residual_angle4)


def test_class2angle():
    # Test case 1
    cls1 = 0
    residual1 = 0.0
    expected_angle1 = 0.0
    angle1 = class2angle(cls1, residual1)
    assert angle2class(angle1) == (cls1, residual1)
    assert np.isclose(angle1, expected_angle1)

    # Test case 2
    cls2 = 3
    residual2 = 0.0
    expected_angle2 = 3 * np.pi / 6
    angle2 = class2angle(cls2, residual2)
    assert angle2class(angle2) == (cls2, residual2)
    assert np.isclose(angle2, expected_angle2)

    # Test case 3
    cls3 = 8
    residual3 = -np.pi / 12
    expected_angle3 = 5 * np.pi / 4
    angle3 = class2angle(cls3, residual3)
    assert np.isclose(angle3, expected_angle3)
