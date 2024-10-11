# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module defines utils for 3D Dataset."""

from __future__ import annotations

import cv2
import numpy as np


def get_calib_from_file(calib_file: str) -> np.ndarray:
    """Get calibration matrix from txt file (KITTI format)."""
    with open(calib_file) as f:  # noqa: PTH123
        lines = f.readlines()

    obj = lines[2].strip().split(" ")[1:]

    return np.array(obj, dtype=np.float32).reshape(3, 4)


def cart_to_hom(pts: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to homogeneous coordinates.

    Args:
        pts (np.ndarray): Array of Cartesian coordinates with shape (N, D),
            where N is the number of points and D is the number of dimensions.

    Returns:
        np.ndarray: Array of homogeneous coordinates with shape (N, D+1),
            where N is the number of points and D is the number of dimensions.
    """
    return np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))


def rect_to_img(p2: np.ndarray, pts_rect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera coordinates to image coordinates.

    Args:
        p2 (np.ndarray): Projection matrix with shape (3, 4).
        pts_rect (np.ndarray): Rectangular coordinates with shape (N, 4).

    Returns:
        np.ndarray: Image coordinates with shape (N, 2).
    """
    pts_rect_hom = cart_to_hom(pts_rect)
    pts_2d_hom = np.dot(pts_rect_hom, p2.T)
    pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
    pts_rect_depth = pts_2d_hom[:, 2] - p2.T[3, 2]  # depth in rect camera coord
    return pts_img, pts_rect_depth


def ry2alpha(p2: np.ndarray, ry: np.ndarray, u: np.ndarray) -> np.ndarray:  #!
    """Get observation angle of object.

    Args:
        p2 (np.ndarray): Projection matrix with shape (3, 4).
        ry (np.ndarray): Observation angle of object with shape (N, ).
        u (np.ndarray): Pixel coordinates with shape (N, 2).

    Returns:
        np.ndarray: Observation angle of object with shape (N, ).
    """
    alpha = ry - np.arctan2(u - p2[0, 2], p2[0, 0])

    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi

    return alpha


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: tuple[int, int],
    shift: np.ndarray | None = None,
    inv: int = 0,
) -> np.ndarray:
    """Get the affine transformation matrix.

    Args:
        center: The center of the bounding box.
        scale: The scale factors for width and height.
        rot: The rotation angle in degrees.
        output_size: The size of the output image.
        shift: The shift vector. Defaults to [0, 0] if None.
        inv: Whether to compute the inverse transformation matrix.

    Returns:
        The affine transformation matrix.

    """

    def _get_dir(src_point: np.ndarray, rot_rad: float) -> np.ndarray:
        """Get the direction of the src_point and rot_rad."""
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Get the third point of the line segment ab."""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    if shift is None:
        shift = np.array([0, 0], dtype=np.float32)

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, trans_inv

    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def affine_transform(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply an affine transformation to the points."""
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def angle2class(angle: float) -> tuple[int, float]:
    """Convert continuous angle to discrete class and residual."""
    num_heading_bin = 12
    angle = angle % (2 * np.pi)
    if not (angle >= 0 and angle <= 2 * np.pi):
        msg = "angle not in 0 ~ 2pi"
        raise ValueError(msg)

    angle_per_class = 2 * np.pi / float(num_heading_bin)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(cls: int, residual: float, to_label_format: bool = False) -> float:
    """Inverse function to angle2class."""
    num_heading_bin = 12
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle
