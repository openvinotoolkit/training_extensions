# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Rotate IoU for KITTI3D metric."""

import math

import numba
import numpy as np


@numba.jit(nopython=True)
def div_up(m: int, n: int) -> int:
    """Divide m by n and round up to the nearest integer.

    Args:
        m (int): Numerator.
        n (int): Denominator.

    Returns:
        int: Result of the division rounded up to the nearest integer.
    """
    return m // n + (m % n > 0)


@numba.jit(nopython=True, inline="always")
def trangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate the area of a triangle given its three vertices.

    Args:
        a (ndarray): First vertex of the triangle.
        b (ndarray): Second vertex of the triangle.
        c (ndarray): Third vertex of the triangle.

    Returns:
        float: Area of the triangle.
    """
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@numba.jit(nopython=True, inline="always")
def area(int_pts: np.ndarray, num_of_inter: int) -> float:
    """Calculate the area of a polygon using the given intersection points.

    Args:
        int_pts (ndarray): Array of intersection points, shape (num_of_inter * 2,).
        num_of_inter (int): Number of intersection points.

    Returns:
        float: The calculated area of the polygon.
    """
    area_val: float = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(
                int_pts[:2],
                int_pts[2 * i + 2 : 2 * i + 4],
                int_pts[2 * i + 4 : 2 * i + 6],
            ),
        )
    return area_val


@numba.jit(nopython=True, inline="always")
def sort_vertex_in_convex_polygon(int_pts: np.ndarray, num_of_inter: int) -> None:
    """Sort the vertices of a convex polygon in counterclockwise order.

    Args:
        int_pts: Array of intersection points.
        num_of_inter: Number of intersection points.
    """
    if num_of_inter > 0:
        center = np.empty((2,), dtype=np.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = np.empty((2,), dtype=np.float32)
        vs = np.empty((16,), dtype=np.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@numba.jit(nopython=True, inline="always")
def line_segment_intersection(
    pts1: np.ndarray,  # array of points representing the first line segment
    pts2: np.ndarray,  # array of points representing the second line segment
    i: int,  # index of the first line segment
    j: int,  # index of the second line segment
    temp_pts: np.ndarray,  # array to store the intersection point
) -> bool:
    """Check if two line segments intersect and find the intersection point.

    Args:
        pts1 (ndarray): Array of points representing the first line segment.
        pts2 (ndarray): Array of points representing the second line segment.
        i (int): Index of the first line segment.
        j (int): Index of the second line segment.
        temp_pts (ndarray): Array to store the intersection point.

    Returns:
        bool: True if the line segments intersect, False otherwise.
    """
    a = np.empty((2,), dtype=np.float32)
    b = np.empty((2,), dtype=np.float32)
    c = np.empty((2,), dtype=np.float32)
    d = np.empty((2,), dtype=np.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    ba0 = b[0] - a[0]
    ba1 = b[1] - a[1]
    da0 = d[0] - a[0]
    ca0 = c[0] - a[0]
    da1 = d[1] - a[1]
    ca1 = c[1] - a[1]

    acd = da1 * ca0 > ca1 * da0
    bcd = (d[1] - b[1]) * (c[0] - b[0]) > (c[1] - b[1]) * (d[0] - b[0])
    if acd != bcd:
        abc = ca1 * ba0 > ba1 * ca0
        abd = da1 * ba0 > ba1 * da0
        if abc != abd:
            dc0 = d[0] - c[0]
            dc1 = d[1] - c[1]
            abba = a[0] * b[1] - b[0] * a[1]
            cddc = c[0] * d[1] - d[0] * c[1]
            dh = ba1 * dc0 - ba0 * dc1
            dx = abba * dc0 - ba0 * cddc
            dy = abba * dc1 - ba1 * cddc
            temp_pts[0] = dx / dh
            temp_pts[1] = dy / dh
            return True
    return False


@numba.jit(nopython=True, inline="always")
def line_segment_intersection_v1(
    pts1: np.ndarray,  # array of points representing the first line segment
    pts2: np.ndarray,  # array of points representing the second line segment
    i: int,  # index of the first line segment
    j: int,  # index of the second line segment
    temp_pts: np.ndarray,  # array to store the intersection point
) -> bool:
    """Check if two line segments intersect and find the intersection point using an alternative method.

    Args:
        pts1: ndarray, array of points representing the first line segment
        pts2: ndarray, array of points representing the second line segment
        i: int, index of the first line segment
        j: int, index of the second line segment
        temp_pts: ndarray, array to store the intersection point

    Returns:
        bool: True if the line segments intersect, False otherwise
    """
    a = np.empty((2,), dtype=np.float32)
    b = np.empty((2,), dtype=np.float32)
    c = np.empty((2,), dtype=np.float32)
    d = np.empty((2,), dtype=np.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True


@numba.jit(nopython=True, inline="always")
def point_in_quadrilateral(
    pt_x: float,  # x coordinate of the point
    pt_y: float,  # y coordinate of the point
    corners: np.ndarray,  # corners of the quadrilateral, shape (8,)
) -> bool:
    """Check if a point is inside a quadrilateral.

    Args:
        pt_x: float, x coordinate of the point
        pt_y: float, y coordinate of the point
        corners: ndarray, shape (8,), corners of the quadrilateral

    Returns:
        bool: True if the point is inside the quadrilateral, False otherwise
    """
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@numba.jit(nopython=True, inline="always")
def quadrilateral_intersection(
    pts1: np.ndarray,  # shape: (8,)
    pts2: np.ndarray,  # shape: (8,)
    int_pts: np.ndarray,  # shape: (16,)
) -> int:
    """Compute the intersection points between two quadrilaterals.

    Args:
        pts1: Array of points representing the first quadrilateral, shape (8,).
        pts2: Array of points representing the second quadrilateral, shape (8,).
        int_pts: Array to store the intersection points, shape (16,).

    Returns:
        int: Number of intersection points.
    """
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.empty((2,), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@numba.jit(nopython=True, inline="always")
def rbbox_to_corners(
    corners: np.ndarray,  # shape: (8,)
    rbbox: np.ndarray,  # shape: (5,)
) -> None:
    """Convert a rotated bounding box to its corner points.

    Args:
        corners (ndarray): Array to store the corner points, shape (8,).
        rbbox (ndarray): Array representing the rotated bounding box, shape (5,).
            The rotated bounding box is represented by (center_x, center_y, width, height, angle).

    Returns:
        None
    """
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = np.empty((4,), dtype=np.float32)
    corners_y = np.empty((4,), dtype=np.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@numba.jit(nopython=True, inline="always")
def inter(
    rbbox1: np.ndarray,  # shape: (5,)
    rbbox2: np.ndarray,  # shape: (5,)
) -> float:  # The intersection area of the two rotated bounding boxes.
    """Calculate the intersection area of two rotated bounding boxes.

    Args:
        rbbox1 (ndarray): Array representing the first rotated bounding box.
            The rotated bounding box is represented by (center_x, center_y, width, height, angle).
        rbbox2 (ndarray): Array representing the second rotated bounding box.
            The rotated bounding box is represented by (center_x, center_y, width, height, angle).

    Returns:
        float: The intersection area of the two rotated bounding boxes.
    """
    corners1 = np.empty((8,), dtype=np.float32)
    corners2 = np.empty((8,), dtype=np.float32)
    intersection_corners = np.empty((16,), dtype=np.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2, intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)

    return area(intersection_corners, num_intersection)


@numba.jit(nopython=True, inline="always")
def dev_rotate_iou_eval(
    rbox1: np.ndarray,  # shape: (5,)
    rbox2: np.ndarray,  # shape: (5,)
    criterion: int = -1,  # IoU criterion to use. Defaults to -1.
) -> float:  # The IoU of the two rotated bounding boxes.
    """Calculate the IoU of two rotated bounding boxes.

    Args:
        rbox1 (ndarray): Array representing the first rotated bounding box.
            The rotated bounding box is represented by (center_x, center_y, width, height, angle).
        rbox2 (ndarray): Array representing the second rotated bounding box.
            The rotated bounding box is represented by (center_x, center_y, width, height, angle).
        criterion (int): The method to calculate the IoU.
            -1: Calculate the IoU.
            0: Calculate the IoU with first box as the reference.
            1: Calculate the IoU with second box as the reference.

    Returns:
        float: The IoU of the two rotated bounding boxes.
    """
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    if criterion == 0:
        return area_inter / area1
    if criterion == 1:
        return area_inter / area2
    return area_inter


@numba.jit(nopython=True, inline="always")
def rotate_iou_eval(
    boxes: np.ndarray,  # shape: (n, 5)
    query_boxes: np.ndarray,  # shape: (k, 5)
    criterion: int = -1,  # IoU criterion to use. Defaults to -1.
) -> np.ndarray:  # shape: (n, k)
    """Compute the rotated box IoU between two sets of boxes on CPU.

    Args:
        boxes (ndarray): Array of shape (n, 5) representing n rotated boxes.
            Each box is represented by (center_x, center_y, width, height, angle).
        query_boxes (ndarray): Array of shape (k, 5) representing k query rotated boxes.
            Each query box is represented by (center_x, center_y, width, height, angle).
        criterion (int, optional): IoU criterion to use. Defaults to -1.

    Returns:
        ndarray: Array of shape (n, k) representing the IoU between each pair of boxes.
    """
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    iou = np.zeros((n, k), dtype=np.float32)
    if n == 0 or k == 0:
        return iou

    for i in range(n):
        for j in range(k):
            iou[i, j] = dev_rotate_iou_eval(boxes[i], query_boxes[j], criterion)

    return iou
