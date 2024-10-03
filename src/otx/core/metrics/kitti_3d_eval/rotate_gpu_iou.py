# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Rotate IoU for KITTI3D metric, gpu version."""

import math

import numba
import numpy as np
from numba import cuda


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


@cuda.jit("(float32[:], float32[:], float32[:])", device=True, inline=True)
def trangle_area(a: cuda.local.array, b: cuda.local.array, c: cuda.local.array) -> float:
    """Calculate the area of a triangle given its three vertices.

    Args:
        a (cuda.local.array): First vertex of the triangle.
        b (cuda.local.array): Second vertex of the triangle.
        c (cuda.local.array): Third vertex of the triangle.

    Returns:
        float: Area of the triangle.
    """
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@cuda.jit("(float32[:], int32)", device=True, inline=True)
def area(int_pts: cuda.local.array, num_of_inter: int) -> float:
    """Calculate the area of a polygon using the given intersection points.

    Args:
        int_pts (cuda.local.array): Array of intersection points, shape (num_of_inter * 2,).
        num_of_inter (int): Number of intersection points.

    Returns:
        float: The calculated area of the polygon.
    """
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(trangle_area(int_pts[:2], int_pts[2 * i + 2 : 2 * i + 4], int_pts[2 * i + 4 : 2 * i + 6]))
    return area_val


@cuda.jit("(float32[:], int32)", device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts: cuda.local.array, num_of_inter: int) -> None:
    """Sort the vertices of a convex polygon in counterclockwise order.

    Args:
        int_pts (cuda.local.array): Array of intersection points.
        num_of_inter (int): Number of intersection points.
    """
    if num_of_inter > 0:
        center = cuda.local.array((2,), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2,), dtype=numba.float32)
        vs = cuda.local.array((16,), dtype=numba.float32)
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


@cuda.jit("(float32[:], float32[:], int32, int32, float32[:])", device=True, inline=True)
def line_segment_intersection(
    pts1: cuda.local.array,  # array of points representing the first line segment
    pts2: cuda.local.array,  # array of points representing the second line segment
    i: int,  # index of the first line segment
    j: int,  # index of the second line segment
    temp_pts: cuda.local.array,  # array to store the intersection point
) -> bool:
    """Check if two line segments intersect and find the intersection point.

    Args:
        pts1 (cuda.local.array): Array of points representing the first line segment.
        pts2 (cuda.local.array): Array of points representing the second line segment.
        i (int): Index of the first line segment.
        j (int): Index of the second line segment.
        temp_pts (cuda.local.array): Array to store the intersection point.

    Returns:
        bool: True if the line segments intersect, False otherwise.
    """
    a = cuda.local.array((2,), dtype=numba.float32)
    b = cuda.local.array((2,), dtype=numba.float32)
    c = cuda.local.array((2,), dtype=numba.float32)
    d = cuda.local.array((2,), dtype=numba.float32)

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


@cuda.jit("(float32[:], float32[:], int32, int32, float32[:])", device=True, inline=True)
def line_segment_intersection_v1(
    pts1: cuda.local.array,  # array of points representing the first line segment
    pts2: cuda.local.array,  # array of points representing the second line segment
    i: int,  # index of the first line segment
    j: int,  # index of the second line segment
    temp_pts: cuda.local.array,  # array to store the intersection point
) -> bool:
    """Check if two line segments intersect and find the intersection point using an alternative method.

    Args:
        pts1(cuda.local.array): array of points representing the first line segment
        pts2(cuda.local.array): cuda.local.array, array of points representing the second line segment
        i(int): int, index of the first line segment
        j(int): int, index of the second line segment
        temp_pts(cuda.local.array): array to store the intersection point

    Returns:
        bool: True if the line segments intersect, False otherwise
    """
    a = cuda.local.array((2,), dtype=numba.float32)
    b = cuda.local.array((2,), dtype=numba.float32)
    c = cuda.local.array((2,), dtype=numba.float32)
    d = cuda.local.array((2,), dtype=numba.float32)

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


@cuda.jit("(float32, float32, float32[:])", device=True, inline=True)
def point_in_quadrilateral(
    pt_x: float,  # x coordinate of the point
    pt_y: float,  # y coordinate of the point
    corners: cuda.local.array,  # corners of the quadrilateral, shape (8,)
) -> bool:
    """Check if a point is inside a quadrilateral.

    Args:
        pt_x (float): x coordinate of the point
        pt_y (float): y coordinate of the point
        corners (cuda.local.array): shape (8,), corners of the quadrilateral

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


@cuda.jit("(float32[:], float32[:], float32[:])", device=True, inline=True)
def quadrilateral_intersection(
    pts1: cuda.local.array,  # shape: (8,)
    pts2: cuda.local.array,  # shape: (8,)
    int_pts: cuda.local.array,  # shape: (16,)
) -> int:
    """Compute the intersection points between two quadrilaterals.

    Args:
        pts1(cuda.local.array): Array of points representing the first quadrilateral, shape (8,).
        pts2(cuda.local.array): Array of points representing the second quadrilateral, shape (8,).
        int_pts(cuda.local.array): Array to store the intersection points, shape (16,).

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
    temp_pts = cuda.local.array((2,), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit("(float32[:], float32[:])", device=True, inline=True)
def rbbox_to_corners(
    corners: cuda.local.array,  # shape: (8,)
    rbbox: cuda.local.array,  # shape: (5,)
) -> None:
    """Convert a rotated bounding box to its corner points.

    Args:
        corners (cuda.local.array): Array to store the corner points, shape (8,).
        rbbox (cuda.local.array): Array representing the rotated bounding box, shape (5,).
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
    corners_x = cuda.local.array((4,), dtype=numba.float32)
    corners_y = cuda.local.array((4,), dtype=numba.float32)
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


@cuda.jit("(float32[:], float32[:])", device=True, inline=True)
def inter(
    rbbox1: cuda.local.array,  # shape: (5,)
    rbbox2: cuda.local.array,  # shape: (5,)
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
    corners1 = cuda.local.array((8,), dtype=numba.float32)
    corners2 = cuda.local.array((8,), dtype=numba.float32)
    intersection_corners = cuda.local.array((16,), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2, intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


@cuda.jit("(float32[:], float32[:], int32)", device=True, inline=True)
def dev_rotate_iou_eval(
    rbox1: cuda.shared.array,  # shape: (5,)
    rbox2: cuda.shared.array,  # shape: (5,)
    criterion: int = -1,  # IoU criterion to use. Defaults to -1.
) -> float:  # The IoU of the two rotated bounding boxes.
    """Calculate the IoU of two rotated bounding boxes.

    Args:
        rbox1 (cuda.shared.array): Array representing the first rotated bounding box.
            The rotated bounding box is represented by (center_x, center_y, width, height, angle).
        rbox2 (cuda.shared.array): Array representing the second rotated bounding box.
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


@cuda.jit("(int64, int64, float32[:], float32[:], float32[:], int32)", fastmath=False)
def rotate_iou_kernel_eval(
    n: int,
    k: int,
    dev_boxes: cuda.shared.array,
    dev_query_boxes: cuda.shared.array,
    dev_iou: cuda.shared.array,
    criterion: int = -1,
) -> None:
    """Calculate the IoU of two rotated bounding boxes.

    Args:
        N (int): Number of boxes.
        K (int): Number of query boxes.
        dev_boxes (cuda.shared.array): Array representing the boxes.
        dev_query_boxes (cuda.shared.array): Array representing the query boxes.
        dev_iou (cuda.shared.array): Array to store the IoU values.
        criterion (int): The method to calculate the IoU.
            -1: Calculate the IoU.
            0: Calculate the IoU with the first box as the reference.
            1: Calculate the IoU with the second box as the reference.

    """
    threads_per_block = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(n - row_start * threads_per_block, threads_per_block)
    col_size = min(k - col_start * threads_per_block, threads_per_block)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)

    dev_query_box_idx = threads_per_block * col_start + tx
    dev_box_idx = threads_per_block * row_start + tx
    if tx < col_size:
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if tx < row_size:
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threads_per_block * k + col_start * threads_per_block + tx * k + i
            dev_iou[offset] = dev_rotate_iou_eval(
                block_qboxes[i * 5 : i * 5 + 5],
                block_boxes[tx * 5 : tx * 5 + 5],
                criterion,
            )


def rotate_iou_eval_gpu(
    boxes: np.ndarray,  # shape: (n, 5)
    query_boxes: np.ndarray,  # shape: (k, 5)
    criterion: int = -1,  # IoU criterion to use. Defaults to -1.
    device_id: int = 0,
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
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    iou = np.zeros((n, k), dtype=np.float32)
    if n == 0 or k == 0:
        return iou
    threads_per_block = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(n, threads_per_block), div_up(k, threads_per_block))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threads_per_block, stream](
            n,
            k,
            boxes_dev,
            query_boxes_dev,
            iou_dev,
            criterion,
        )
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)
