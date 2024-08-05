# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation of keypoint evaluation utilities."""
from __future__ import annotations

from itertools import product

import cv2
import numpy as np


def get_simcc_maximum(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    apply_softmax: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    if simcc_x.ndim not in (2, 3):
        msg = f"Invalid shape {simcc_x.shape}"
        raise ValueError(msg)
    if simcc_y.ndim not in (2, 3):
        msg = f"Invalid shape {simcc_y.shape}"
        raise ValueError(msg)
    if simcc_x.ndim != simcc_y.ndim:
        msg = f"{simcc_x.shape} != {simcc_y.shape}"
        raise ValueError(msg)

    if simcc_x.ndim == 3:
        batch_size, num_keypoints, _ = simcc_x.shape
        simcc_x = simcc_x.reshape(batch_size * num_keypoints, -1)
        simcc_y = simcc_y.reshape(batch_size * num_keypoints, -1)
    else:
        batch_size = None

    if apply_softmax:
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1

    if batch_size:
        locs = locs.reshape(batch_size, num_keypoints, 2)
        vals = vals.reshape(batch_size, num_keypoints)

    return locs, vals


def get_heatmap_maximum(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    if heatmaps.ndim != 3 or heatmaps.ndim != 4:
        msg = f"Invalid shape {heatmaps.shape}"
        raise ValueError(msg)

    if heatmaps.ndim == 3:
        num_keypoints, y_dim, x_dim = heatmaps.shape
        batch_size = None
        heatmaps_flatten = heatmaps.reshape(num_keypoints, -1)
    else:
        batch_size, num_keypoints, y_dim, x_dim = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(batch_size * num_keypoints, -1)

    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(y_dim, x_dim))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1

    if batch_size:
        locs = locs.reshape(batch_size, num_keypoints, 2)
        vals = vals.reshape(batch_size, num_keypoints)

    return locs, vals


def _calc_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray, norm_factor: np.ndarray) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    batch_size, num_keypoints, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((batch_size, num_keypoints), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances: np.ndarray, thr: float = 0.5) -> float | int:
    """Return the percentage below the distance threshold.

    Ignore distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def keypoint_pck_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    thr: np.ndarray,
    norm_factor: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Calculate the pose accuracy of PCK for each individual keypoint.

    And the averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc: np.ndarray = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc: np.ndarray = acc[acc >= 0]
    cnt: int = len(valid_acc)
    avg_acc: float = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt


def pose_pck_accuracy(
    output: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    thr: float = 0.05,
    normalize: np.ndarray | None = None,
) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint.

    And the averaged accuracy across all keypoints from heatmaps.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    batch_size, num_keypoints, y_dim, x_dim = output.shape
    if num_keypoints == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[y_dim, x_dim]]), (batch_size, 1))

    pred, _ = get_heatmap_maximum(output)
    gt, _ = get_heatmap_maximum(target)
    return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)


def simcc_pck_accuracy(
    output: tuple[np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray],
    simcc_split_ratio: float,
    mask: np.ndarray,
    thr: float = 0.05,
    normalize: np.ndarray | None = None,
) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint.

    And the averaged accuracy across all keypoints from SimCC.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        output (Tuple[np.ndarray, np.ndarray]): Model predicted SimCC.
        target (Tuple[np.ndarray, np.ndarray]): Groundtruth SimCC.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    pred_x, pred_y = output
    gt_x, gt_y = target

    batch_size, _, x_dim = pred_x.shape
    _, _, y_dim = pred_y.shape
    x_dim, y_dim = int(x_dim / simcc_split_ratio), int(y_dim / simcc_split_ratio)

    if normalize is None:
        normalize = np.tile(np.array([[y_dim, x_dim]]), (batch_size, 1))

    pred_coords, _ = get_simcc_maximum(pred_x, pred_y)
    pred_coords /= simcc_split_ratio
    gt_coords, _ = get_simcc_maximum(gt_x, gt_y)
    gt_coords /= simcc_split_ratio

    return keypoint_pck_accuracy(pred_coords, gt_coords, mask, thr, normalize)


def refine_simcc_dark(keypoints: np.ndarray, simcc: np.ndarray, blur_kernel_size: int) -> np.ndarray:
    """SimCC version.

    Refine keypoint predictions using distribution aware coordinate decoding for UDP.
    See `UDP`_ for details. The operation is inplace.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        simcc (np.ndarray): The heatmaps in shape (N, K, Wx)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    batch_size = simcc.shape[0]

    # modulate simcc
    simcc = gaussian_blur1d(simcc, blur_kernel_size)
    np.clip(simcc, 1e-3, 50.0, simcc)
    np.log(simcc, simcc)

    simcc = np.pad(simcc, ((0, 0), (0, 0), (2, 2)), "edge")

    for n in range(batch_size):
        px = (keypoints[n] + 2.5).astype(np.int64).reshape(-1, 1)  # K, 1

        dx0 = np.take_along_axis(simcc[n], px, axis=1)  # K, 1
        dx1 = np.take_along_axis(simcc[n], px + 1, axis=1)
        dx_1 = np.take_along_axis(simcc[n], px - 1, axis=1)
        dx2 = np.take_along_axis(simcc[n], px + 2, axis=1)
        dx_2 = np.take_along_axis(simcc[n], px - 2, axis=1)

        dx = 0.5 * (dx1 - dx_1)
        dxx = 1e-9 + 0.25 * (dx2 - 2 * dx0 + dx_2)

        offset = dx / dxx
        keypoints[n] -= offset.reshape(-1)

    return keypoints


def gaussian_blur1d(simcc: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate simcc distribution with Gaussian.

    Note:
        - num_keypoints: K
        - simcc length: Wx

    Args:
        simcc (np.ndarray[K, Wx]): model predicted simcc.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the simcc gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, Wx]): Modulated simcc distribution.
    """
    if kernel % 2 != 1:
        msg = f"Kernel size should be odd, but got {kernel}."
        raise ValueError(msg)

    border = (kernel - 1) // 2
    batch_size, num_keypoints, x_dim = simcc.shape

    for n, k in product(range(batch_size), range(num_keypoints)):
        origin_max = np.max(simcc[n, k])
        dr = np.zeros((1, x_dim + 2 * border), dtype=np.float32)
        dr[0, border:-border] = simcc[n, k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, 1), 0)
        simcc[n, k] = dr[0, border:-border].copy()
        simcc[n, k] *= origin_max / np.max(simcc[n, k])
    return simcc
