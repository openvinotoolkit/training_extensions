import numpy as np


def get_iou(a, b):
    tl = [max(a[0], b[0]), max(a[1], b[1])]
    br = [min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])]

    area_intersection = max(0, br[0] - tl[0]) * max(0, br[1] - tl[1])
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]

    iou = 0
    if area_intersection > 0:
        iou = area_intersection / (area_a + area_b - area_intersection)
    return iou


def propagate_person_id(previous_poses, current_poses, threshold=0.5):
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose_id in range(len(current_poses)):
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = -1
        for previous_pose_id in range(len(previous_poses)):
            if not mask[previous_pose_id]:
                continue
            iou = get_iou(current_poses[current_pose_id].bbox, previous_poses[previous_pose_id].bbox)
            if (iou > threshold
                    and iou > best_matched_iou):
                best_matched_iou = iou
                best_matched_pose_id = previous_poses[previous_pose_id].id
                best_matched_id = previous_pose_id
        if best_matched_iou > -1:
            mask[best_matched_id] = 0
        current_poses[current_pose_id].update_id(best_matched_pose_id)
