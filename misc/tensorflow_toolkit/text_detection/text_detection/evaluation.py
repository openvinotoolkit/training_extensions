# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

""" This module contains evaluation procedure. """

import numpy as np
import cv2
import Polygon as plg

IOU_CONSTRAINT = 0.5
AREA_PRECISION_CONSTRAINT = 0.5


def polygon_from_points(points):
    """ Returns a Polygon object to use with the Polygon2 class from a list of 8 points:
        x1,y1,x2,y2,x3,y3,x4,y4
    """

    point_mat = np.array(points[:8]).astype(np.int32).reshape(4, 2)
    return plg.Polygon(point_mat)


def draw_gt_polygons(image, gt_polygons, gt_dont_care_nums):
    """ Draws groundtruth polygons on image. """

    for point_idx, polygon in enumerate(gt_polygons):
        color = (128, 128, 128) if point_idx in gt_dont_care_nums else (255, 0, 0)
        for i in range(4):
            pt1 = int(polygon[0][i][0]), int(polygon[0][i][1])
            pt2 = int(polygon[0][(i + 1) % 4][0]), int(polygon[0][(i + 1) % 4][1])
            cv2.line(image, pt1, pt2, color, 2)
    return image


def draw_pr_polygons(image, pr_polygons, pr_dont_care_nums, pr_matched_nums, pr_confidences_list):
    """ Draws predicted polygons on image. """

    for point_idx, _ in enumerate(pr_polygons):
        if pr_confidences_list[point_idx] > 0.25:
            polygon = pr_polygons[point_idx]
            color = (0, 0, 255)
            if point_idx in pr_dont_care_nums:
                color = (255, 255, 255)
            if point_idx in pr_matched_nums:
                color = (0, 255, 0)
            for i in range(4):
                pt1 = int(polygon[0][i][0]), int(polygon[0][i][1])
                pt2 = int(polygon[0][(i + 1) % 4][0]), int(polygon[0][(i + 1) % 4][1])
                cv2.line(image, pt1, pt2, color, 2)
    return image


def get_union(polygon1, polygon2):
    """ Returns area of union of two polygons. """

    return polygon1.area() + polygon2.area() - get_intersection(polygon1, polygon2)


def get_intersection_over_union(polygon1, polygon2):
    """ Returns intersection over union of two polygons. """

    union = get_union(polygon1, polygon2)
    return get_intersection(polygon1, polygon2) / union if union else 0.0


def get_intersection(polygon1, polygon2):
    """ Returns are of intersection of two polygons. """

    intersection = polygon1 & polygon2
    if len(intersection) == 0:
        return 0
    return intersection.area()


def compute_ap(conf_list, match_list, num_gt_care):
    """ Returns average precision metrics. """

    correct = 0
    average_precision = 0
    if conf_list:
        conf_list = np.array(conf_list)
        match_list = np.array(match_list)
        sorted_ind = np.argsort(-conf_list)
        match_list = match_list[sorted_ind]

        for idx, matched in enumerate(match_list):
            if matched:
                correct += 1
                average_precision += float(correct) / (idx + 1)

        if num_gt_care > 0:
            average_precision /= num_gt_care

    return average_precision


def parse_gt_objects(gt_annotation):
    """ Parses groundtruth objects from annotation. """

    gt_polygons_list = []
    gt_dont_care_polygon_nums = []
    for gt_object in gt_annotation:
        polygon = polygon_from_points(gt_object['points'])
        gt_polygons_list.append(polygon)
        if gt_object['transcription'] == '###':
            gt_dont_care_polygon_nums.append(len(gt_polygons_list) - 1)

    return gt_polygons_list, gt_dont_care_polygon_nums


def parse_pr_objects(pr_annotation):
    """ Parses predicted objects from annotation. """

    pr_polygons_list = []
    pr_confidences_list = []
    for pr_object in pr_annotation:
        polygon = polygon_from_points(pr_object['points'])
        pr_polygons_list.append(polygon)
        pr_confidences_list.append(pr_object['confidence'])

    return pr_polygons_list, pr_confidences_list


def match_dont_care_objects(gt_polygons_list, gt_dont_care_polygon_nums, pr_polygons_list):
    """ Matches ignored objects. """

    pr_dont_care_polygon_nums = []

    if gt_dont_care_polygon_nums:
        for pr_polygon_idx, pr_polygon in enumerate(pr_polygons_list):
            for dont_care_polygon_num in gt_dont_care_polygon_nums:
                intersected_area = get_intersection(gt_polygons_list[dont_care_polygon_num],
                                                    pr_polygon)
                pd_dimensions = pr_polygon.area()
                precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                if precision > AREA_PRECISION_CONSTRAINT:
                    pr_dont_care_polygon_nums.append(pr_polygon_idx)
                    break

    return pr_dont_care_polygon_nums


def match(gt_polygons_list, gt_dont_care_polygon_nums, pr_polygons_list, pr_dont_care_polygon_nums):
    """ Matches all objects. """

    pr_matched_nums = []

    output_shape = [len(gt_polygons_list), len(pr_polygons_list)]
    iou_mat = np.empty(output_shape)
    gt_rect_mat = np.zeros(len(gt_polygons_list), np.int8)
    pr_rect_mat = np.zeros(len(pr_polygons_list), np.int8)
    for gt_idx, gt_polygon in enumerate(gt_polygons_list):
        for pr_idx, pr_polygon in enumerate(pr_polygons_list):
            iou_mat[gt_idx, pr_idx] = get_intersection_over_union(gt_polygon, pr_polygon)

    for gt_idx, _ in enumerate(gt_polygons_list):
        for pr_idx, _ in enumerate(pr_polygons_list):
            if gt_rect_mat[gt_idx] == 0 and pr_rect_mat[pr_idx] == 0 \
                    and gt_idx not in gt_dont_care_polygon_nums \
                    and pr_idx not in pr_dont_care_polygon_nums:
                if iou_mat[gt_idx, pr_idx] > IOU_CONSTRAINT:
                    gt_rect_mat[gt_idx] = 1
                    pr_rect_mat[pr_idx] = 1
                    pr_matched_nums.append(pr_idx)

    return pr_matched_nums


def eval(pr_annotations, gt_annotations, images=None, imshow_delay=1):
    """ Annotation format:
        {"image_path": [
                            {"points": [x1,y1,x2,y2,x3,y3,x4,y4],
                             "confidence": float,
                             "transcription", str}
                        ],
         "image_path": [points],

         ### - is a transcription of non-valid word.

    """

    assert len(pr_annotations) == len(gt_annotations)

    matched_sum = 0
    num_global_care_gt = 0
    num_global_care_pr = 0

    arr_global_confidences = []
    arr_global_matches = []

    for frame_id, _ in enumerate(pr_annotations):
        gt_polygons_list, gt_dont_care_polygon_nums = parse_gt_objects(gt_annotations[frame_id])
        pr_polygons_list, pr_confidences_list = parse_pr_objects(pr_annotations[frame_id])

        pr_dont_care_polygon_nums = match_dont_care_objects(
            gt_polygons_list, gt_dont_care_polygon_nums, pr_polygons_list)

        pr_matched_nums = []

        if gt_polygons_list and pr_polygons_list:
            pr_matched_nums = match(gt_polygons_list, gt_dont_care_polygon_nums, pr_polygons_list,
                                    pr_dont_care_polygon_nums)

            matched_sum += len(pr_matched_nums)

            for pr_num in range(len(pr_polygons_list)):
                if pr_num not in pr_dont_care_polygon_nums:
                    # we exclude the don't care detections
                    matched = pr_num in pr_matched_nums
                    arr_global_confidences.append(pr_confidences_list[pr_num])
                    arr_global_matches.append(matched)

        num_global_care_gt += len(gt_polygons_list) - len(gt_dont_care_polygon_nums)
        num_global_care_pr += len(pr_polygons_list) - len(pr_dont_care_polygon_nums)

        if images is not None:
            image = images[frame_id]
            draw_gt_polygons(image, gt_polygons_list, gt_dont_care_polygon_nums)
            draw_pr_polygons(image, pr_polygons_list, pr_dont_care_polygon_nums,
                             pr_matched_nums, pr_confidences_list)
            cv2.imshow('result', image)
            k = cv2.waitKey(imshow_delay)
            if k == 27:
                return -1, -1, -1

    method_recall = 0 if num_global_care_gt == 0 else float(matched_sum) / num_global_care_gt
    method_precision = 0 if num_global_care_pr == 0 else float(matched_sum) / num_global_care_pr
    denominator = method_precision + method_recall
    method_hmean = 0 if denominator == 0 else 2.0 * method_precision * method_recall / denominator

    average_precision = compute_ap(arr_global_confidences, arr_global_matches, num_global_care_gt)

    return method_recall, method_precision, method_hmean, average_precision
