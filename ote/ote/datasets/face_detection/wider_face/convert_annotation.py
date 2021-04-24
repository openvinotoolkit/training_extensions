# pylint: disable=C0301,W0622,R0914,R0801

import json
import os

import imagesize # pylint: disable=import-error
from tqdm import tqdm


def parse_wider_gt(ann_file):
    """ Parses wider annotation. """

    bboxes = dict()
    landmarks = dict()
    with open(ann_file) as read_file:
        content = [line.strip() for line in read_file]
        new_file = True
        i = 0
        while True:
            if new_file:
                image_name = content[i]
                bboxes[image_name] = list()
                landmarks[image_name] = list()
                new_file = False
                i += 1
            else:
                bbox_num = int(content[i])
                if bbox_num == 0:
                    i += 1
                i += 1
                for _ in range(bbox_num):
                    xmin, ymin, width, height = [int(x) for x in content[i].split(' ')[:4]]
                    if width >= 0 and height >= 0:
                        bboxes[image_name].append([xmin, ymin, width, height])
                        landmarks[image_name].append([])
                    else:
                        print('Ignored because of invalid size: ', [xmin, ymin, width, height])
                    i += 1
                if i == len(content):
                    break
                new_file = True

    return bboxes, landmarks


def parse_wider_gt_with_landmarks(ann_file):
    """ Parses wider annotation with landmarks. """

    bboxes = dict()
    landmarks = dict()
    with open(ann_file) as read_file:
        content = [line.strip() for line in read_file]
        new_file = True
        i = 0
        while True:
            if new_file:
                image_name = content[i][2:]
                bboxes[image_name] = list()
                landmarks[image_name] = list()
                new_file = False
                i += 1
            else:
                while True:
                    if i == len(content) or content[i].startswith('#'):
                        break
                    line_split = content[i].split(' ')
                    xmin, ymin, width, height = [int(x) for x in line_split[:4]]
                    if width >= 0 and height >= 0:
                        bboxes[image_name].append([xmin, ymin, width, height])
                        points = [float(x) if (i + 1) % 3 != 0 else float(x) + 1 for i, x in
                                  enumerate(line_split[4:-1])]
                        landmarks[image_name].append(points)
                    else:
                        print('Ignored because of invalid size: ', [xmin, ymin, width, height])
                    i += 1
                if i == len(content):
                    break
                new_file = True

    return bboxes, landmarks


def convert_to_coco(ann_file, data_dir, out_file, with_landmarks):
    """ Converts wider annotation to COCO format. """

    img_id = 0
    ann_id = 0
    cat_id = 1

    ann_dict = {}
    categories = [{"id": 1, "name": 'face'}]
    images_info = []
    annotations = []

    if with_landmarks:
        boxes, landmarks = parse_wider_gt_with_landmarks(ann_file)
    else:
        boxes, landmarks = parse_wider_gt(ann_file)

    for filename in tqdm(boxes.keys()):
        image_info = {}
        image_info['id'] = img_id
        img_id += 1
        image_info['width'], image_info['height'] = imagesize.get(os.path.join(data_dir, filename))
        image_info['file_name'] = os.path.relpath(
            os.path.join(data_dir, filename), os.path.dirname(out_file))
        images_info.append(image_info)

        for gt_bbox, gt_landmarks in zip(boxes[filename], landmarks[filename]):
            ann = {
                'id': ann_id,
                'image_id': image_info['id'],
                'segmentation': [],
                'keypoints': gt_landmarks,
                'category_id': cat_id,
                'iscrowd': 0,
                'area': gt_bbox[2] * gt_bbox[3],
                'bbox': gt_bbox
            }
            ann_id += 1
            annotations.append(ann)

    ann_dict['images'] = images_info
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, 'w') as outfile:
        outfile.write(json.dumps(ann_dict))
