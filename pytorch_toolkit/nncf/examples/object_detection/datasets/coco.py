"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
from collections import OrderedDict
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data

from examples.common.example_logger import logger

COCO_CLASSES = (  # always index 0
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

COCO_NAMES = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22",
    "23", "24", "25", "27", "28", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44",
    "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64",
    "65", "67", "70", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "84", "85", "86", "87", "88",
    "89", "90"
)

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


def _read_coco_annotation(annotation_file, images_folder):
    images_folder = Path(images_folder)
    anno_dict = OrderedDict()

    with open(annotation_file) as data_file:
        json_annotation = json.load(data_file)
    annotation = json_annotation["annotations"]

    for imgAnnotation in annotation:
        img_path = images_folder / "{0:012d}.jpg".format(imgAnnotation['image_id'])

        name = str(imgAnnotation["category_id"])
        label_idx = COCO_NAMES.index(name)
        bbox = imgAnnotation["bbox"]

        if bbox is None or bbox == "":
            raise ValueError("No annotation for {}".format(img_path))

        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        anno_dict.setdefault(img_path.as_posix(), []).append({'bbox': bbox, 'label_idx': label_idx})

    return anno_dict


class COCODataset(data.Dataset):
    classes = COCO_CLASSES
    name = 'coco'

    def __init__(self, annotation_file, images_folder, transform=None, target_transform=None, scale_bboxes=True,
                 return_image_info=False, rgb=True):
        self.rgb = rgb
        self.target_transform = target_transform
        self.return_image_info = return_image_info
        self.annotationFile = annotation_file
        self.imagesFolder = images_folder
        self.transform = transform
        self.scale_bboxes = scale_bboxes
        self.annotation = _read_coco_annotation(annotation_file, images_folder)

    def __getitem__(self, index):
        """
        Returns image at index in torch tensor form (RGB) and
        corresponding normalized annotation in 2d array [[xmin, ymin, xmax, ymax, label_ind],
                                                         ... ]
        """
        im, gt, h, w = self.pull_item(index)
        if self.return_image_info:
            return im, gt, h, w
        return im, gt

    def __len__(self):
        return len(self.annotation)

    def pull_item(self, index):
        """
        Returns image at index in torch tensor form (RGB),
        corresponding normalized annotation in 2d array [[xmin, ymin, xmax, ymax, label_ind],
                                                          ... ],
        height and width of image
        """
        img_path = list(self.annotation.keys())[index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None
        height, width, _ = img.shape

        boxes = np.asarray([anno['bbox'] for anno in self.annotation[img_path]])
        labels = np.asarray([anno['label_idx'] for anno in self.annotation[img_path]])

        if self.scale_bboxes:
            boxes /= np.array([width, height, width, height])

        if not boxes.size:
            logger.error("error: no annotation on image")
            sys.exit(-1)

        if self.target_transform is not None:
            annotation = self.target_transform(self.annotation, width, height)

        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)
            if self.rgb:
                img = img[:, :, (2, 1, 0)]
            annotation = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), annotation, height, width

    def pull_image(self, index):
        return cv2.imread(list(self.annotation.keys())[index], cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        """Returns the original annotation of image at index

        eg: ['001718', [[xmin, ymin, xmax, ymax, label_ind], ... ]]
        """
        img_path = list(self.annotation.keys())[index]
        return img_path[img_path.rfind("/") + 1: img_path.rfind(".")], self.annotation[img_path]

    def get_img_names(self):
        img_names = []
        for full_name in self.annotation.keys():
            img_names.append(full_name[full_name.rfind("/") + 1: full_name.rfind(".")])
        return img_names
