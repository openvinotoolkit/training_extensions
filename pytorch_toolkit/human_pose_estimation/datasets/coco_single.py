import os
import json
import copy

import cv2
import numpy as np
from torch.utils.data.dataset import Dataset

def preprocess_bbox(bbox, image):
    aspect_ratio = 0.75
    bbox[0] = np.max((0, bbox[0]))
    bbox[1] = np.max((0, bbox[1]))
    x2 = np.min((image.shape[1] - 1, bbox[0] + np.max((0, bbox[2] - 1))))
    y2 = np.min((image.shape[0] - 1, bbox[1] + np.max((0, bbox[3] - 1))))

    if x2 >= bbox[0] and y2 >= bbox[1]:
        bbox = [bbox[0], bbox[1], x2 - bbox[0], y2 - bbox[1]]

    cx_bbox = bbox[0] + bbox[2] * 0.5
    cy_bbox = bbox[1] + bbox[3] * 0.5
    center = np.array([np.float32(cx_bbox), np.float32(cy_bbox)])

    if bbox[2] > aspect_ratio * bbox[3]:
        bbox[3] = bbox[2] * 1.0 / aspect_ratio
    elif bbox[2] < aspect_ratio * bbox[3]:
        bbox[2] = bbox[3] * aspect_ratio

    s = np.array([bbox[2] / 200., bbox[3] / 200.], np.float32)
    scale = s * 1.25

    return center, scale

class CocoSingleTrainDataset(Dataset):

    right_keypoints_indice = [6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26, 30,
                                    31, 32, 36, 37, 38, 42, 43, 44, 48, 49, 50]
    left_keypoints_indice = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23, 27,
                                   28, 29, 33, 34, 35, 39, 40, 41, 45, 46, 47]

    def __init__(self, dataset_folder, input_size_img=(288, 384), stride=(8, 8), sigma=3, transform=None):
        super().__init__()
        self._num_keypoints = 17
        self._dataset_folder = dataset_folder
        self._input_size_img = input_size_img
        self._stride = stride
        self._sigma = sigma
        self._transform = transform
        self._aspect_ratio = self._input_size_img[0] / self._input_size_img[1]
        self._heatmap_size = [self._input_size_img[0] // self._stride[0], self._input_size_img[1] // self._stride[1]]
        with open(os.path.join(self._dataset_folder, 'annotations', 'person_keypoints_train2017_converted.json')) as f:
            self._labels = json.load(f)

    def __getitem__(self, idx):
        image_path = self._labels['annotations'][idx]['image_path']
        image = cv2.imread(os.path.join(self._dataset_folder, 'train2017', image_path), cv2.IMREAD_COLOR)
        tokens = self._labels['annotations'][idx]['keypoints']
        bbox = copy.deepcopy(self._labels['annotations'][idx]['bbox'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        center, scale = preprocess_bbox(bbox, image)
        rotate = 0

        keypoints = np.zeros(self._num_keypoints * 3, dtype=np.float32)

        for id in range(self._num_keypoints):
            if tokens[id * 3] != 0:
                keypoints[id * 3] = int(tokens[id*3])          # x
                keypoints[id * 3 + 1] = int(tokens[id*3 + 1])  # y
                if tokens[id * 3 + 2] > 0:
                    keypoints[id * 3 + 2] = 1  # visible == 1, not visible == 0

        sample = {
            'keypoints': keypoints,
            'image': image,
            'center': center,
            'scale': scale,
            'rotate': rotate
        }

        if self._transform:
            sample = self._transform(sample)

        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps

        sample['image'] = sample['image'].transpose(2, 0, 1)

        return sample

    def __len__(self):
        return len(self._labels['annotations'])

    def _generate_keypoint_maps(self, sample):
        keypoints = sample['keypoints']
        target = np.zeros((len(keypoints) // 3, self._heatmap_size[1], self._heatmap_size[0]), dtype=np.float32)
        eps = self._sigma * self._sigma
        for i in range(self._num_keypoints):
            if keypoints[i * 3 + 2] > 0:  # visible
                x = int(keypoints[i * 3] / self._stride[0] + 0.5)
                y = int(keypoints[i * 3 + 1] / self._stride[1] + 0.5)
                lt = [x - eps, y - eps]
                rb = [x + eps + 1, y + eps + 1]
                if lt[0] >= self._heatmap_size[0] or lt[1] >= self._heatmap_size[1] \
                        or rb[0] < 0 or rb[1] < 0:
                    keypoints[i * 3 + 2] = 0
                    continue
                grid_x = np.arange(0, 2 * eps + 1, 1, np.float32)
                grid_y = grid_x[:, None]
                x0 = (2 * eps + 1) // 2
                y0 = x0
                gaussian = np.exp(- ((grid_x - x0) ** 2 + (grid_y - y0) ** 2) / (2 * self._sigma ** 2))
                lt_heatmap = [max(0, lt[0]), max(0, lt[1])]
                rb_heatmap = [min(rb[0], self._heatmap_size[0]), min(rb[1], self._heatmap_size[1])]

                lt_gaussian = [max(0, -lt[0]), max(0, -lt[1])]
                rb_gaussian = [min(rb[0], self._heatmap_size[0]) - lt[0], min(rb[1], self._heatmap_size[1]) - lt[1]]

                if keypoints[i * 3 + 1] > 0.5:
                    target[i][lt_heatmap[1]:rb_heatmap[1], lt_heatmap[0]:rb_heatmap[0]] = \
                        gaussian[lt_gaussian[1]:rb_gaussian[1], lt_gaussian[0]:rb_gaussian[0]]

        return target

class CocoSingleValDataset(Dataset):
    def __init__(self, dataset_folder, num_images=None, transform=None):
        super().__init__()
        self._num_keypoints = 17
        self._dataset_folder = dataset_folder
        self._transform = transform
        with open(os.path.join(self._dataset_folder, 'annotations', 'person_keypoints_val2017.json')) as f:
            data = json.load(f)

        self._annotations = data['annotations']

        if num_images is not None:
            self._annotations = data['annotations'][:num_images]

        self._images = data['images']

    def __getitem__(self, idx):
        image_path = next(it['file_name'] for it in self._images if it['id'] == self._annotations[idx]['image_id'])
        image = cv2.imread(os.path.join(self._dataset_folder, 'val2017', image_path), cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = self._annotations[idx]['bbox']
        center, scale = preprocess_bbox(bbox, image)
        rotate = 0

        sample = {
            'image': image,
            'image_id': self._annotations[idx]['image_id'],
            'bbox': bbox,
            'center': center,
            'scale': scale,
            'rotate': rotate
        }

        if self._transform:
            sample = self._transform(sample)

        sample['image'] = sample['image'].transpose(2, 0, 1)
        return sample

    def __len__(self):
        return len(self._annotations)
