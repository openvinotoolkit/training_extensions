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
import os.path as osp
import tempfile
from collections import OrderedDict

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .instance_dataset import InstanceDataset


class COCODataset(InstanceDataset):
    def __init__(self, root_dir_path, ann_file_path, with_gt, remove_images_without_gt, transforms=None):
        super().__init__(with_gt)

        self.root_dir_path = root_dir_path
        self.coco = COCO(ann_file_path)
        self.ids = list(self.coco.imgs.keys())
        self.ids = sorted(self.ids)
        if remove_images_without_gt:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0
            ]

        self.transforms = transforms

        # Set up dataset classes
        category_ids = self.coco.getCatIds()
        categories = [c['name'] for c in self.coco.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.classes_num = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.contiguous_category_id_to_json_id[0] = 0

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anno = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        original_image = cv2.imread(osp.join(self.root_dir_path, path))
        original_image_size = original_image.shape[:2]
        h, w = original_image_size

        sample = {'image': original_image, }
        if self.with_gt:
            boxes = [obj['bbox'] for obj in anno]
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if len(boxes) > 0:
                boxes[:, 2] += boxes[:, 0] - 1
                boxes[:, 3] += boxes[:, 1] - 1
                boxes = np.clip(boxes, 0, [w - 1, h - 1, w - 1, h - 1])
                sample['gt_boxes'] = boxes

            classes = [obj['category_id'] for obj in anno]
            classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
            classes = np.asarray(classes)
            assert len(classes) == len(boxes)
            if len(classes) > 0:
                sample['gt_classes'] = classes

            polygons = [obj['segmentation'] for obj in anno]
            polygons = [[np.clip(np.asarray(part, dtype=np.float32).reshape(-1, 2), 0, [[w - 1, h - 1]]) for part in obj]
                        for obj in polygons]
            assert len(polygons) == len(boxes)
            if len(polygons) > 0:
                sample['gt_masks'] = polygons

        if self.transforms is not None:
            sample = self.transforms(sample)

        IM_HEIGHT = 1
        IM_WIDTH = 2
        out_sample = dict(index=index,
                          original_image=original_image,
                          meta=dict(original_size=original_image_size,
                                    processed_size=sample['image'].shape[1:3]),
                          im_data=sample['image'],
                          im_info=torch.as_tensor([sample['image'].shape[IM_HEIGHT],
                                                   sample['image'].shape[IM_WIDTH],
                                                   1.0],
                                                  dtype=torch.float32))
        if self.with_gt:
            out_sample.update(dict(gt_boxes=sample['gt_boxes'],
                                   gt_labels=sample['gt_classes'],
                                   gt_masks=sample['gt_masks'],
                                   gt_is_ignored=torch.zeros(sample['gt_classes'].shape, dtype=torch.int32)
                                   ))
        return out_sample

    def get_image_info(self, idx):
        image_id = self.ids[idx]
        image_data = self.coco.imgs[image_id]
        return image_data

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {:,}\n'.format(self.__len__())
        fmt_str += '    Number of classes: {}\n'.format(self.classes_num)
        tmp = '    Transforms: '
        fmt_str += '{}{}\n'.format(tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def convert_predictions_to_coco_format(self, classes, scores, boxes, masks):

        def convert_box(b):
            x1, y1, x2, y2 = b
            return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

        coco_predictions = []
        for i, (im_classes, im_scores, im_boxes, im_masks) in enumerate(zip(classes, scores, boxes, masks)):
            image_id = self.ids[i]

            if len(im_classes) == 0:
                continue

            im_boxes = im_boxes.tolist()
            im_scores = im_scores.tolist()
            im_classes = im_classes.tolist()
            im_classes_ids = [self.contiguous_category_id_to_json_id[j] for j in im_classes]
            encoded_masks = []
            for mask in im_masks:
                if 'counts' in mask:
                    encoded_masks.append(mask)
                else:
                    encoded_mask = mask_util.encode(np.array(mask[:, :, np.newaxis].astype(np.uint8), order='F'))[0]
                    encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')
                    encoded_masks.append(encoded_mask)

            coco_predictions.extend(
                [
                    {
                        'image_id': image_id,
                        'category_id': class_id,
                        'bbox': convert_box(box),
                        'segmentation': encoded_mask,
                        'score': score,
                    }
                    for class_id, score, box, encoded_mask in zip(im_classes_ids, im_scores, im_boxes, encoded_masks)
                ]
            )
        return coco_predictions

    def evaluate(self, scores, classes, boxes, masks, output_dir='', iou_types=('bbox', 'segm')):
        predictions_coco = self.convert_predictions_to_coco_format(classes, scores, boxes, masks)

        results = COCOResults(*iou_types)
        for iou_type in iou_types:
            with tempfile.NamedTemporaryFile() as f:
                file_path = f.name
                if output_dir:
                    file_path = osp.join(output_dir, iou_type + '.json')
                res = evaluate_predictions_on_coco(self.coco, predictions_coco, file_path, iou_type)
                results.update(res)
        flat_results = {}
        for iou_type, metrics in results.results.items():
            for k, v in metrics.items():
                flat_results['{}/{}'.format(iou_type, k)] = v
        return flat_results


def evaluate_predictions_on_coco(coco_gt, coco_results, json_result_file, iou_type='bbox'):
    with open(json_result_file, 'w') as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


class COCOResults(object):
    METRICS = {
        'bbox': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'segm': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
    }

    def __init__(self, *iou_types):
        allowed_types = ('bbox', 'segm')
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        return repr(self.results)
