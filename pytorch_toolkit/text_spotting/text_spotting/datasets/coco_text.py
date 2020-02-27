"""
 Copyright (c) 2020 Intel Corporation

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

import os
import os.path as osp

import cv2
import numpy as np

import torch
from segmentoly.datasets.coco import COCODataset
from .text_evaluation import text_eval, masks_to_rects


class COCOTextDataset(COCODataset):
    """ Class for representing text dataset in MSCOCO format. """

    def __init__(self, root_dir_path, ann_file_path, with_gt, remove_images_without_gt,
                 transforms=None, remove_images_without_text=False, alphabet_decoder=None):
        super().__init__(root_dir_path, ann_file_path, with_gt, remove_images_without_gt,
                         transforms)

        self.remove_images_without_text = remove_images_without_text
        self.alphabet_decoder = alphabet_decoder
        if self.remove_images_without_text:
            self._remove_images_wihout_text()

        self.gt_annotations = []
        self.file_names = []
        for index, _ in enumerate(self.ids):
            ann_ids = self.coco.getAnnIds(imgIds=self.ids[index], iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            path = self.coco.loadImgs(self.ids[index])[0]['file_name']
            self.file_names.append(osp.join(self.root_dir_path, path))

            gt_annotation = []
            for obj in anno:
                gt_annotation.append(
                    {'points': obj['segmentation'][0],
                     'transcription': '###' if obj['iscrowd'] else obj['text']['transcription']})

            self.gt_annotations.append(gt_annotation)

    def _remove_images_wihout_text(self):
        ids_with_text = []
        for index, _ in enumerate(self.ids):
            img_id = self.ids[index]

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)

            texts = [obj.get('text', {}).get('transcription', "") for obj in anno]

            if self._any_text_valid(texts, self.alphabet_decoder):
                ids_with_text.append(img_id)

        self.ids = ids_with_text

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anno = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        original_image = cv2.imread(osp.join(self.root_dir_path, path))

        if original_image is None:
            raise Exception(f'Failed to read: {osp.join(self.root_dir_path, path)}')

        height, width = original_image.shape[:2]

        sample = {'image': original_image, 'path': osp.join(self.root_dir_path, path)}
        if self.with_gt:
            boxes = [obj['bbox'] for obj in anno]
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if len(boxes) > 0:
                boxes[:, 2] += boxes[:, 0] - 1
                boxes[:, 3] += boxes[:, 1] - 1
                boxes = np.clip(boxes, 0, [width - 1, height - 1, width - 1, height - 1])
                sample['gt_boxes'] = boxes

            classes = [obj['category_id'] for obj in anno]
            classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
            classes = np.asarray(classes)
            assert len(classes) == len(boxes)
            if len(classes) > 0:
                sample['gt_classes'] = classes

            polygons = [obj['segmentation'] for obj in anno]
            polygons = [
                [np.clip(np.asarray(part, dtype=np.float32).reshape(-1, 2), 0,
                         [[width - 1, height - 1]])
                 for part in obj]
                for obj in polygons]
            assert len(polygons) == len(boxes)
            if len(polygons) > 0:
                sample['gt_masks'] = polygons

            gt_texts = [obj.get('text', {}).get('transcription', "") for obj in anno]
            assert len(gt_texts) == len(boxes)
            if len(gt_texts) > 0:
                sample['gt_texts'] = gt_texts

        if self.transforms is not None:
            sample = self.transforms(sample)

        out_sample = dict(index=index,
                          original_image=original_image,
                          meta=dict(original_size=[height, width],
                                    processed_size=sample['image'].shape[1:3]),
                          im_data=sample['image'],
                          im_info=torch.as_tensor([sample['image'].shape[1],
                                                   sample['image'].shape[2],
                                                   1.0],
                                                  dtype=torch.float32),
                          path=sample['path'])
        if self.with_gt:
            out_sample.update(dict(gt_boxes=sample['gt_boxes'],
                                   gt_labels=sample['gt_classes'],
                                   gt_masks=sample['gt_masks'],
                                   gt_is_ignored=torch.zeros(sample['gt_classes'].shape,
                                                             dtype=torch.int32),
                                   gt_texts=sample['gt_texts'],
                                   gt_texts_str=sample.get('gt_texts_str')
                                   ))

        return out_sample

    def prepare_predictions(self, masks, text_log_softmax):
        """
        Transforms masks to polygons and decodes encoded text making predictions suitable for
        evaluation.
        """

        predictions = []
        for i, image_masks in enumerate(masks):
            rectangles = []
            if image_masks:
                rectangles = masks_to_rects(image_masks, is_rle=True)
                rectangles = [{'points': bbox.reshape([-1]), 'confidence': 1.0} for bbox in
                              rectangles]
                if text_log_softmax:
                    if isinstance(text_log_softmax[i], np.ndarray):
                        texts = text_log_softmax[i]
                    else:
                        texts = text_log_softmax[i].cpu().numpy()
                    encoded_transcriptions = np.argmax(texts, 2)
                    transcriptions = self.alphabet_decoder.batch_decode(encoded_transcriptions)
                    for rectangle, transcription in zip(rectangles, transcriptions):
                        rectangle['transcription'] = transcription

            predictions.append(rectangles)

        return predictions

    def dump_predictions(self, folder, predictions):
        """
        Dumps prediction to folder as per image text files.
        :param folder: dump folder
        :param predictions: list of predictions
        :return:
        """
        os.makedirs(folder, exist_ok=True)
        for image_path, pr_annotation in zip(self.file_names, predictions):
            with open(osp.join(folder, f'res_{osp.basename(image_path)[:-3]}txt'), 'w') as file:
                for obj in pr_annotation:
                    file.write(','.join([str(c) for c in obj['points']]))
                    if 'transcription' in obj.keys():
                        transcription = obj['transcription']
                        file.write(f',{transcription}')
                    file.write('\n')

    def evaluate(self, scores, classes, boxes, masks, text_log_softmax=None, output_dir='',
                 iou_types=('bbox', 'segm'), dump=None, visualize=False):
        """
        Runs COCO procedure to evaluate mAP of detection and segmentation as well as
        ICDAR procedure to evaluate text detection and word spotting recall, recall and
        their harmonic mean (hmean).
        """
        flat_results = super().evaluate(scores, classes, boxes, masks, output_dir, iou_types)

        predictions = self.prepare_predictions(masks, text_log_softmax)

        recall, precision, hmean, _ = text_eval(predictions, self.gt_annotations,
                                                use_transcriptions=False)
        print(' Text detection recall={} precision={} hmean={}'.format(recall, precision, hmean))
        flat_results['text_detection/recall'] = recall
        flat_results['text_detection/precision'] = precision
        flat_results['text_detection/hmean'] = hmean

        if text_log_softmax:
            paths = []
            if visualize:
                for img_id in self.ids:
                    path = self.coco.loadImgs(img_id)[0]['file_name']
                    paths.append(osp.join(self.root_dir_path, path))
                assert len(predictions) == len(self.gt_annotations)
                assert len(predictions) == len(paths), f'len(predictions)={len(predictions)}, len(paths)={len(paths)}'

            recall, precision, hmean, _ = text_eval(predictions, self.gt_annotations, paths,
                                                    imshow_delay=0, use_transcriptions=True)

            print(
                ' Text spotting  recall={} precision={} hmean={}'.format(recall, precision, hmean))
            flat_results['text_spotting/recall'] = recall
            flat_results['text_spotting/precision'] = precision
            flat_results['text_spotting/hmean'] = hmean

        if dump:
            self.dump_predictions(dump, predictions)

        return flat_results

    @staticmethod
    def _any_text_valid(texts, alphabet_decoder=None):
        if not texts:
            return False
        if all((not t) for t in texts):
            return False
        if alphabet_decoder is None:
            return True
        return any(alphabet_decoder.string_in_alphabet(t) for t in texts)
