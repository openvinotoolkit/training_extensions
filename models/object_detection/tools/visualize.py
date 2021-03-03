# Copyright (C) 2020 Intel Corporation
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

import argparse
import cv2
import mmcv
import numpy as np
import sys
from mmcv.parallel import collate

from mmdet.apis.inference import LoadImage
from mmdet.core import encode_mask_results
from mmdet.core.bbox.transforms import bbox2result
from mmdet.core.mask.transforms import mask2result
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.datasets.pipelines import Compose


def postprocess(result, img_meta, num_classes=80, rescale=True):
    det_bboxes = result['boxes']
    det_labels = result['labels']

    if rescale:
        img_h, img_w = img_meta[0]['ori_shape'][:2]
        scale = img_meta[0]['scale_factor']
        det_bboxes[:, :4] /= scale
    else:
        img_h, img_w = img_meta[0]['img_shape'][:2]

    det_bboxes[:, 0:4:2] = np.clip(det_bboxes[:, 0:4:2], 0, img_w - 1)
    det_bboxes[:, 1:4:2] = np.clip(det_bboxes[:, 1:4:2], 0, img_h - 1)

    bbox_results = bbox2result(det_bboxes, det_labels, num_classes)
    return bbox_results

class VideoDataset:
    def __init__(self, path, cfg, device='cpu'):
        self.path = path
        self.video = cv2.VideoCapture(self.path)
        assert self.video.isOpened()
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cfg = cfg
        self.device = device

        # build the data pipeline
        self.test_pipeline = [LoadImage()] + self.cfg.test.pipeline[1:]
        self.test_pipeline = Compose(self.test_pipeline)

    def __getitem__(self, idx):
        status, img = self.video.read()
        if not status:
            self.video.release()
            raise StopIteration

        data = dict(img=img)
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        return img, data

    def __len__(self):
        return sys.maxsize


def main(args):
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    assert cfg.data.test.pipeline[1]['type'] == 'MultiScaleFlipAug'
    normalize_idx = [i for i, v in enumerate(cfg.data.test.pipeline[1]['transforms']) if v['type'] == 'Normalize'][0]
    cfg.data.test.pipeline[1]['transforms'][normalize_idx]['mean'] = [0.0, 0.0, 0.0]
    cfg.data.test.pipeline[1]['transforms'][normalize_idx]['std'] = [1.0, 1.0, 1.0]
    cfg.data.test.pipeline[1]['transforms'][normalize_idx]['to_rgb'] = False

    dataset = VideoDataset(int(args.video), cfg.data)
    data_loader = iter(dataset)
    wait_key = 1

    # Valid classes + background.
    classes_num = 2

    extra_args = {}
    from mmdet.utils.deployment.openvino_backend import DetectorOpenVINO as Model
    assert args.load_weights.endswith('.xml') or args.load_weights.endswith('.bin')
    model = Model(args.load_weights[:-3] + 'xml',
                  args.load_weights[:-3] + 'bin',
                  mapping_file_path=args.load_weights[:-3] + 'mapping',
                  cfg=cfg,
                  classes=('face', ),
                  **extra_args)

    GREEN_COLOR = (0, 255, 0)
    BLUE_COLOR = (255, 0, 0)
    results = []
    for i, (display_image, data) in enumerate(data_loader):
        im_data = data['img'][0].cpu().numpy()
        result = model(im_data)
        result = postprocess(
            result,
            data['img_metas'][0].data[0],
            num_classes=classes_num,
            rescale=True)

        for box in result[0]:
            if box[-1] > 0.3:
                p0, p1 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(display_image, p0, p1, GREEN_COLOR, 1)
                cv2.putText(display_image, 'face', p0, 1, 2, BLUE_COLOR, 2)
        cv2.imshow('image', display_image)
        if cv2.waitKey(1) == 27:
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Test model deployed to OpenVINO')
    parser.add_argument('--load-weights', help='Path to OpenVINO model.')
    parser.add_argument('--video', help='Web camera index.')
    parser.add_argument('--config', help=argparse.SUPPRESS, default='model.py')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
