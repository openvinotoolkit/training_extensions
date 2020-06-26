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

""" This script uses output of test.py (mmdetection) to
    calculate precision, recall and f-mean of predictions."""

import argparse

import mmcv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.datasets import build_dataset
from mmdet.core.evaluation.coco_utils import results2json
from mmdet.core.evaluation.text_evaluation import text_eval

def parse_args():
    """ Parses input arguments. """
    parser = argparse.ArgumentParser(
        description='This script uses output of test.py (mmdetection) to '
                    'calculate precision, recall and f-mean of predictions.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('input', help='output result file from test.py')
    parser.add_argument('--draw_graph', action='store_true', help='draw histogram of recall')
    parser.add_argument('--visualize', action='store_true', help='show detection result on images')
    args = parser.parse_args()
    return args


def main():
    """ Main function. """

    args = parse_args()
    if args.input is not None and not args.input.endswith(('.pkl', '.pickle')):
        raise ValueError('The input file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)

    results = mmcv.load(args.input)
    result_file = results2json(dataset, results, args.input)

    coco = dataset.coco
    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    eval_type = 'bbox'
    result_file = result_file[eval_type]

    coco_dets = coco.loadRes(result_file)
    img_ids = coco.getImgIds()
    iou_type = 'bbox'
    cocoEval = COCOeval(coco, coco_dets, iou_type)
    cocoEval.params.imgIds = img_ids

    predictions = cocoEval.cocoDt.imgToAnns
    gt_annotations = cocoEval.cocoGt.imgToAnns

    if args.visualize:
        img_paths = [dataset.img_prefix + image['file_name']
                     for image in coco_dets.dataset['images']]
    else:
        img_paths = None

    recall, precision, hmean, _ = text_eval(
          predictions, gt_annotations,
          cfg.test_cfg.score_thr,
          images=img_paths,
          show_recall_graph=args.draw_graph)
    print('Text detection recall={:.4f} precision={:.4f} hmean={:.4f}'.
                      format(recall, precision, hmean))


if __name__ == '__main__':
    main()
