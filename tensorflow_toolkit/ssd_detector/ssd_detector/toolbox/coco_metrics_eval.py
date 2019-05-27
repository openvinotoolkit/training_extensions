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

from pycocotools.cocoeval import COCOeval
from ssd_detector.readers.object_detector_json import ObjectDetectorJson

METRICS_NAMES = ["Average Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 100]",
                 "Average Precision(AP) @ [IoU = 0.50 | area = all | maxDets = 100]",
                 "Average Precision(AP) @ [IoU = 0.75 | area = all | maxDets = 100]",
                 "Average Precision(AP) @ [IoU = 0.50:0.95 | area = small | maxDets = 100]",
                 "Average Precision(AP) @ [IoU = 0.50:0.95 | area = medium | maxDets = 100]",
                 "Average Precision(AP) @ [IoU = 0.50:0.95 | area = large | maxDets = 100]",
                 "Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 1]",
                 "Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 10]",
                 "Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 100]",
                 "Average Recall(AR) @ [IoU = 0.50:0.95 | area = small | maxDets = 100]",
                 "Average Recall(AR) @ [IoU = 0.50:0.95 | area = medium | maxDets = 100]",
                 "Average Recall(AR) @ [IoU = 0.50:0.95 | area = large | maxDets = 100]"]


# pylint: disable=too-many-locals
def calc_coco_metrics(coco_annotations, predictions, classes):
  annotations = ObjectDetectorJson.convert_coco_to_toolbox_format(coco_annotations, classes)
  detections = []
  for annotation, prediction in zip(annotations, predictions):
    width, height = annotation['image_size']
    image_id = annotation['image_id']

    for obj_id, obj in enumerate(prediction):
      label = int(obj[1])
      score = float(obj[2])
      if obj_id != 0 and score == 0:  # At least one prediction must be (COCO API issue)
        continue
      bbox = (obj[3:]).tolist()
      bbox[::2] = [width * i for i in bbox[::2]]
      bbox[1::2] = [height * i for i in bbox[1::2]]

      xmin, ymin, xmax, ymax = bbox
      w_bbox = round(xmax - xmin, 1)
      h_bbox = round(ymax - ymin, 1)
      xmin, ymin = round(xmin, 1), round(ymin, 1)

      coco_det = {}
      coco_det['image_id'] = image_id
      coco_det['category_id'] = label
      coco_det['bbox'] = [xmin, ymin, w_bbox, h_bbox]
      coco_det['score'] = score
      detections.append(coco_det)

  coco_dt = coco_annotations.loadRes(detections)
  img_ids = sorted(coco_annotations.getImgIds())
  coco_eval = COCOeval(coco_annotations, coco_dt, 'bbox')
  coco_eval.params.imgIds = img_ids
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()

  metrics = {}
  for metric_name, value in zip(METRICS_NAMES, coco_eval.stats):
    metrics[metric_name] = value

  return metrics
