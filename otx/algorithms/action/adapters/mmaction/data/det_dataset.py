"""Adapt AVADataset in mmaction2 into OTXDataset."""

# Copyright (C) 2022 Intel Corporation
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

import copy
import json
import os
from collections import defaultdict
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmaction.core.evaluation.ava_utils import det2csv
from mmaction.datasets.ava_dataset import AVADataset
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from mmaction.utils import get_root_logger
from mmcv.utils import print_log

from otx.algorithms.action.adapters.mmaction.utils import det_eval
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

root_logger = get_root_logger()


# pylint: disable=too-many-instance-attributes, too-many-arguments, invalid-name, super-init-not-called, too-many-locals
@DATASETS.register_module()
class OTXActionDetDataset(AVADataset):
    """Wrapper that allows using a OTX dataset to train action detection models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.
    It is adapted from AVADataset of mmaction, but it supports other dataset such as UCF and JHMDB.
    """

    class _DataInfoProxy:
        def __init__(self, otx_dataset, labels, fps, test_mode):
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}
            self.fps = fps
            self.data_root = self.otx_dataset[0].media.data_root

            if not test_mode:
                self.proposal_file = os.path.join(self.data_root, "train.pkl")
            else:
                self.proposal_file = os.path.join(self.data_root, "valid.pkl")
            if os.path.exists(self.proposal_file):
                self.proposals = mmcv.load(self.proposal_file)
                self.patch_proposals()
            else:
                self.proposals = None

        def __len__(self):
            return len(self.otx_dataset)

        def __getitem__(self, index):
            """Prepare a dict 'data_info' that is expected by the mmaction pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """

            dataset = self.otx_dataset
            item = dataset[index]
            shot_info = (0, (item.media.timestamp_end - item.media.timestamp_start)) * self.fps
            ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

            data_info = dict(
                **item.media,
                shot_info=shot_info,
                index=index,
                ann_info=dict(label_list=self.labels),
                ignored_labels=ignored_labels,
                fps=self._FPS,
            )

            anns = item.media.data.get_annotations()
            if len(anns) > 0:
                bboxes, labels = [], []
                for ann in anns:
                    bbox = np.asarray([ann.shape.x1, ann.shape.y1, ann.shape.x2, ann.shape.y2])
                    valid_labels = np.array([int(label.id) for label in ann.get_labels()], dtype=int)
                    label = np.zeros(len(self.labels) + 1, dtype=np.float32)
                    label[valid_labels] = 1.0
                    bboxes.append(bbox)
                    labels.append(label)
                data_info["gt_bboxes"] = np.stack(bboxes)
                data_info["gt_labels"] = np.stack(labels)
            else:
                # Insert dummy gt bboxes for data pipeline in mmaction
                data_info["gt_bboxes"] = np.zeros((1, 4))

            if self.proposals is not None:
                img_key = data_info["img_key"]
                if img_key in self.proposals:
                    proposal = self.proposals[img_key]
                else:
                    proposal = np.array([[0, 0, 1, 1, 1]])
                data_info["proposals"] = proposal[:, :4]
                data_info["scores"] = proposal[:, 4]

            return data_info

        def patch_proposals(self):
            """Remove fixed string format.

            AVA dataset pre-proposals have fixed string format.
            Fixed string format have scalability issues so here we remove it
            """
            # FIXME This may consume lots of time depends on size of proposals
            root_logger.info("Patching pre proposals...")
            for img_key in list(self.proposals):
                proposal = self.proposals.pop(img_key)
                new_img_key = img_key.split(",")[0] + "," + str(int(img_key.split(",")[1]))
                self.proposals[f"{new_img_key}"] = proposal
            root_logger.info("Done.")

    @check_input_parameters_type({"otx_dataset": DatasetParamTypeCheck})
    # TODO Remove duplicated codes with mmaction's AVADataset
    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: List[LabelEntity],
        pipeline: Sequence[dict],
        exclude_file: Optional[str],
        proposal_file: Optional[str],
        timestamp_start: Union[int, str],
        timestamp_end: Union[int, str],
        test_mode: bool = False,
        person_det_score_thr: float = 0.9,
        num_max_proposals: int = 1000,
        filename_tmpl: str = "_{:06}.jpg",
        start_index: int = 1,
        modality: str = "RGB",
        fps: int = 30,
    ):
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.test_mode = test_mode
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        self.modality = modality
        self._FPS = fps
        self.exclude_file = exclude_file
        self.proposal_file = proposal_file
        self.person_det_score_thr = person_det_score_thr
        self.num_max_proposals = num_max_proposals

        # Load start and end frame index
        # This will be changed with CVAT annotation
        if isinstance(timestamp_start, int):
            self.timestamp_start = timestamp_start
        else:
            with open(timestamp_start, encoding="utf-8") as time_file:
                self.timestamp_start = json.load(time_file)
        if isinstance(timestamp_end, int):
            self.timestamp_end = timestamp_end
        else:
            with open(timestamp_end, encoding="utf-8") as time_file:
                self.timestamp_end = json.load(time_file)

        # OTX does not support custom_classes
        self.custom_classes = None

        self.video_infos = OTXActionDetDataset._DataInfoProxy(otx_dataset, labels, fps, test_mode)

        self.pipeline = Compose(pipeline)

        # if not test_mode:
        #     valid_indexes = self.filter_exclude_file()
        #     self.video_infos = [self.video_infos[i] for i in valid_indexes]

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results["modality"] = self.modality
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results["modality"] = self.modality
        return self.pipeline(results)

    # pylint: disable=too-many-locals, unused-argument
    def evaluate(self, results, *args, metrics=("mAP",), metric_options=None, logger=None, **kwargs):
        """Evaluate the prediction results and report mAP."""
        assert len(metrics) == 1 and metrics[0] == "mAP", (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            "See https://github.com/open-mmlab/mmaction2/pull/567 "
            "for more info."
        )
        csv_results = det2csv(self, results, self.custom_classes)
        predictions = self.get_predictions(csv_results)

        ret = {}
        for metric in metrics:
            msg = f"Evaluating {metric} ..."
            if logger is None:
                msg = "\n" + msg
            print_log(msg, logger=logger)

            eval_result = det_eval(
                predictions,
                metric,
                self.labels,
                self.video_infos,
                self.exclude_file,
                custom_classes=self.custom_classes,
            )
            log_msg = []
            for key, value in eval_result.items():
                log_msg.append(f"\n{key}\t{value: .4f}")
            log_msg = "".join(log_msg)
            print_log(log_msg, logger=logger)
            ret.update(eval_result)
        return ret

    @staticmethod
    def get_predictions(csv_results):
        """Convert model's inference results to predictions."""
        csv_results = np.array(csv_results)
        _img_keys = csv_results[:, :2]
        _boxes = csv_results[:, 2:6]
        _labels = csv_results[:, 6]
        _scores = csv_results[:, 7]

        boxes = defaultdict(list)
        labels = defaultdict(list)
        scores = defaultdict(list)

        for _img_key, _box, _label, _score in zip(_img_keys, _boxes, _labels, _scores):
            img_key = _img_key[0] + "," + _img_key[1]
            box = _box.astype("float")
            label = _label.astype("int")
            score = _score.astype("float")
            boxes[img_key].append(box)
            labels[img_key].append(label)
            scores[img_key].append(score)
        return (boxes, labels, scores)
