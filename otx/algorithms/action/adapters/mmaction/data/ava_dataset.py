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
from datetime import datetime
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmaction.core.evaluation.ava_utils import results2csv
from mmaction.datasets.ava_dataset import AVADataset
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from mmcv.utils import print_log

from otx.algorithms.action.adapters.mmaction.core.evaluation import ava_eval
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)


# pylint: disable=too-many-instance-attributes
@DATASETS.register_module()
class OTXAVADataset(AVADataset):
    """Wrapper that allows using a OTX dataset to train mmaction models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.
    """

    class _DataInfoProxy:
        def __init__(self, otx_dataset, labels):
            self.otx_dataset = otx_dataset
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}

        def __len__(self):
            return len(self.otx_dataset)

        def __getitem__(self, index):
            """Prepare a dict 'data_info' that is expected by the mmaction pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            """

            dataset = self.otx_dataset
            item = dataset[index]
            ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

            data_info = dict(
                dataset_item=item,
                index=index,
                ann_info=dict(label_list=self.labels),
                ignored_labels=ignored_labels,
            )

            return data_info

    @check_input_parameters_type({"otx_dataset": DatasetParamTypeCheck})
    # pylint: disable=too-many-arguments, invalid-name, super-init-not-called, too-many-locals
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

        self.data_infos = OTXAVADataset._DataInfoProxy(otx_dataset, labels)

        if self.proposal_file is not None:
            self.proposals = mmcv.load(self.proposal_file)
        else:
            self.proposals = None

        self.pipeline = Compose(pipeline)
        self.make_video_infos()

    def __len__(self):
        """Return length of dataset."""
        return len(self.data_infos)

    # FIXME This is very similar with mmation's ava_dataset
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results["img_key"]
        video_id = results["video_id"]

        results["filename_tmpl"] = self.get_filename_tmpl(img_key)
        results["modality"] = self.modality
        results["start_index"] = self.start_index
        results["timestamp_start"] = self.get_timestamp("start", video_id)
        results["timestamp_end"] = self.get_timestamp("end", video_id)

        if self.proposals is not None:
            if img_key not in self.proposals:
                results["proposals"] = np.array([[0, 0, 1, 1]])
                results["scores"] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = proposals[:, 4] >= thr
                    proposals = proposals[positive_inds]
                    proposals = proposals[: self.num_max_proposals]
                    results["proposals"] = proposals[:, :4]
                    results["scores"] = proposals[:, 4]
                else:
                    proposals = proposals[: self.num_max_proposals]
                    results["proposals"] = proposals

        ann = results.pop("ann")
        if ann is not None:
            results["gt_bboxes"] = ann["gt_bboxes"]
            results["gt_labels"] = ann["gt_labels"]
            results["entity_ids"] = ann["entity_ids"]
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        img_key = results["img_key"]
        video_id = results["video_id"]

        results["filename_tmpl"] = self.get_filename_tmpl(img_key)
        results["modality"] = self.modality
        results["start_index"] = self.start_index
        results["timestamp_start"] = self.get_timestamp("start", video_id)
        results["timestamp_end"] = self.get_timestamp("end", video_id)

        if self.proposals is not None:
            if img_key not in self.proposals:
                results["proposals"] = np.array([[0, 0, 1, 1]])
                results["scores"] = np.array([1])
            else:
                proposals = self.proposals[img_key]
                assert proposals.shape[-1] in [4, 5]
                if proposals.shape[-1] == 5:
                    thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                    positive_inds = proposals[:, 4] >= thr
                    proposals = proposals[positive_inds]
                    proposals = proposals[: self.num_max_proposals]
                    results["proposals"] = proposals[:, :4]
                    results["scores"] = proposals[:, 4]
                else:
                    proposals = proposals[: self.num_max_proposals]
                    results["proposals"] = proposals

        ann = results.pop("ann")
        if ann is not None:
            results["gt_bboxes"] = ann["gt_bboxes"]
            results["gt_labels"] = ann["gt_labels"]
            results["entity_ids"] = ann["entity_ids"]
        else:
            # This is for RawFrameDecode pipeline
            results["gt_bboxes"] = np.zeros((1, 4))
        return self.pipeline(results)

    def get_timestamp(self, key, video_id=None):
        """Get start or end timestamp for video."""
        if key == "start":
            timestamp = self.timestamp_start
        else:
            timestamp = self.timestamp_end
        if isinstance(timestamp, int):
            return timestamp
        return timestamp[video_id]

    def get_filename_tmpl(self, img_key):
        """Get dataset's own filename template."""
        # FIXME This is very heuristic way. CVAT format may change this
        if self.filename_tmpl[0] == "_":
            return img_key.split(",")[0] + self.filename_tmpl
        return self.filename_tmpl

    def make_video_infos(self):
        """Make mmaction style video infos."""
        self.video_infos = []
        for data_info in self.data_infos:
            media = data_info["dataset_item"].media
            media["fps"] = self._FPS
            video_id = media["video_id"]
            timestamp_start = self.get_timestamp("start", video_id)
            timestamp_end = self.get_timestamp("end", video_id)
            shot_info = (0, timestamp_end - timestamp_start) * self._FPS
            anns = data_info["dataset_item"].get_annotations()
            media["ann"] = None
            media["shot_info"] = shot_info
            if len(anns) > 0:
                bboxes, labels = [], []
                for ann in anns:
                    bbox = np.asarray([ann.shape.x1, ann.shape.y1, ann.shape.x2, ann.shape.y2])
                    valid_labels = np.array([int(label.id) for label in ann.get_labels()], dtype=np.int8)
                    label = np.zeros(len(self.labels) + 1, dtype=np.float32)
                    label[valid_labels] = 1.0
                    bboxes.append(bbox)
                    labels.append(label)
                bboxes = np.stack(bboxes)
                labels = np.stack(labels)
                media["ann"] = {"gt_bboxes": bboxes, "gt_labels": labels, "entity_ids": []}
            media["timestamp_start"] = timestamp_start
            media["timestamp_end"] = timestamp_end
            self.video_infos.append(media)

        if not self.test_mode:
            valid_indexes = self.filter_exclude_file()
            self.video_infos = [self.video_infos[i] for i in valid_indexes]

    # pylint: disable=too-many-locals, unused-argument
    def evaluate(self, results, *args, metrics=("mAP",), metric_options=None, logger=None, **kwargs):
        """Evaluate the prediction results and report mAP."""
        assert len(metrics) == 1 and metrics[0] == "mAP", (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            "See https://github.com/open-mmlab/mmaction2/pull/567 "
            "for more info."
        )
        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = f"AVA_{time_now}_result.csv"
        results2csv(self, results, temp_file, self.custom_classes)

        ret = {}
        for metric in metrics:
            msg = f"Evaluating {metric} ..."
            if logger is None:
                msg = "\n" + msg
            print_log(msg, logger=logger)

            eval_result = ava_eval(
                temp_file, metric, self.labels, self.video_infos, self.exclude_file, custom_classes=self.custom_classes
            )
            log_msg = []
            for key, value in eval_result.items():
                log_msg.append(f"\n{key}\t{value: .4f}")
            log_msg = "".join(log_msg)
            print_log(log_msg, logger=logger)
            ret.update(eval_result)

        os.remove(temp_file)

        return ret
