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

import os
from collections import defaultdict
from copy import copy, deepcopy
from logging import Logger
from typing import Any, Dict, List, Sequence, Tuple

import mmcv
import numpy as np
from mmaction.core.evaluation.ava_utils import det2csv
from mmaction.datasets.ava_dataset import AVADataset
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from mmaction.utils import get_root_logger
from mmcv.utils import print_log

from otx.algorithms.action.adapters.mmaction.data.pipelines import RawFrameDecode
from otx.algorithms.action.adapters.mmaction.utils import det_eval
from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.metadata import VideoMetadata
from otx.api.utils.shape_factory import ShapeFactory

root_logger = get_root_logger()


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, super-init-not-called
@DATASETS.register_module()
class OTXActionDetDataset(AVADataset):
    """Wrapper that allows using a OTX dataset to train action detection models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX DatasetEntity object.
    It is adapted from AVADataset of mmaction, but it supports other dataset such as UCF and JHMDB.
    """

    class _DataInfoProxy:
        def __init__(
            self,
            otx_dataset: DatasetEntity,
            labels: List[LabelEntity],
            person_det_score_thr: float = 0.9,
            num_max_proposals: int = 1000,
            modality: str = "RGB",
            fps: int = 30,
        ):
            self.otx_dataset = deepcopy(otx_dataset)
            self.labels = labels
            self.label_idx = {label.id: i for i, label in enumerate(labels)}
            self.person_det_score_thr = person_det_score_thr
            self.num_max_proposals = num_max_proposals
            self.modality = modality
            self.fps = fps
            self.data_root = "/" + os.path.join(*os.path.abspath(str(self.otx_dataset[0].media.path)).split("/")[:-4])
            self.proposal_file_name = os.path.abspath(str(self.otx_dataset[0].media.path)).split("/")[-4]
            self.proposal_file = os.path.join(self.data_root, f"{self.proposal_file_name}.pkl")
            self.video_info: Dict[str, Any] = {}

            if os.path.exists(self.proposal_file):
                self.proposals = mmcv.load(self.proposal_file)
                self._patch_proposals()
            else:
                self.proposals = None

            self._update_meta_data()

        def __len__(self) -> int:
            return len(self.otx_dataset)

        def _update_meta_data(self):
            """Update video metadata of each item in self.otx_dataset.

            During iterating DatasetEntity, this function generate video_info(dictionary) to record metadata of video
            After that, this function update metadata of each DatasetItemEntity of DatasetEntity
                - start_index: Offset for the video, this value will be added to sampled frame indices
                - timestamp: Timestamp of the DatasetItemEntity
                - timestamp_start: Start timestamp of the video, this will be used to generate shot_info
                - timestamp_end: End timestamp of the video, this will be used to generate shot_info
                - shot_info = (0, (timestamp_end - timestamp_start)) * self.fps:
                              Range of frame indices, this is used to sample frame indices
                - img_key = "video_id,frame_idx": key of pre-proposal dictionary
                - modality = Modality of data, 'RGB' or 'Flow(Optical Flow)'
            This function removes empty frames(frame with no action), since they are only used for background of clips
            """

            video_info = {}
            start_index = 0
            for idx, item in enumerate(self.otx_dataset):
                metadata = item.get_metadata()[0].data
                if metadata.video_id in video_info:
                    video_info[metadata.video_id]["start_index"] = start_index
                    if metadata.frame_idx < video_info[metadata.video_id]["timestamp_start"]:
                        video_info[metadata.video_id]["timestamp_start"] = metadata.frame_idx
                    if metadata.frame_idx > video_info[metadata.video_id]["timestamp_end"]:
                        video_info[metadata.video_id]["timestamp_end"] = metadata.frame_idx
                else:
                    video_info[metadata.video_id] = {
                        "start_index": idx,
                        "timestamp_start": metadata.frame_idx,
                        "timestamp_end": metadata.frame_idx,
                    }
                    start_index = idx

            remove_indices = []
            for idx, item in enumerate(self.otx_dataset):
                metadata = item.get_metadata()[0].data
                if metadata.is_empty_frame:
                    remove_indices.append(idx)
                    continue

                for key, value in video_info[metadata.video_id].items():
                    metadata.update(key, value)

                shot_info = (0, (metadata.timestamp_end - metadata.timestamp_start)) * self.fps
                img_key = f"{metadata.video_id},{metadata.frame_idx}"
                ignored_labels = np.array([self.label_idx[label.id] for label in item.ignored_labels])
                metadata.update("shot_info", shot_info)
                metadata.update("img_key", img_key)
                metadata.update("timestamp", metadata.frame_idx)
                metadata.update("ignored_labels", ignored_labels)
                metadata.update("modality", self.modality)

                anns = item.get_annotations()
                self._update_annotations(metadata, anns)

            self.otx_dataset.remove_at_indices(remove_indices)
            self.video_info.update(video_info)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            """Prepare a dict 'data_info' that is expected by the mmaction pipeline to handle images and annotations.

            :return data_info: dictionary that contains the image and image metadata, as well as the labels of
            the objects in the image
            This iterates self.otx_dataset, which removes empty frames
            """

            item = self.otx_dataset[index]
            metadata = item.get_metadata()[0].data

            data_info = dict(
                **metadata.metadata,
                ann_info=dict(label_list=self.labels),
                fps=self.fps,
            )

            return data_info

        def _patch_proposals(self):
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

        def _update_annotations(self, metadata: VideoMetadata, anns: List[Annotation]):
            """Update annotation information to item's metadata."""
            if len(anns) > 0:
                bboxes, labels = [], []
                for ann in anns:
                    rectangle = ShapeFactory.shape_as_rectangle(ann.shape)
                    bbox = np.asarray([rectangle.x1, rectangle.y1, rectangle.x2, rectangle.y2])
                    valid_labels = np.array([int(label.id) for label in ann.get_labels()], dtype=int)
                    label = np.zeros(len(self.labels) + 1, dtype=np.float32)
                    label[valid_labels] = 1.0
                    bboxes.append(bbox)
                    labels.append(label)
                metadata.update("gt_bboxes", np.stack(bboxes))
                metadata.update("gt_labels", np.stack(labels))
            else:
                # Insert dummy gt bboxes for data pipeline in mmaction
                metadata.update("gt_bboxes", np.zeros((1, 4)))

            if self.proposals is not None:
                if metadata.img_key in self.proposals:  # type: ignore[attr-defined]
                    proposals = self.proposals[metadata.img_key]  # type: ignore[attr-defined]
                else:
                    proposals = np.array([[0, 0, 1, 1, 1]])
                thr = min(self.person_det_score_thr, max(proposals[:, 4]))
                positive_inds = proposals[:, 4] >= thr
                proposals = proposals[positive_inds]
                proposals = proposals[: self.num_max_proposals]
                metadata.update("proposals", proposals[:, :4])
                metadata.update("scores", proposals[:, 4])

    # TODO Remove duplicated codes with mmaction's AVADataset
    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: List[LabelEntity],
        pipeline: Sequence[dict],
        test_mode: bool = False,
        person_det_score_thr: float = 0.9,
        num_max_proposals: int = 1000,
        modality: str = "RGB",
        fps: int = 30,
    ):
        self.otx_dataset = otx_dataset
        self.labels = labels
        self.CLASSES = [label.name for label in labels]
        self.test_mode = test_mode
        self.modality = modality
        self._FPS = fps
        self.person_det_score_thr = person_det_score_thr
        self.num_max_proposals = num_max_proposals

        # OTX does not support custom_classes
        self.custom_classes = None

        self.video_infos = OTXActionDetDataset._DataInfoProxy(
            otx_dataset, labels, person_det_score_thr, num_max_proposals, modality, fps
        )

        self.pipeline = Compose(pipeline)
        for transform in self.pipeline.transforms:
            if isinstance(transform, RawFrameDecode):
                transform.otx_dataset = self.otx_dataset

        # TODO. Handle exclude file for AVA dataset
        self.exclude_file = None

    def prepare_train_frames(self, idx: int) -> Dict[str, Any]:
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data
        """
        item = copy(self.video_infos[idx])  # Copying dict(), not contents
        return self.pipeline(item)

    def prepare_test_frames(self, idx: int) -> Dict[str, Any]:
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data
        """
        item = copy(self.video_infos[idx])  # Copying dict(), not contents
        return self.pipeline(item)

    # pylint: disable=too-many-locals, unused-argument
    def evaluate(
        self,
        results: List[List[np.ndarray]],
        metrics: Tuple[str] = ("mAP",),
        logger: Logger = None,
        **kwargs,
    ):
        """Evaluate the prediction results and report mAP."""
        assert len(metrics) == 1 and metrics[0] == "mAP", (
            'For evaluation on AVADataset, you need to use metrics "mAP" '
            "See https://github.com/open-mmlab/mmaction2/pull/567 "
            "for more info."
        )
        csv_results = det2csv(self, results, self.custom_classes)
        predictions = self._get_predictions(csv_results)

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
            str_log_msg = "".join(log_msg)
            print_log(str_log_msg, logger=logger)
            ret.update(eval_result)
        return ret

    @staticmethod
    def _get_predictions(csv_results: List[Tuple]):
        """Convert model's inference results to predictions."""
        np_csv_results = np.array(csv_results)
        _img_keys = np_csv_results[:, :2]
        _boxes = np_csv_results[:, 2:6]
        _labels = np_csv_results[:, 6]
        _scores = np_csv_results[:, 7]

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
