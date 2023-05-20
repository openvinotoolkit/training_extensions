"""OTX MVTec Dataset facilitate OTX Anomaly Training.

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Reference:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
      Unsupervised Anomaly Detection; in: International Journal of Computer Vision
      129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

    - Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
      A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
      in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
      9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

# Copyright (C) 2021 Intel Corporation
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

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from anomalib.data.mvtec import make_mvtec_dataset
from pandas.core.frame import DataFrame

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.color import Color
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map


class OtxMvtecDataset:
    """Generate OTX MVTec Dataset from the anomaly detection datasets that follows the MVTec format.

    Args:
        path (Union[str, Path], optional): Path to the MVTec dataset category.
            Defaults to "./datasets/MVTec/bottle".
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.5.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Create validation set from the test set by splitting
            it to half. Default to True.

    Examples:
        >>> dataset_generator = OtxMvtecDataset()
        >>> dataset = dataset_generator.generate()
        >>> dataset[0].media.numpy.shape
        (900, 900, 3)
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path: Union[str, Path],
        task_type: TaskType = TaskType.ANOMALY_CLASSIFICATION,
    ):
        self.path = path if isinstance(path, Path) else Path(path)
        self.task_type = task_type

        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            self.label_domain = Domain.ANOMALY_CLASSIFICATION
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            self.label_domain = Domain.ANOMALY_SEGMENTATION

        self.normal_label = LabelEntity(name="Normal", domain=self.label_domain, id=ID(), color=Color(0, 255, 0))
        self.abnormal_label = LabelEntity(
            name="Anomalous",
            domain=self.label_domain,
            id=ID(),
            is_anomalous=True,
            color=Color(255, 0, 0),
        )
        self.label_map = {0: self.normal_label, 1: self.abnormal_label}

    def get_samples(self) -> DataFrame:
        """Get MVTec samples.

        Get MVTec samples in a pandas DataFrame. Update the certain columns
        to match the OTX naming terminology. For example, column `split` is
        renamed to `subset`. Labels are also renamed by creating their
        corresponding OTX LabelEntities

        Returns:
            DataFrame: Final list of samples comprising all the required
                information to create the OTX Dataset.
        """
        samples = make_mvtec_dataset(root=self.path)

        # Set the OTX SDK Splits
        samples = samples.rename(columns={"split": "subset"})
        samples.loc[samples.subset == "train", "subset"] = Subset.TRAINING
        samples.loc[samples.subset == "val", "subset"] = Subset.VALIDATION
        samples.loc[samples.subset == "test", "subset"] = Subset.TESTING

        # Create and Set the OTX Labels
        samples.loc[samples.label != "good", "label"] = self.abnormal_label
        samples.loc[samples.label == "good", "label"] = self.normal_label

        samples = samples.reset_index(drop=True)

        return samples

    def generate(self) -> DatasetEntity:
        """Generate OTX Anomaly Dataset.

        Returns:
            DatasetEntity: Output OTX Anomaly Dataset from an MVTec
        """
        samples = self.get_samples()
        dataset_items: List[DatasetItemEntity] = []
        for _, sample in samples.iterrows():
            # Create image
            image = Image(file_path=sample.image_path)

            # Create annotation
            if self.task_type == TaskType.ANOMALY_CLASSIFICATION or sample.label == self.normal_label:
                shape = Rectangle(x1=0, y1=0, x2=1, y2=1)
                labels = [ScoredLabel(sample.label)]
                annotations = [Annotation(shape=shape, labels=labels)]
                annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)
            elif self.task_type == TaskType.ANOMALY_SEGMENTATION and sample.label == self.abnormal_label:
                mask = (cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE) / 255).astype(np.uint8)
                annotations = create_annotation_from_segmentation_map(mask, np.ones_like(mask), self.label_map)
                annotation_scene = AnnotationSceneEntity(annotations=annotations, kind=AnnotationSceneKind.ANNOTATION)
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

            # Create dataset item
            dataset_item = DatasetItemEntity(media=image, annotation_scene=annotation_scene, subset=sample.subset)

            # Add to dataset items
            dataset_items.append(dataset_item)

        dataset = DatasetEntity(items=dataset_items)
        return dataset
