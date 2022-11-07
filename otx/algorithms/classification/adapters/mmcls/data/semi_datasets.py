import numpy as np
from mmcls.core import average_performance, mAP
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS, PIPELINES
from mmcls.datasets.pipelines import Compose
from mmcv.utils.registry import build_from_cfg
from mpa.utils.logger import get_logger
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from otx.algorithms.common.utils import get_cls_img_indices, get_old_new_img_indices


@DATASETS.register_module()
class MPAClsUnlabelDataset(BaseDataset):
    """Multi-class classification dataset class."""

    def __init__(
        self, otx_dataset=None, labels=None, empty_label=None, **kwargs
    ):  # pylint: disable=super-init-not-called
        self.otx_dataset = otx_dataset
        pipeline = kwargs["pipeline"]
        if isinstance(pipeline, dict):
            self.pipeline = {}
            for k, v in pipeline.items():
                _pipeline = [dict(type="LoadImageFromOTXDataset"), *v]
                _pipeline = [build_from_cfg(p, PIPELINES) for p in _pipeline]
                self.pipeline[k] = Compose(_pipeline)
            self.num_pipes = len(pipeline)
        elif isinstance(pipeline, list):
            self.num_pipes = 1
            _pipeline = [dict(type="LoadImageFromOTXDataset"), *pipeline]
            self.pipeline = Compose([build_from_cfg(p, PIPELINES) for p in _pipeline])

    def load_annotations(self):
        pass

    def __getitem__(self, index):
        """Get item from dataset."""
        dataset = self.otx_dataset
        item = dataset[index]
        ignored_labels = np.array([self.label_idx[lbs.id] for lbs in item.ignored_labels])

        height, width = item.height, item.width

        data_info = dict(
            dataset_item=item,
            width=width,
            height=height,
            index=index,
            ignored_labels=ignored_labels,
        )

        if self.pipeline is None:
            return data_info
        return self.pipeline(data_info)

    def __len__(self):
        """Get dataset length."""
        return len(self.otx_dataset)

    def evaluate(
        self, results, metric="accuracy", metric_options=None, logger=None
    ):  # pylint: disable=redefined-outer-name
        pass

    def class_accuracy(self, results, gt_labels):
        pass