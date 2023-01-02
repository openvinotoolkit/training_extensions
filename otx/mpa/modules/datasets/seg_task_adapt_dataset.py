# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.datasets import DATASETS, build_dataset


@DATASETS.register_module()
class SegTaskAdaptDataset(object):
    """Dataset wrapper for task-adaptive semantic segmentation.

    When setting up a dataset for semantic segmentation, it is important
    whether background class is included. If background class is required, it is set
    as the first class regardless of whether the dataset has background class. (with_background=True)

    (TODO) If background class is not required, it is ignored. (with_background=False)

    Args
        classes (iterable): model classes set in stage.
        new_classes (iterable): new classes in stage.
        with_background (bool): whether to include background class. (default=True)
    """

    def __init__(self, classes, new_classes, with_background=True, **kwargs):
        self.classes = classes
        self.new_classes = new_classes
        self.dataset_classes = [c for c in self.classes if c != "background"]

        dataset_cfg = kwargs.copy()
        org_type = dataset_cfg.pop("org_type")
        dataset_cfg["type"] = org_type
        if "dataset" in dataset_cfg:
            dataset_cfg["dataset"]["classes"] = self.dataset_classes
            dataset_cfg["dataset"]["new_classes"] = self.new_classes
        else:
            dataset_cfg["classes"] = self.dataset_classes
            dataset_cfg["new_classes"] = self.new_classes

        self.dataset = build_dataset(dataset_cfg)

        # reset background class
        if with_background:
            # TODO : check if 'while' loop is required
            _dataset = self.dataset
            while True:
                if hasattr(_dataset, "CLASSES"):
                    _dataset.CLASSES = [c for c in _dataset.CLASSES if c != "background"]
                    _dataset.CLASSES = ["background"] + _dataset.CLASSES
                    _dataset.PALETTE = [[0, 0, 0]] + _dataset.PALETTE

                if hasattr(_dataset, "label_map"):
                    if _dataset.label_map is None:
                        _label_map = {}
                        for i, c in enumerate(_dataset.CLASSES):
                            if c not in self.dataset_classes:
                                _label_map[i] = -1
                            else:
                                _label_map[i] = self.dataset_classes.index(c)
                        _dataset.label_map = _label_map
                    _dataset.label_map = {k: v + 1 for k, v in _dataset.label_map.items()}

                if hasattr(_dataset, "dataset"):
                    _dataset = _dataset.dataset
                else:
                    break

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def evaluate(self, results, **kwargs):
        return self.dataset.evaluate(results, **kwargs)
