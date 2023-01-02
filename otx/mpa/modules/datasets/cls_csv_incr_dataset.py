# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcls.datasets.builder import DATASETS

from otx.mpa.modules.utils.task_adapt import map_class_names

from .cls_csv_dataset import CSVDatasetCls
from .multi_cls_dataset import MultiClsDataset


@DATASETS.register_module()
class LwfTaskIncDataset(MultiClsDataset):
    def __init__(self, pre_stage_res=None, model_tasks=None, **kwargs):
        self.pre_stage_res = pre_stage_res
        self.model_tasks = model_tasks
        if self.pre_stage_res is not None:
            self.pre_stage_data = np.load(self.pre_stage_res, allow_pickle=True)
            for p in kwargs["pipeline"]:
                if p["type"] == "Collect":
                    p["keys"] += ["soft_label"]
        super(LwfTaskIncDataset, self).__init__(**kwargs)

    def load_annotations(self):
        if self.pre_stage_res is not None:
            data_infos = self.pre_stage_data
            index_map = dict()
            for i, k in enumerate(self.tasks.keys()):
                index_map.update({i: map_class_names(self.tasks[k], self.model_tasks[k])})
            for data in data_infos:
                data["img_prefix"] = self.data_prefix
                for i, map in index_map.items():
                    data["gt_label"][i] = map[data["gt_label"][i]]
        else:
            data_infos = super().load_annotations()
        return data_infos


@DATASETS.register_module()
class ClassIncDataset(CSVDatasetCls):
    def __init__(self, pre_stage_res=None, dst_classes=None, **kwargs):
        self.pre_stage_res = pre_stage_res
        self.dst_classes = dst_classes
        if self.pre_stage_res is not None:
            self.pre_stage_data = np.load(self.pre_stage_res, allow_pickle=True)
            for p in kwargs["pipeline"]:
                if p["type"] == "Collect":
                    p["keys"] += ["soft_label"]
                    p["keys"] += ["center"]
        super(ClassIncDataset, self).__init__(**kwargs)

    def load_annotations(self):
        if self.pre_stage_res is not None:
            dataframe = self._read_csvs()
            num_new_class = len(dataframe)
            data_infos = self.pre_stage_data
            index_map = map_class_names(self.CLASSES, self.dst_classes)
            for i, data in enumerate(data_infos):
                data["img_prefix"] = self.data_prefix
                if i < num_new_class:
                    data["gt_label"] = np.array(index_map[data["gt_label"]], dtype=np.int64)
        else:
            if self.dst_classes is not None:
                self.CLASSES = self.dst_classes
            data_infos = super().load_annotations()
        return data_infos
