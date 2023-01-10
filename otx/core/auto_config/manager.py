"""Auto Configuration Manager ."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from otx.core.data.utils.datumaro_helper import DatumaroHelper
from datumaro.components.dataset import Dataset

class AutoConfigManager:
    """Auto configuration manager that could set the proper configuration."""
    def __init__(self):
        self.task_data_dict = {
            "CLASSIFICATION": ['imagenet'],
            "DETECTION": ['coco', 'voc', 'yolo'],
            "INSTANCE_SEGMENTATION": ['coco', 'voc'],
            "SEGMENTATION": [
                'common_semantic_segmentation',
                'voc',
                'cityscapes',
                'ade20k2017',
                'ade20k2020'
            ],
            "ACTION_CLASSIFICATION": ['multi-cvat'],
            "ACTION_DETECTION": ['multi-cvat'],
            "ANOMALY_CLASSIFICATION": ['mvtec'],
            "ANOMALY_DETECTION": ['mvtec'],
            "ANOMALY_SEGMENTATION": ['mvtec'],
        }
     
    def get_task_type(self, data_format: str) -> str:
        """Detect task type.
        
        For some datasets (i.e. COCO, VOC, MVTec), can't be fully automated.
        Because those datasets have several format at the same time. 
        (i.e. for the COCO case, object detection and instance segmentation annotations coexist)
        In this case, the task_type will be selected to default value.
        
        For action tasks, currently action_classification is default.
        
        If Datumaro supports the Kinetics, AVA datasets, MVTec, _is_cvat_format(), _is_mvtec_format()
        functions will be deleted.
        """
        
        task = ""
        if data_format == "multi-cvat":
            task = "action_classification"
        elif data_format == 'mvtec':
            task = "anomaly_classification"
        else:
            # pick task type
            for task_key in self.task_data_dict:
                if data_format in self.task_data_dict[task_key]:
                    task = task_key
        return task
    
    def write_data_with_cfg(self):
        pass