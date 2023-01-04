"""Auto Configuration Manager ."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any
import os
import datumaro
from datumaro.components.dataset import Dataset
from otx.core.data.utils.datumaro_helper import DatumaroHelper

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
        
        self.task = None # type: str
    
    def _find_data_format(self, data_root: str) -> str:
        """Find the format of dataset."""
        data_root = os.path.abspath(data_root)
        
        data_format = ""
        if self._is_cvat_format(data_root):
            data_format = "multi-cvat"
        elif self._is_mvtec_format(data_root):
            data_format = "mvtec"
        else: 
            data_formats = datumaro.Environment().detect_dataset(data_root)
            data_format = data_formats[0] if 'imagenet' not in data_formats else 'imagenet'
        return data_format
         
    def find_task_type(self, data_root: str) -> str:
        """Detect task type.
        
        For some datasets (i.e. COCO, VOC, MVTec), can't be fully automated.
        Because those datasets have several format at the same time. 
        (i.e. for the COCO case, object detection and instance segmentation annotations coexist)
        In this case, the task_type will be selected to default value.
        
        For action tasks, currently action_classification is default.
        
        If Datumaro supports the Kinetics, AVA datasets, MVTec, _is_cvat_format(), _is_mvtec_format()
        functions will be deleted.
        """
        data_format = self._find_data_format(data_root)
        
        task = ""
        if data_format == "multi-cvat":
            task = "action_classification"
        elif data_format == 'mvtec':
            task = "anomaly_classification"
        else:
            for task_key in self.task_data_dict:
                if data_format in self.task_data_dict[task_key]:
                    task = task_key
        return task
    
    def get_data_cfg(self, train_data_root: str, val_data_root: str = None) -> dict:
        """Automatically generate data configuration."""
        data_config = {}
         
        # Make Datumaro dataset by using training data 
        train_data_format = self._find_data_format(train_data_root) 
        datumaro_dataset = Dataset.import_from(train_data_root, format=train_data_format)
        
        # Find train and val set 
        train_set = DatumaroHelper.get_train_dataset(datumaro_dataset)
        val_set = DatumaroHelper.get_val_dataset(datumaro_dataset)
        
        # If there is input from user for validation dataset
        # Make validation set by using user's input
        # If Datumaro automatically made validation dataset, it will be overwritten
        if val_data_root:
            val_dataset = Dataset.import_from(val_data_root, format=self.task)
            val_set = DatumaroHelper.get_val_dataset(val_dataset)
        
         
            
     
    def _is_cvat_format(self, path: str) -> bool:
        """Detect whether data path is CVAT format or not.
        Currently, we used multi-video CVAT format for Action tasks.
        
        This function can detect the multi-video CVAT format.
        
        Multi-video CVAT format
        root
        |--video_0
            |--images
                |--frame0001.png
            |--annotations.xml
        |--video_1
        |--video_2
        """
        
        cvat_format = sorted(['images', 'annotations.xml'])
        for sub_folder in os.listdir(path):
            # video_0, video_1, ...
            sub_folder_path = os.path.join(path, sub_folder)
            # files must be same with cvat_format
            if os.path.isdir(sub_folder_path):
                files = sorted(os.listdir(sub_folder_path))
                if files != cvat_format:
                    return False
        return True

    def _is_mvtec_format(self, path: str) -> bool:
        """Detect whether data path is MVTec format or not.
        Check the first-level architecture folder, to know whether the dataset is MVTec or not.
        
        MVTec default structure like as below:
        root
        |--ground_truth
        |--train
        |--test
        """
        
        mvtec_format = sorted(['ground_truth', 'train', 'test'])
        folder_list = []
        for sub_folder in os.listdir(path):
            sub_folder_path = os.path.join(path, sub_folder)
            # only use the folder name.
            if os.path.isdir(sub_folder_path):
                folder_list.append(sub_folder)
        return sorted(folder_list) == mvtec_format
        