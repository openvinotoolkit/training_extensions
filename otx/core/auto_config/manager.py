"""Auto Configuration Manager ."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import datumaro

class AutoConfigManager:
    """Auto configuration manager that could set the proper configuration."""
    def __init__(self):
        self.task_data_dict = {
            "classification": ['imagenet'],
            "detection": ['coco', 'voc', 'yolo'],
            "instance_segmentation": ['coco', 'voc'],
            "semantic_segmentation": [
                'common_semantic_segmentation',
                'voc',
                'cityscapes',
                'ade20k2017',
                'ade20k2020'
            ]
        }
    @classmethod
    def find_task_type(cls, data_root):
        """Detect task type."""
        task_type_candidates = []
        data_format = datumaro.Environment().detect_dataset(data_root)
        for task in cls.task_data_dict:
            if data_format in cls.task_data_dict[task]:
                task_type_candidates.append(task)
    
    def _is_cvat_format(self, path):
        """Detect whether data path is CVAT format or not."""
        #TODO: Will be supported by dautmaro detect_dataset. 
        pass


    def _is_mvtec(self, path):
        """Detect whether data path is MVTec format or not."""

        # condition 1: 'ground_truth', 'train', 'test' folder are located
        mvtec_folders = sorted(['ground_truth', 'train', 'test'])
        folder_list = []
        for sub in os.listdir(path):
            if os.isdir(sub):
                folder_list.append(sub)
        return mvtec_folders == sorted(folder_list)
        
        # condition 2: 