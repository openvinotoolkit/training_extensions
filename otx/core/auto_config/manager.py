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
            ],
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
        for root, dirs, path in os.listdir(path):
            sub_folder_path = os.path.join(path, sub_folder)
        pass


    def _is_mvtec(self, path):
        """Detect whether data path is MVTec format or not.
        Check the first-level architecture folder, to know whether the dataset is MVTec or not.
        
        MVTec default structure like as below:
        root
        |--ground_truth
        |--train
        |--test
        
        """
        # conditio 'ground_truth', 'train', 'test' folder are located
        mvtec_folders = sorted(['ground_truth', 'train', 'test'])
        folder_list = []
        for sub_folder in os.listdir(path):
            # only use the folder name.
            if os.isdir(sub_folder):
                folder_list.append(sub_folder)
        return mvtec_folders == sorted(folder_list)
        