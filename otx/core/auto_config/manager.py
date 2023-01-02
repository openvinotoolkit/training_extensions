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
    
    def get_task_type(self, data_root: str) -> str:
        """Detect task type.
        
        For some datasets (i.e. COCO, VOC, MVTec), can't be fully automated.
        Because those datasets have several format at the same time. 
        (i.e. for the COCO case, object detection and instance segmentation annotations coexist)
        In this case, the task_type will be selected to default value.
        
        For action tasks, currently action_classification is default.
        
        If Datumaro supports the Kinetics, AVA datasets, MVTec, _is_cvat_format(), _is_mvtec_format()
        functions will be deleted.
        """
        data_root = os.path.abspath(data_root)
        if self._is_cvat_format(data_root):
            return "action_classification"
        elif self._is_mvtec_format(data_root):
            return "anomaly_classification"
        else:
            data_formats = datumaro.Environment().detect_dataset(data_root)
            data_format = data_formats[0] if 'imagenet' not in data_formats else 'imagenet'
            print(f"data_format: {data_format}")
            for task in self.task_data_dict:
                if data_format in self.task_data_dict[task]:
                    return task
                else:
                    continue
        
        raise ValueError()
    
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
            # only use the folder name.
            if os.path.isdir(sub_folder):
                folder_list.append(sub_folder)
        return sorted(folder_list) == mvtec_format
        