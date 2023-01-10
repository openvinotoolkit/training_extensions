"""Datumaro Helper."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from typing import Optional

import datumaro
from datumaro.components.dataset import Dataset
from datumaro.components.dataset import DatasetSubset


class DatumaroHelper:
    """The aim of DatumaroHelper is support datumaro functions at easy use.
    All kind of functions implemented in Datumaro are supported by this helper.
    """
    @staticmethod
    def get_train_dataset(dataset:Dataset) -> DatasetSubset:
        """Returns train dataset."""
        for k, v in dataset.subsets().items():
            if "train" in k or "default" in k:
                return v
    
    @staticmethod
    def get_val_dataset(dataset:Dataset) -> DatasetSubset:
        """Returns validation dataset."""
        for k, v in dataset.subsets().items():
            if "val" in k or "default" in k:
                return v
    
    @staticmethod 
    def get_data_format(data_root: str) -> str:
        """Find the format of dataset."""
        data_root = os.path.abspath(data_root)
        
        data_format = ""
        
        # TODO #
        # Currently, below `if/else` statements is mandatory 
        # because Datumaro can't detect the multi-cvat and mvtec.
        # After, the upgrade of Datumaro, below codes will be changed. 
        if DatumaroHelper._is_cvat_format(data_root):
            data_format = "multi-cvat"
        elif DatumaroHelper._is_mvtec_format(data_root):
            data_format = "mvtec"
        else: 
            data_formats = datumaro.Environment().detect_dataset(data_root)
            #TODO: how to avoid hard-coded part
            data_format = data_formats[0] if 'imagenet' not in data_formats else 'imagenet'
        return data_format
    
    @staticmethod 
    def import_dataset(
            self, 
            train_data_root: str, 
            train_data_format: str,
            val_data_root: Optional[str] = None
        ) -> dict:
        """Import dataset."""
         
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
        pass
     
    @staticmethod
    def auto_split(task, dataset:Dataset):
        pass 

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
        
        will be deprecated soon.
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
        
        will be deprecated soon.
        """
        
        mvtec_format = sorted(['ground_truth', 'train', 'test'])
        folder_list = []
        for sub_folder in os.listdir(path):
            sub_folder_path = os.path.join(path, sub_folder)
            # only use the folder name.
            if os.path.isdir(sub_folder_path):
                folder_list.append(sub_folder)
        return sorted(folder_list) == mvtec_format
        