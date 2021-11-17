from mmdet.apis.ote.extension.datasets.data_utils import load_dataset_items_coco_format

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.subset import Subset


class ObjectDetectionDataset(DatasetEntity):

    def __init__(self,
                 train_ann_file=None,
                 train_data_root=None,
                 val_ann_file=None,
                 val_data_root=None,
                 test_ann_file=None,
                 test_data_root=None):

        labels_list = []
        items = []

        if train_ann_file is not None and train_data_root is not None:
            items.extend(load_dataset_items_coco_format(
                ann_file_path=train_ann_file,
                data_root_dir=train_data_root,
                subset=Subset.TRAINING,
                labels_list=labels_list))

        if val_ann_file is not None and val_data_root is not None:
            items.extend(load_dataset_items_coco_format(
                ann_file_path=val_ann_file,
                data_root_dir=val_data_root,
                subset=Subset.VALIDATION,
                labels_list=labels_list))

        if test_ann_file is not None and test_data_root is not None:
            items.extend(load_dataset_items_coco_format(
                ann_file_path=test_ann_file,
                data_root_dir=test_data_root,
                subset=Subset.TESTING,
                labels_list=labels_list))

        super().__init__(items=items)
