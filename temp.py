from otx.core.auto_config.manager import AutoConfigManager

config_manager = AutoConfigManager()

"""
##### Classification #####
# ImageNet
task_type = config_manager.get_task_type(data_root="./data/datumaro/imagenet_dataset")
print(f"Classification:  {task_type}")

##### Detection #####
# COCO
task_type = config_manager.get_task_type(data_root="./data/datumaro/coco_dataset/coco_detection")
print(f"Detection (COCO):  {task_type}")
# YOLO
task_type = config_manager.get_task_type(data_root="./data/datumaro/yolo_dataset")
print(f"Detection (YOLO):  {task_type}")
# VOC1
task_type = config_manager.get_task_type(data_root="./data/datumaro/voc_dataset/voc_dataset1")
print(f"Detection (VOC1):  {task_type}")
# VOC2
task_type = config_manager.get_task_type(data_root="./data/datumaro/voc_dataset/voc_dataset2")
print(f"Detection (VOC2):  {task_type}")
# VOC3
task_type = config_manager.get_task_type(data_root="./data/datumaro/voc_dataset/voc_dataset3")
print(f"Detection (VOC3):  {task_type}")

##### Segmentation #####
# Common Semantic Segmentation - dataset
task_type = config_manager.get_task_type(data_root="./data/datumaro/common_semantic_segmentation_dataset/dataset")
print(f"Segmentation (Common Semantic Segmentation - dataset):  {task_type}")

# Common Semantic Segmentation - non standard dataset
task_type = config_manager.get_task_type(data_root="./data/datumaro/common_semantic_segmentation_dataset/non_standard_dataset")
print(f"Segmentation (Common Semantic Segmentation - non standard dataset):  {task_type}")

# Cityscapes
task_type = config_manager.get_task_type(data_root="./data/datumaro/cityscapes_dataset/dataset")
print(f"Segmentation (Cityscapes):  {task_type}")

# ADE20K2017
task_type = config_manager.get_task_type(data_root="./data/datumaro/ade20k2017_dataset/dataset")
print(f"Segmentation (ADE20K 2017):  {task_type}")

# ADE20K2017
task_type = config_manager.get_task_type(data_root="./data/datumaro/ade20k2020_dataset/dataset")
print(f"Segmentation (ADE20K 2020):  {task_type}")

##### Action Classification, Detection #####
task_type = config_manager.get_task_type(data_root="./data/datumaro/cvat_dataset/action_classification")
print(f"Action Classification (CVAT):  {task_type}")

task_type = config_manager.get_task_type(data_root="./data/datumaro/cvat_dataset/action_detection")
print(f"Action Classification (CVAT):  {task_type}")

##### Anomaly Classification, Detection, Segmentation #####
task_type = config_manager.get_task_type(data_root="./data/datumaro/mvtec")
print(f"Anomaly Classification (MVTec):  {task_type}")
"""

task_type = config_manager.find_task_type(data_root="./data/datumaro/coco_dataset/coco_detection")
print(f"Detection (COCO):  {task_type}")

data_config = config_manager.get_data_cfg(
    train_data_root="./data/datumaro/coco_dataset/coco_detection",
    val_data_root="./data/datumaro/coco_dataset/coco_val_temp"
)