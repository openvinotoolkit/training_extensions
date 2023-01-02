from otx.core.auto_config.manager import AutoConfigManager

config_manager = AutoConfigManager()

##### Classification #####
# Classification: ImageNet
task_type = config_manager.get_task_type(data_root="./data/datumaro/coco_dataset/coco_detection")
print(f"Classification:  {task_type}")

##### Detection #####
# Detection: COCO
task_type = config_manager.get_task_type(data_root="./data/datumaro/coco_dataset/coco_detection")
print(f"Detection (COCO):  {task_type}")
# Detection: YOLO
task_type = config_manager.get_task_type(data_root="./data/datumaro/yolo_dataset")
print(f"Detection (YOLO):  {task_type}")
# Detection: VOC1
task_type = config_manager.get_task_type(data_root="./data/datumaro/voc_dataset/voc_dataset1")
print(f"Detection (VOC1):  {task_type}")
# Detection: VOC2
task_type = config_manager.get_task_type(data_root="./data/datumaro/voc_dataset/voc_dataset2")
print(f"Detection (VOC2):  {task_type}")
# Detection: VOC3
task_type = config_manager.get_task_type(data_root="./data/datumaro/voc_dataset/voc_dataset3")
print(f"Detection (VOC3):  {task_type}")

##### Segmentation #####