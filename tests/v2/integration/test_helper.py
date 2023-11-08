# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


TASK_CONFIGURATION = {
    "classification": {
        "train_data_roots": "tests/assets/classification_dataset_class_incremental",
        "val_data_roots": "tests/assets/classification_dataset_class_incremental",
        "test_data_roots": "tests/assets/classification_dataset_class_incremental",
        "sample": "tests/assets/classification_dataset_class_incremental/2/22.jpg",
        "models": ["otx_efficientnet_b0", "otx_mobilenet_v3_large"],
    },
    "anomaly_classification": {
        "train_data_roots": "tests/assets/anomaly/hazelnut/train",
        "val_data_roots": "tests/assets/anomaly/hazelnut/test",
        "test_data_roots": "tests/assets/anomaly/hazelnut/test",
        "sample": "tests/assets/anomaly/hazelnut/test/colour/01.jpg",
        "models": ["otx_padim"],
    },
    "action_classification": {
        "train_data_roots": "tests/assets/cvat_dataset/action_classification/train",
        "val_data_roots": "tests/assets/cvat_dataset/action_classification/train",
        "test_data_roots": "tests/assets/cvat_dataset/action_classification/train",
        "sample": "tests/assets/cvat_dataset/action_classification/predict",
        "models": ["otx_x3d"],
    },
    "visual_prompting": {
        "train_data_roots": "tests/assets/car_tree_bug",
        "val_data_roots": "tests/assets/car_tree_bug",
        "test_data_roots": "tests/assets/car_tree_bug",
        "sample": "tests/assets/car_tree_bug/images/train/Slide6.PNG",
        "models": ["otx_tiny_vit"],
    },
    "segmentation": {
        "train_data_roots": "tests/assets/common_semantic_segmentation_dataset/train",
        "val_data_roots": "tests/assets/common_semantic_segmentation_dataset/val",
        "test_data_roots": "tests/assets/common_semantic_segmentation_dataset/val",
        "sample": "tests/assets/common_semantic_segmentation_dataset/train/images/0001.png",
        "models": ["otx_lite_hrnet_s_mod2"],
    },
    "detection": {
        "train_data_roots": "tests/assets/car_tree_bug",
        "val_data_roots": "tests/assets/car_tree_bug",
        "test_data_roots": "tests/assets/car_tree_bug",
        "sample": "tests/assets/car_tree_bug/images/val/Slide5.PNG",
        "models": ["otx_mobilenetv2-atss"],
    },
    "instance_segmentation": {
        "train_data_roots": "tests/assets/car_tree_bug",
        "val_data_roots": "tests/assets/car_tree_bug",
        "test_data_roots": "tests/assets/car_tree_bug",
        "sample": "tests/assets/car_tree_bug/images/val/Slide5.PNG",
        "models": ["otx_resnet50_maskrcnn"],
    },
}
