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
        "sample": "tests/assets/cvat_dataset/action_classification/v2_test",
        "models": ["otx_x3d"],
    },
    "visual_prompting": {
        "train_data_roots": "tests/assets/car_tree_bug",
        "val_data_roots": "tests/assets/car_tree_bug",
        "test_data_roots": "tests/assets/car_tree_bug",
        "sample": "tests/assets/car_tree_bug/images/train/Slide6.PNG",
        "models": ["otx_tiny_vit"],
    },
}
