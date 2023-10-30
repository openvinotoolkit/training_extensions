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
    "segmentation": {
        "train_data_roots": "tests/assets/kvasir_36",
        "val_data_roots": "tests/assets/kvasir_36",
        "test_data_roots": "tests/assets/kvasir_36",
        "sample": "tests/assets/kvasir_36/val_set/images/cju0t4oil7vzk099370nun5h9.png",
        "models": ["ocr_lite_hrnet_s_mod2"],
    }
}
