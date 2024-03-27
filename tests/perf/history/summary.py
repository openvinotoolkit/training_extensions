# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX perfomance benchamrk history summary utilities."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

V1_V2_NAME_MAP = {
    # Columns
    "data_size": "data_group",
    "avg_data_time": "train/data_time",
    "avg_iter_time": "train/iter_time",
    "epoch": "train/epoch",
    "f-measure(export)": "export/f1-score",
    "f-measure(optimize)": "optimize/f1-score",
    "f-measure(train)": "test/f1-score",
    "train_e2e_time": "train/e2e_time",
    "val_score": "val/score",
    "Precision(export)": "export/precision",
    "Precision(optimize)": "optimize/precision",
    "Precision(train)": "test/precision",
    "Recall(export)": "export/recall",
    "Recall(optimize)": "optimize/recall",
    "Recall(train)": "test/recall",
    "commit": "test_commit",
    "branch": "test_branch",
    "version": "otx_version",
    "avg_time_per_image(export)": "export/iter_time",
    "avg_time_per_image(optimize)": "optimize/iter_time",
    "Accuracy(export)": "export/accuracy",
    "Accuracy(optimize)": "optimize/accuracy",
    "Accuracy(train)": "test/accuracy",
    "Dice Average(export)": "export/dice",
    "Dice Average(optimize)": "optimize/dice",
    "Dice Average(train)": "test/dice",
    # Task names
    "single_label_classification": "classification/multi_class_cls",
    "multi_label_classification": "classification/multi_label_cls",
    "hierarchical_label_classification": "classification/h_label_cls",
    # Model names
    "ote_anomaly_classification_padim": "padim",
    "ote_anomaly_classification_stfpm": "stfpm",
    "ote_anomaly_detection_padim": "padim",
    "ote_anomaly_detection_stfpm": "stfpm",
    "ote_anomaly_segmentation_padim": "padim",
    "ote_anomaly_segmentation_stfpm": "stfpm",
    "Custom_Image_Classification_EfficientNet-V2-S": "efficientnet_v2_light",
    "Custom_Image_Classification_EfficinetNet-B0": "efficientnet_b0_light",
    "Custom_Image_Classification_MobileNet-V3-large-1x": "mobilenet_v3_large_light",
    "Custom_Image_Classification_DeiT-Tiny": "otx_deit_tiny",
    "Custom_Object_Detection_Gen3_ATSS": "atss_mobilenetv2",
    "Custom_Object_Detection_Gen3_SSD": "ssd_mobilenetv2",
    "Custom_Object_Detection_YOLOX": "yolox_tiny",
    "Object_Detection_YOLOX_S": "yolox_s",
    "Object_Detection_YOLOX_L": "yolox_l",
    "Object_Detection_YOLOX_X": "yolox_x",
    "Object_Detection_ResNeXt101_ATSS": "atss_resnext101",
    "all_tile": "all",
    "Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B": "maskrcnn_efficientnetb2b",
    "Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50": "maskrcnn_r50",
    "Custom_Counting_Instance_Segmentation_MaskRCNN_SwinT_FP16": "maskrcnn_swint",
    "Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B_tile": "maskrcnn_efficientnetb2b_tile",
    "Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50_tile": "maskrcnn_r50_tile",
    "Custom_Counting_Instance_Segmentation_MaskRCNN_SwinT_FP16_tile": "maskrcnn_swint_tile",
    "Custom_Semantic_Segmentation_Lite-HRNet-18_OCR": "litehrnet_18_old",
    "Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR": "litehrnet_18",
    "Custom_Semantic_Segmentation_Lite-HRNet-s-mod2_OCR": "litehrnet_s",
    "Custom_Semantic_Segmentation_Lite-HRNet-x-mod3_OCR": "litehrnet_x",
    # Dataset names
    "anomaly/mvtec/bottle_small/1": "mvtec_bottle_small_1",
    "anomaly/mvtec/bottle_small/2": "mvtec_bottle_small_2",
    "anomaly/mvtec/bottle_small/3": "mvtec_bottle_small_3",
    "anomaly/mvtec/wood_medium": "mvtec_wood_medium",
    "anomaly/mvtec/hazelnut_large": "mvtec_hazelut_large",
    "anomaly/mvtec/bottle_small/": "all",
    "anomaly/mvtec/": "all",
    "classification/single_label/multiclass_CUB_small/1": "multiclass_CUB_small_1",
    "classification/single_label/multiclass_CUB_small/2": "multiclass_CUB_small_2",
    "classification/single_label/multiclass_CUB_small/3": "multiclass_CUB_small_3",
    "classification/single_label/multiclass_CUB_small/": "all",
    "classification/single_label/multiclass_CUB_medium": "multiclass_CUB_medium",
    "classification/single_label/multiclass_food101_large": "multiclass_food101_large",
    "classification/single_label/multiclass_CUB_small/": "all",
    "classification/single_label/multiclass_": "all",
    "classification/multi_label/multilabel_CUB_small/1": "multilabel_CUB_small_1",
    "classification/multi_label/multilabel_CUB_small/2": "multilabel_CUB_small_2",
    "classification/multi_label/multilabel_CUB_small/3": "multilabel_CUB_small_3",
    "classification/multi_label/multilabel_CUB_small/": "all",
    "classification/multi_label/multilabel_CUB_medium": "multilabel_CUB_medium",
    "classification/multi_label/multilabel_food101_large": "multilabel_food101_large",
    "classification/multi_label/multilabel_CUB_small/": "all",
    "classification/multi_label/multilabel_": "all",
    "classification/h_label/h_label_CUB_small/1": "hlabel_CUB_small_1",
    "classification/h_label/h_label_CUB_small/2": "hlabel_CUB_small_2",
    "classification/h_label/h_label_CUB_small/3": "hlabel_CUB_small_3",
    "classification/h_label/h_label_CUB_small/": "all",
    "classification/h_label/h_label_CUB_medium": "hlabel_CUB_medium",
    "classification/h_label/h_label_food101_large": "hlabel_food101_large",
    "classification/h_label/h_label_CUB_small/": "all",
    "classification/h_label/h_label_CUB_": "all",
    "detection/pothole_small/1": "pothole_small_1",
    "detection/pothole_small/2": "pothole_small_2",
    "detection/pothole_small/3": "pothole_small_3",
    "detection/pothole_small/": "all",
    "detection/pothole_medium": "pothole_medium",
    "detection/vitens_large": "vitens_large",
    "detection/": "all",
    "instance_seg/wgisd_small/1": "wgisd_small_1",
    "instance_seg/wgisd_small/2": "wgisd_small_2",
    "instance_seg/wgisd_small/3": "wgisd_small_3",
    "instance_seg/wgisd_small/": "all",
    "instance_seg/coco_car_person_medium": "coco_car_person_medium",
    "instance_seg/coco_car_person_medium": "coco_car_person_medium",
    "instance_seg/": "all",
    "tiling_instance_seg/vitens_aeromonas_small/1": "vitens_aeromonas_small_1",
    "tiling_instance_seg/vitens_aeromonas_small/2": "vitens_aeromonas_small_2",
    "tiling_instance_seg/vitens_aeromonas_small/3": "vitens_aeromonas_small_3",
    "tiling_instance_seg/vitens_aeromonas_small/": "all",
    "tiling_instance_seg/vitens_aeromonas_medium": "vitens_aeromonas_medium",
    "tiling_instance_seg/vitens_aeromonas_": "all",
    "tiling_instance_seg/": "all",
    "semantic_seg/kvasir_small/1": "kvasir_small_1",
    "semantic_seg/kvasir_small/2": "kvasir_small_2",
    "semantic_seg/kvasir_small/3": "kvasir_small_3",
    "semantic_seg/kvasir_small/": "all",
    "semantic_seg/kvasir_medium": "kvasir_medium",
    "semantic_seg/kvasir_large": "kvasir_large",
    "semantic_seg/kvasir_": "all",
}

# Load all csv data
def load_all(root_dir: Path, normalize: bool = False):
    """Load all csv files and csv in zip files."""
    def _normalize(data: pd.DataFrame) -> pd.DataFrame:
        """v1 -> v2"""
        # Map v1 tiling task -> v2 tiling model
        tiling_indices = data["task"] == "tiling_instance_segmentation"
        data.loc[tiling_indices, "task"] = data.loc[tiling_indices, "task"].str.replace("tiling_", "")
        data.loc[tiling_indices, "model"] = data.loc[tiling_indices, "model"] + "_tile"
        # Map anomaly metrics
        if "test/image_F1Score" in data:
            anomaly_indices = data["task"] == "anomaly_classification"
            data.loc[anomaly_indices, "test/f1-score"] = data.loc[anomaly_indices, "test/image_F1Score"]
            data.loc[anomaly_indices, "export/f1-score"] = data.loc[anomaly_indices, "export/image_F1Score"]
            data.loc[anomaly_indices, "optimize/f1-score"] = data.loc[anomaly_indices, "optimize/image_F1Score"]
            anomaly_indices = data["task"] == "anomaly_detection"
            data.loc[anomaly_indices, "test/f1-score"] = data.loc[anomaly_indices, "test/image_F1Score"]
            data.loc[anomaly_indices, "export/f1-score"] = data.loc[anomaly_indices, "export/image_F1Score"]
            data.loc[anomaly_indices, "optimize/f1-score"] = data.loc[anomaly_indices, "optimize/image_F1Score"]
        if "test/pixel_F1Score" in data:
            anomaly_indices = data["task"] == "anomaly_segmentation"
            data.loc[anomaly_indices, "test/f1-score"] = data.loc[anomaly_indices, "test/pixel_F1Score"]
            data.loc[anomaly_indices, "export/f1-score"] = data.loc[anomaly_indices, "export/pixel_F1Score"]
            data.loc[anomaly_indices, "optimize/f1-score"] = data.loc[anomaly_indices, "optimize/pixel_F1Score"]
        # Map other names
        data = data.rename(columns=V1_V2_NAME_MAP).replace(V1_V2_NAME_MAP)
        # Fill blanks
        data.loc[data["model"].isna(), "model"] = "all"
        data.loc[data["data"].isna(), "data"] = "all"
        data.loc[data["data_group"].isna(), "data_group"] = "all"
        return data

    csvs = root_dir.glob("**/*.csv")
    all_data = []
    for csv in csvs:
        data = pd.read_csv(csv)
        if normalize:
            data = _normalize(data)
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

all_data = load_all(Path("."), normalize=True)


TASK_METRIC_MAP = {
    "anomaly_classification": "test/f1-score",
    "anomaly_detection": "test/f1-score",
    "anomaly_segmentation": "test/f1-score",
    "classification/multi_class_cls": "test/accuracy",
    "classification/multi_label_cls": "test/accuracy",
    "classification/h_label_cls": "test/accuracy",
    "detection": "test/f1-score",
    "instance_segmentation": "test/f1-score",
    "semantic_segmentation": "test/dice",
    "visual_prompting": "test/dice",
    "zero_shot_visual_prompting": "test/dice",
}


def summarize_task(task: str):
    """Summarize benchmark histoy by task."""
    score_metric = TASK_METRIC_MAP[task]
    metrics = [f"{score_metric}", "train/e2e_time", "export/iter_time"]
    column_order = [
        (f"{score_metric}", "all"),
        (f"{score_metric}", "small"),
        (f"{score_metric}", "medium"),
        (f"{score_metric}", "large"),
        ("train/e2e_time", "all"),
        ("train/e2e_time", "small"),
        ("train/e2e_time", "medium"),
        ("train/e2e_time", "large"),
        ("export/iter_time", "all"),
        ("export/iter_time", "small"),
        ("export/iter_time", "medium"),
        ("export/iter_time", "large"),
    ]
    data = all_data.query(f"task == '{task}'")
    # data = data.pivot_table(index=["model", "otx_version"], columns=["data_group", "data"], values=metrics, aggfunc="mean")
    data = data.pivot_table(index=["model", "otx_version"], columns=["data_group"], values=metrics, aggfunc="mean")
    # data = data.style.set_sticky(axis="index")
    data = data.reindex(column_order, axis=1)
    return data


def summarize_meta():
    """Summarize benchmark metadata by version."""
    entries = [
        "date",
        "otx_ref",
        "test_branch",
        "test_commit",
        "cpu_info",
        "accelerator_info",
        "user_name",
        "machine_name",
    ]
    data = all_data.pivot_table(index=["otx_version"], values=entries, aggfunc="first")
    data = data.reindex(entries, axis=1)
    # data = data.style.set_sticky(axis="index")
    return data
