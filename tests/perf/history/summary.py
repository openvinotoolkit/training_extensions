# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX perfomance benchmark history summary utilities."""

from __future__ import annotations

import argparse
import fnmatch
import io
import os
import sys
import numpy as np
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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
    "test/Dice": "test/dice",
    "export/Dice": "export/dice",
    "optimize/Dice": "optimize/dice",
    "repeat": "seed",
    # Task names
    "single_label_classification": "classification/multi_class_cls",
    "Single_label_semi_sl_classificaiton": "classification/multi_class_cls_semisl",
    "multi_label_classification": "classification/multi_label_cls",
    "hierarchical_label_classification": "classification/h_label_cls",
    # Model names
    "ote_anomaly_classification_padim": "padim",
    "ote_anomaly_classification_stfpm": "stfpm",
    "ote_anomaly_detection_padim": "padim",
    "ote_anomaly_detection_stfpm": "stfpm",
    "ote_anomaly_segmentation_padim": "padim",
    "ote_anomaly_segmentation_stfpm": "stfpm",
    "Custom_Image_Classification_EfficientNet-V2-S": "efficientnet_v2",
    "Custom_Image_Classification_EfficinetNet-B0": "efficientnet_b0",
    "Custom_Image_Classification_MobileNet-V3-large-1x": "mobilenet_v3_large",
    "Custom_Image_Classification_DeiT-Tiny": "deit_tiny",
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
    "Custom_Semantic_Segmentation_SegNext_B": "segnext_b",
    "Custom_Semantic_Segmentation_SegNext_s": "segnext_s",
    "Custom_Semantic_Segmentation_SegNext_t": "segnext_t",
    "Visual_Prompting_SAM_Tiny_ViT": "sam_tiny_vit",
    "Visual_Prompting_SAM_ViT_B": "sam_vit_b",
    "Zero_Shot_SAM_Tiny_ViT": "sam_tiny_vit",
    "Zero_Shot_SAM_ViT_B": "sam_vit_b",
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
    "classification/single_label/multiclass_": "all",
    "classification/multi_label/multilabel_CUB_small/1": "multilabel_CUB_small_1",
    "classification/multi_label/multilabel_CUB_small/2": "multilabel_CUB_small_2",
    "classification/multi_label/multilabel_CUB_small/3": "multilabel_CUB_small_3",
    "classification/multi_label/multilabel_CUB_small/": "all",
    "classification/multi_label/multilabel_CUB_medium": "multilabel_CUB_medium",
    "classification/multi_label/multilabel_food101_large": "multilabel_food101_large",
    "classification/multi_label/multilabel_": "all",
    "classification/h_label/h_label_CUB_small/1": "hlabel_CUB_small_1",
    "classification/h_label/h_label_CUB_small/2": "hlabel_CUB_small_2",
    "classification/h_label/h_label_CUB_small/3": "hlabel_CUB_small_3",
    "classification/h_label/h_label_CUB_small/": "all",
    "classification/h_label/h_label_CUB_medium": "hlabel_CUB_medium",
    "classification/h_label/h_label_food101_large": "hlabel_food101_large",
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
    "visual_prompting/wgisd_small/1": "wgisd_small_1",
    "visual_prompting/wgisd_small/2": "wgisd_small_2",
    "visual_prompting/wgisd_small/3": "wgisd_small_3",
    "visual_prompting/coco_car_person_medium": "coco_car_person_medium",
    "visual_prompting/Vitens-Coliform-coco": "vitens_coliform",
    "zero_shot_visual_prompting/coco_car_person_medium": "coco_car_person_medium",
}


TASK_METRIC_MAP = {
    "anomaly_classification": "f1-score",
    "anomaly_detection": "f1-score",
    "anomaly_segmentation": "f1-score",
    "classification/multi_class_cls": "accuracy",
    "classification/multi_class_cls_semisl": "accuracy",
    "classification/multi_label_cls": "accuracy",
    "classification/h_label_cls": "accuracy",
    "detection": "f1-score",
    "instance_segmentation": "f1-score",
    "semantic_segmentation": "dice",
    "visual_prompting": "dice",
    "zero_shot_visual_prompting": "dice",
}


TASK_ABBR_MAP = {
    "anomaly_classification": "ano_cls",
    "anomaly_detection": "ano_det",
    "anomaly_segmentation": "ano_seg",
    "classification/multi_class_cls": "cls",
    "classification/multi_class_cls_semisl": "cls_semisl",
    "classification/multi_label_cls": "m_cls",
    "classification/h_label_cls": "h_cls",
    "detection": "det",
    "instance_segmentation": "iseg",
    "semantic_segmentation": "sseg",
    "visual_prompting": "vp",
    "zero_shot_visual_prompting": "zvp",
}


GETI_DEFAULT_MODEL = {
    "anomaly_classification": {"padim", "stfpm"},
    "anomaly_detection": {"padim", "stfpm"},
    "anomaly_segmentation": {"padim", "stfpm"},
    "classification/multi_class_cls": {"efficientnet_b0", "efficientnet_v2", "mobilenet_v3_large"},
    "classification/multi_label_cls": {"efficientnet_b0", "efficientnet_v2", "mobilenet_v3_large"},
    "classification/h_label_cls": {"efficientnet_b0", "efficientnet_v2", "mobilenet_v3_large"},
    "detection": {"atss_mobilenetv2", "ssd_mobilenetv2", "yolox_tiny"},
    "instance_segmentation": {
        "non-tile": ["maskrcnn_efficientnetb2b", "maskrcnn_r50"],
        "tile": ["maskrcnn_efficientnetb2b_tile", "maskrcnn_r50_tile"]
    },
    "semantic_segmentation": {'litehrnet_18', 'litehrnet_s', 'litehrnet_x'}
}


METADATA_ENTRIES = [
    "date",
    "otx_ref",
    "test_branch",
    "test_commit",
    "cpu_info",
    "accelerator_info",
    "user_name",
    "machine_name",
]


def load(root_dir: Path, need_normalize: bool = False, pattern="*raw*.csv") -> pd.DataFrame:
    """Load all csv files and csv in zip files."""

    history = []
    # Load csv files in the directory
    csv_files = root_dir.rglob(pattern)
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        if need_normalize:
            data = normalize(data)
        history.append(data)
    # Load csv files in zip files
    zip_files = Path(root_dir).rglob("*.zip")
    for zip_file in zip_files:
        with ZipFile(zip_file) as zf:
            csv_files = fnmatch.filter(zf.namelist(), pattern)
            for csv_file in csv_files:
                csv_bytes = io.BytesIO(zf.read(csv_file))
                data = pd.read_csv(csv_bytes)
                if need_normalize:
                    data = normalize(data)
                history.append(data)
    if len(history) == 0:
        return pd.DataFrame()
    history = pd.concat(history, ignore_index=True)
    # Post process
    version_entry = "otx_version" if "otx_version" in history else "version"
    history[version_entry] = history[version_entry].astype(str)
    history["seed"] = history["seed"].fillna(0)
    history = average(
        history,
        [version_entry, "task", "model", "data_group", "data", "seed"],
    )  # Average mulitple retrials w/ same seed
    if "index" in history:
        history.drop("index", axis=1)
    return history


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    """Map v1 terms -> v2"""
    # Map v1 tiling task -> v2 tiling model
    tiling_indices = data["task"] == "tiling_instance_segmentation"
    data.loc[tiling_indices, "task"] = data.loc[tiling_indices, "task"].str.replace("tiling_", "")
    data.loc[tiling_indices, "model"] = data.loc[tiling_indices, "model"] + "_tile"
    # Map anomaly metrics
    anomaly_indices = data["task"] == "anomaly_classification"
    if "test/image_F1Score" in data:
        data.loc[anomaly_indices, "test/f1-score"] = data.loc[anomaly_indices, "test/image_F1Score"]
    if "export/image_F1Score" in data:
        data.loc[anomaly_indices, "export/f1-score"] = data.loc[anomaly_indices, "export/image_F1Score"]
    if "optimize/image_F1Score" in data:
        data.loc[anomaly_indices, "optimize/f1-score"] = data.loc[anomaly_indices, "optimize/image_F1Score"]
    anomaly_indices = data["task"] == "anomaly_detection"
    if "test/image_F1Score" in data:
        data.loc[anomaly_indices, "test/f1-score"] = data.loc[anomaly_indices, "test/image_F1Score"]
    if "export/image_F1Score" in data:
        data.loc[anomaly_indices, "export/f1-score"] = data.loc[anomaly_indices, "export/image_F1Score"]
    if "optimize/image_F1Score" in data:
        data.loc[anomaly_indices, "optimize/f1-score"] = data.loc[anomaly_indices, "optimize/image_F1Score"]
    anomaly_indices = data["task"] == "anomaly_segmentation"
    if "test/pixel_F1Score" in data:
        data.loc[anomaly_indices, "test/f1-score"] = data.loc[anomaly_indices, "test/pixel_F1Score"]
    if "export/pixel_F1Score" in data:
        data.loc[anomaly_indices, "export/f1-score"] = data.loc[anomaly_indices, "export/pixel_F1Score"]
    if "optimize/pixel_F1Score" in data:
        data.loc[anomaly_indices, "optimize/f1-score"] = data.loc[anomaly_indices, "optimize/pixel_F1Score"]
    # Map other names
    data = data.rename(columns=V1_V2_NAME_MAP).replace(V1_V2_NAME_MAP)
    # Fill blanks
    data.loc[data["model"].isna(), "model"] = "all"
    data.loc[data["data"].isna(), "data"] = "all"
    data.loc[data["data_group"].isna(), "data_group"] = "all"
    return data


def average(raw_data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average raw data w.r.t. given keys."""
    if raw_data is None or len(raw_data) == 0:
        return pd.DataFrame()
    # To avoid SettingWithCopyWarning
    raw_data = raw_data.copy()
    # Preproc
    for col in METADATA_ENTRIES:
        raw_data.loc[:, col] = raw_data[col].astype(str)  # Prevent strings like '2.0.0' being loaded as float
    # Average by keys
    grouped = raw_data.groupby(keys)
    aggregated = grouped.mean(numeric_only=True)
    # Merge tag columns (non-numeric & non-index)
    tag_columns = set(raw_data.columns) - set(aggregated.columns) - set(keys)
    for col in tag_columns:
        # Take common string prefix such as: ["data/1", "data/2", "data/3"] -> "data/"
        aggregated[col] = grouped[col].agg(lambda x: os.path.commonprefix(x.tolist()))
    return aggregated.reset_index()


def summarize(raw_data: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Summarize raw data into pivot table w.r.t given metrics"""
    if raw_data is None or len(raw_data) == 0:
        return pd.DataFrame()
    if not metrics:
        # Add all numeric metrics
        metrics = raw_data.select_dtypes(include=["number"]).columns.to_list()

    grouped_data = raw_data.groupby(["otx_version", "task", "model", "data_group"])
    aggregated = grouped_data.agg({metric: ['mean', 'std'] for metric in metrics}).reset_index()

    # Flatten the MultiIndex columns, excluding 'otx_version', 'task', 'model', 'data_group'
    cols_to_exclude = {'otx_version', 'task', 'model', 'data_group'}
    aggregated.columns = [('_'.join(col) if col[0] not in cols_to_exclude else col[0]) for col in aggregated.columns.values]

    # Calculate the 'all' data group by taking the mean of the means across all data groups for each model
    # Only numeric columns should be included in the mean calculation
    numeric_cols = aggregated.select_dtypes(include=["number"]).columns
    all_data = aggregated.groupby(["otx_version", "task", "model"])[numeric_cols].mean().reset_index()
    all_data['data_group'] = 'all'

    # Combine the 'all' data group with the rest of the data
    combined_data = pd.concat([all_data, aggregated], ignore_index=True)

    # Pivot the combined data to create the summary table
    summary_data = combined_data.pivot_table(
        index=["task", "model", "otx_version"],
        columns=["data_group"],
        values=[col for col in aggregated.columns if col not in cols_to_exclude],
        aggfunc='first'
    )

    # Split the flattened column names and create a MultiIndex
    new_columns = []
    for col in summary_data.columns:
        metric_stat, data_group = col
        metric, stat = metric_stat.rsplit('_', 1)  # Split on the last underscore
        new_columns.append((metric, stat, data_group))

    # Assign the new MultiIndex to the columns of the DataFrame
    summary_data.columns = pd.MultiIndex.from_tuples(new_columns, names=['metric', 'stat', 'data_group'])

    # Sort the index and columns
    summary_data = summary_data.reorder_levels(['data_group', 'metric', 'stat'], axis=1)
    summary_data.sort_index(axis=1, inplace=True)
    return summary_data


def summarize_table(history: pd.DataFrame, task: str) -> pd.DataFrame:
    """Summarize benchmark histoy table by task."""
    score_metric = TASK_METRIC_MAP[task]
    metrics = [
        f"export/{score_metric}",
        f"optimize/{score_metric}",
        f"test/{score_metric}",
        "train/e2e_time",
        "train/epoch",
        "train/iter_time",
    ]
    raw_data = history.query(f"task == '{task}'")
    summary_data = summarize(raw_data, metrics)
    avg_row = summary_data.groupby(['task', 'otx_version']).mean().reset_index()
    avg_row['model'] = 'avg'
    summary_data = pd.concat([summary_data, avg_row.set_index(['task', 'model', 'otx_version'])])

    geti_models = GETI_DEFAULT_MODEL.get(task, set())
    if task == "instance_segmentation":
        for tile_type, models in geti_models.items():
            model_name = f'geti_default_avg ({tile_type})'
            geti_data = summary_data.loc[summary_data.index.get_level_values('model').isin(models)]
            geti_default_avg_row = geti_data.groupby(['task', 'otx_version']).mean().reset_index()
            geti_default_avg_row['model'] = model_name
            geti_default_avg_row['task'] = task
            summary_data = pd.concat([summary_data, geti_default_avg_row.set_index(['task', 'model', 'otx_version'])])
    else:
        geti_data = summary_data.loc[summary_data.index.get_level_values('model').isin(geti_models)]
        geti_default_avg_row = geti_data.groupby(['task', 'otx_version']).mean().reset_index()
        geti_default_avg_row['model'] = 'geti_default_avg'
        geti_default_avg_row['task'] = task
        summary_data = pd.concat([summary_data, geti_default_avg_row.set_index(['task', 'model', 'otx_version'])])

    if task == "instance_segmentation":
        order = ['avg', 'geti_default_avg (non-tile)', 'geti_default_avg (tile)'] + [model for model in summary_data.index.get_level_values('model').unique() if model not in ['avg', 'geti_default_avg (non-tile)', 'geti_default_avg (tile)']]
    else:
        order = ['avg', 'geti_default_avg'] + [model for model in summary_data.index.get_level_values('model').unique() if model not in ['avg', 'geti_default_avg']]
    summary_data = summary_data.reindex(order, level='model')

    # Round all numeric columns to four decimal places
    for col in summary_data.columns:
        if isinstance(col, tuple):  # Check if the column name is a tuple (MultiIndex)
            metric = col[1]  # Get the metric part of the column name
            stat = col[2]    # Get the stat part of the column name
            if metric in ['train/e2e_time', 'train/epoch'] and stat in ['mean', 'std']:
                # Convert to integers
                summary_data[col] = summary_data[col].fillna(0).astype(int)
            else:
                # Round to four decimal places
                summary_data[col] = summary_data[col].round(4)
    return summary_data


def create_graphs(data: pd.DataFrame, metrics: list[str], title_prefix: str) -> list[plt.Figure]:
    """Create and return graphs for the given data and metrics."""
    figures = []
    for metric in metrics:
        fig, ax = plt.subplots()
        pivot_data = data.pivot_table(index=["otx_version"], columns=["model"], values=metric, aggfunc="mean")
        if not pivot_data.empty and pivot_data.count().sum() > 0:
            pivot_data.plot(ax=ax, title=f"{title_prefix} - {metric}", marker="o", legend=True)
            ax.set_xlabel('OTX Version')
            ax.set_ylabel(metric)
            figures.append(fig)
        else:
            print(f"No numeric data available to plot for {metric}.")
            plt.close(fig)
    return figures


def display_graphs_in_row(figures, num_columns):
    """Display a list of matplotlib Figures in a single row with a given number of columns."""
    num_graphs = len(figures)
    num_rows = 1  # We want a single row
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5))
    
    # Flatten the axs array and remove excess axes if there are any
    axs = axs.flatten() if num_columns > 1 else [axs]
    for idx, ax in enumerate(axs):
        if idx < num_graphs:
            figure = figures[idx]
            orig_ax = figure.axes[0]
            for line in orig_ax.lines:
                ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), marker=line.get_marker())
            for patch in orig_ax.patches:
                ax.add_patch(patch)
            ax.set_xlim(orig_ax.get_xlim())
            ax.set_ylim(orig_ax.get_ylim())
            ax.set_xlabel(orig_ax.get_xlabel())
            ax.set_ylabel(orig_ax.get_ylabel())
            ax.set_title(orig_ax.get_title())
            ax.set_xticks(orig_ax.get_xticks())
            ax.set_xticklabels([tick.get_text() for tick in orig_ax.get_xticklabels()])
            if orig_ax.get_legend() is not None:
                ax.legend()
            plt.close(figure)

    # Hide any unused Axes
    for ax in axs[num_graphs:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()


def summarize_graph(history: pd.DataFrame, task: str):
    """Summarize benchmark history graph by task."""
    score_metric = TASK_METRIC_MAP[task]
    metrics = [
        f"test/{score_metric}",
        f"export/{score_metric}",
        f"optimize/{score_metric}",
        "train/iter_time",
        "train/epoch",
        "train/e2e_time",
    ]
    filtered_data = history.query(f"task == '{task}'")
    task_abbr = TASK_ABBR_MAP.get(task, task)  # Get the abbreviation for the task
    title_prefix = f"(avg) {task_abbr}"
    figures = create_graphs(filtered_data, metrics, title_prefix)
    display_graphs_in_row(figures, len(metrics))


def summarize_graph_geti(history: pd.DataFrame, task: str):
    """Summarize benchmark history graph by task using GETI data from raw data."""
    score_metric = TASK_METRIC_MAP[task]
    metrics = [
        f"test/{score_metric}",
        f"export/{score_metric}",
        f"optimize/{score_metric}",
        "train/iter_time",
        "train/epoch",
        "train/e2e_time",
    ]
    task_abbr = TASK_ABBR_MAP.get(task, task)  # Get the abbreviation for the task
    geti_models = GETI_DEFAULT_MODEL.get(task, set())

    # Determine if the task has specific tile types
    if isinstance(geti_models, dict) and task == "instance_segmentation":
        # Handle 'instance_segmentation' task with tile types
        for tile_type, models in geti_models.items():
            models_list = list(models)
            # Filter data for each tile type and create graphs
            tile_filtered_data = history.query(f"task == '{task}' and model in {models_list}")
            title_prefix = f"(geti_default_avg) {task_abbr}_{tile_type}"
            figures = create_graphs(tile_filtered_data, metrics, title_prefix)
            display_graphs_in_row(figures, len(metrics))
            # Create and display graphs for each tile type and data_group
            for data_group in tile_filtered_data['data_group'].unique():
                group_filtered_data = tile_filtered_data.query(f"data_group == '{data_group}'")
                title_prefix = f"(geti_default_{data_group}) {task_abbr}_{tile_type}"
                figures = create_graphs(group_filtered_data, metrics, title_prefix)
                display_graphs_in_row(figures, len(metrics))
    else:
        # Handle general tasks without tile types
        geti_models_list = list(geti_models)
        # Filter data for GETI models and create graphs for 'geti_default_avg'
        geti_filtered_data = history.query(f"task == '{task}' and model in {geti_models_list}")
        title_prefix = f"(geti_default_avg) {task_abbr}"
        figures = create_graphs(geti_filtered_data, metrics, title_prefix)
        display_graphs_in_row(figures, len(metrics))
        # Create and display graphs for each data_group
        for data_group in geti_filtered_data['data_group'].unique():
            group_filtered_data = geti_filtered_data.query(f"data_group == '{data_group}'")
            title_prefix = f"(geti_default_{data_group}) {task_abbr}"
            figures = create_graphs(group_filtered_data, metrics, title_prefix)
            display_graphs_in_row(figures, len(metrics))


def summarize_meta(history: pd.DataFrame):
    """Summarize benchmark metadata by version."""
    data = history.pivot_table(index=["otx_version"], values=METADATA_ENTRIES, aggfunc="first")
    # NOTE: if needed -> data = data.style.set_sticky(axis="index")
    return data.reindex(METADATA_ENTRIES, axis=1)


if __name__ == "__main__":
    """Load csv files in directory & zip files, merge them, summarize per task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root")
    parser.add_argument("output_root")
    parser.add_argument("--pattern", default="*raw*.csv")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    print(f"Loading {args.pattern} in {input_root}...")
    raw_data = load(input_root, need_normalize=args.normalize, pattern=args.pattern)
    if len(raw_data) == 0:
        print("No data loaded")
        sys.exit(-1)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_data.to_csv(output_root / "perf-benchmark-raw-all.csv", index=False)
    print("Saved merged raw data to", str(output_root / "perf-benchmark-raw-all.csv"))

    tasks = sorted(raw_data["task"].unique())
    for task in tasks:
        # Skip action tasks
        if task in ['action_classification', 'action_detection']:
            print(f"Skipping task: {task}")
            continue

        # Use summarize_table function to get a detailed summary for each task
        summary_data = summarize_table(raw_data, task)

        # Save the detailed summary data to an Excel file, including the index
        task_str = task.replace("/", "_")
        summary_data.to_excel(output_root / f"perf-benchmark-summary-{task_str}.xlsx")
        print(f"    Saved {task} summary to", str(output_root / f"perf-benchmark-summary-{task_str}.xlsx"))
