import argparse
import json
import os
from typing import Dict, Union, List, Any

import pandas as pd

from tests.regression.regression_test_helpers import (
    ANOMALY_DATASET_CATEGORIES,
    LABEL_TYPES,
    TRAIN_TYPES,
)


ANOMALY_DATA = {
    "Task type": [],
    "MVTec Category": [],
    "Model": [],
    "train": [],
    "export": [],
    "deploy": [],
    "nncf": [],
    "ptq": [],
    "train E2E Time (Sec.)": [],
    "export E2E Time (Sec.)": [],
    "deploy E2E Time (Sec.)": [],
    "nncf E2E Time (Sec.)": [],
    "ptq E2E Time (Sec.)": [],
    "train Eval Time (Sec.)": [],
    "export Eval Time (Sec.)": [],
    "deploy Eval Time (Sec.)": [],
    "nncf Eval Time (Sec.)": [],
    "ptq Eval Time (Sec.)": [],
}

NON_ANOMALY_DATA = {
    "Task type": [],
    "Train type": [],
    "Label type": [],
    "Model": [],
    "train": [],
    "export": [],
    "deploy": [],
    "nncf": [],
    "ptq": [],
    "train E2E Time (Sec.)": [],
    "export E2E Time (Sec.)": [],
    "deploy E2E Time (Sec.)": [],
    "nncf E2E Time (Sec.)": [],
    "ptq E2E Time (Sec.)": [],
    "train Eval Time (Sec.)": [],
    "export Eval Time (Sec.)": [],
    "deploy Eval Time (Sec.)": [],
    "nncf Eval Time (Sec.)": [],
    "ptq Eval Time (Sec.)": [],
}


def get_metric_dict(dict_data: Union[List[Dict[str, Any]], None], idx: int, model: str):
    """Get the proper dict item by referencing the index and model information.

    Since all models could be optimized by PTQ or NNCF, we need to check that there are proper values in the data.
    For example, if model A could be optimized by both PTQ and NNCF and model B couldn't be supported by PTQ and NNCF.
    In this case, we have PTQ, NNCF results about A, however, we don't have PTQ, NNCF results about B.

    So, if we don't have results, we need to mark the empty result as "-".

    """
    if dict_data and len(dict_data) > idx:
        if dict_data[idx].get(model) is None:
            return "-"
        return dict_data[idx][model]
    else:
        return "-"


def get_metric_items(input_data: Union[str, List[Dict[str, Any]]]):
    """Divide the data by using the model name.

    i.e.
    input_data : {
        'A': {
            'Accuracy': 0.5,
            'Model size (MB)': 12.65,
            ...
        },
        'B':{
            ...
        }
    }

    --> return_list: [(A, {'Accuracy': 0.5, 'Model size(MB)': 12.65, ...}), (B, {...})]

    """
    if isinstance(input_data, dict):
        return_list = []
        for k, v in input_data.items():
            return_list.append((k, v))
        return return_list
    else:
        return "-"


def filter_task(root: str) -> Dict[str, str]:
    """Find prpoer task and task_key."""
    task = root.split("/")[-1]
    if "tiling" in task:
        task_key = "_".join(task.split("_")[1:])
    else:
        task_key = task
    return task_key, task


def is_anomaly_task(task: str) -> bool:
    """Returns True if task is anomaly."""
    return "anomaly" in task


def fill_model_performance(items: Union[list, str], test_type: str, result_data: dict):
    """Fill the result_data by checking the index of data."""
    if isinstance(items, list):
        result_data[test_type].append(f"{items[0][0]}: {items[0][1]}")
        if test_type == "train":
            result_data[f"{test_type} E2E Time (Sec.)"].append(f"{items[2][1]}")
            result_data[f"{test_type} Eval Time (Sec.)"].append(f"{items[3][1]}")
        else:
            result_data[f"{test_type} E2E Time (Sec.)"].append(f"{items[1][1]}")
            result_data[f"{test_type} Eval Time (Sec.)"].append(f"{items[2][1]}")
    else:
        result_data[test_type].append(items)
        result_data[f"{test_type} E2E Time (Sec.)"].append(items)
        result_data[f"{test_type} Eval Time (Sec.)"].append(items)


def summarize_non_anomaly_data(json_data: dict, result_data: dict) -> dict:
    """Make DataFrame by gathering all results."""
    for task_key in json_data.keys():
        for label_type in LABEL_TYPES:
            if label_type not in json_data[task_key].keys():
                continue
            for train_type in TRAIN_TYPES:
                if train_type not in json_data[task_key][label_type].keys():
                    continue
                task_data = json_data[task_key][label_type][train_type]

                train_data = task_data.get("train")
                if train_data is None:
                    raise ValueError("Train data can't be empty.")
                export_data = task_data.get("export", None)
                deploy_data = task_data.get("deploy", None)
                nncf_data = task_data.get("nncf", None)
                ptq_data = task_data.get("ptq", None)

                for i, per_model_data in enumerate(train_data):
                    for model in per_model_data:
                        train_items = get_metric_items(get_metric_dict(train_data, i, model))
                        export_items = get_metric_items(get_metric_dict(export_data, i, model))
                        deploy_items = get_metric_items(get_metric_dict(deploy_data, i, model))
                        nncf_items = get_metric_items(get_metric_dict(nncf_data, i, model))
                        ptq_items = get_metric_items(get_metric_dict(ptq_data, i, model))

                        result_data["Task type"].append(task_key)
                        result_data["Train type"].append(train_type)
                        result_data["Label type"].append(label_type)
                        result_data["Model"].append(model)

                        fill_model_performance(train_items, "train", result_data)
                        fill_model_performance(export_items, "export", result_data)
                        fill_model_performance(deploy_items, "deploy", result_data)
                        fill_model_performance(nncf_items, "nncf", result_data)
                        fill_model_performance(ptq_items, "ptq", result_data)


def summarize_anomaly_data(json_data: dict, result_data: dict) -> dict:
    """Make DataFrame by gathering all results."""
    for task_key in json_data.keys():
        task_data = json_data[task_key]

        train_data = task_data.get("train")
        if train_data is None:
            raise ValueError("Train data can't be empty.")
        export_data = task_data.get("export")
        deploy_data = task_data.get("deploy")
        nncf_data = task_data.get("nncf")
        ptq_data = task_data.get("ptq")

        for anomaly_category in ANOMALY_DATASET_CATEGORIES:
            train_cat_data = train_data.get(anomaly_category)
            if train_cat_data is None:
                continue
            export_cat_data = export_data.get(anomaly_category)
            deploy_cat_data = deploy_data.get(anomaly_category)
            nncf_cat_data = nncf_data.get(anomaly_category)
            ptq_cat_data = ptq_data.get(anomaly_category)

            for i, per_model_data in enumerate(train_cat_data):
                for model in per_model_data:
                    train_items = get_metric_items(get_metric_dict(train_cat_data, i, model))
                    export_items = get_metric_items(get_metric_dict(export_cat_data, i, model))
                    deploy_items = get_metric_items(get_metric_dict(deploy_cat_data, i, model))
                    nncf_items = get_metric_items(get_metric_dict(nncf_cat_data, i, model))
                    ptq_items = get_metric_items(get_metric_dict(ptq_cat_data, i, model))

                    result_data["Task type"].append(task_key)
                    result_data["MVTec Category"].append(anomaly_category)
                    result_data["Model"].append(model)

                    fill_model_performance(train_items, "train", result_data)
                    fill_model_performance(export_items, "export", result_data)
                    fill_model_performance(deploy_items, "deploy", result_data)
                    fill_model_performance(nncf_items, "nncf", result_data)
                    fill_model_performance(ptq_items, "ptq", result_data)


def save_file(result_data: dict, output_path: str, file_name: str):
    df = pd.DataFrame(result_data)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, file_name))


def merge_reg_results_dict(target, source, overwrite=False):
    target = target.copy()
    for k, v in source.items():
        if isinstance(v, Dict):
            if k in target:
                target[k] = merge_reg_results_dict(target[k], v)
            else:
                target[k] = v
        elif isinstance(v, List):
            if len(target[k]) == 0 or overwrite:
                target[k] = v
    return target


def merge_results_list(results_list: List[Dict]):
    if len(results_list) == 1:
        return results_list[0]
    results_dict = {}
    for results in results_list:
        results_dict = merge_reg_results_dict(results_dict, results)
    return results_dict


def summarize_results_data(input_path: str, output_path: str):
    """summarize regression test result data."""
    input_path = input_path

    for entity in os.listdir(input_path):
        entity_path = os.path.join(input_path, entity)
        if os.path.isdir(entity_path):
            _, task = filter_task(entity_path)
            results_list = []
            for result_json in os.listdir(entity_path):
                result_json_path = os.path.join(entity_path, result_json)
                if os.path.isfile(result_json_path) and result_json_path.split(".")[-1] == "json":
                    with open(result_json_path, "r") as f:
                        results_list.append(json.load(f))
            json_data = merge_results_list(results_list)

            assert len(json_data) != 0, "no json results to summary"

            if is_anomaly_task(task) is True:
                summarize_anomaly_data(json_data, ANOMALY_DATA)
                save_file(ANOMALY_DATA, output_path, f"tests-reg_{task}.csv")
            else:
                summarize_non_anomaly_data(json_data, NON_ANOMALY_DATA)
                save_file(NON_ANOMALY_DATA, output_path, f"tests-reg_{task}.csv")
