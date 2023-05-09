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
    "pot": [],
    "train E2E Time (Sec.)": [],
    "export E2E Time (Sec.)": [],
    "deploy E2E Time (Sec.)": [],
    "nncf E2E Time (Sec.)": [],
    "pot E2E Time (Sec.)": [],
    "train Eval Time (Sec.)": [],
    "export Eval Time (Sec.)": [],
    "deploy Eval Time (Sec.)": [],
    "nncf Eval Time (Sec.)": [],
    "pot Eval Time (Sec.)": [],
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
    "pot": [],
    "train E2E Time (Sec.)": [],
    "export E2E Time (Sec.)": [],
    "deploy E2E Time (Sec.)": [],
    "nncf E2E Time (Sec.)": [],
    "pot E2E Time (Sec.)": [],
    "train Eval Time (Sec.)": [],
    "export Eval Time (Sec.)": [],
    "deploy Eval Time (Sec.)": [],
    "nncf Eval Time (Sec.)": [],
    "pot Eval Time (Sec.)": [],
}


def get_metric_dict(dict_data: Union[List[Dict[str, Any]], None], idx: int, model: str):
    """Get the proper dict item by referencing the index and model information.

    Since all models could be optimized by POT or NNCF, we need to check that there are proper values in the data.
    For example, if model A could be optimized by both POT and NNCF and model B couldn't be supported by POT and NNCF.
    In this case, we have POT, NNCF results about A, however, we don't have POT, NNCF results about B.

    So, if we don't have results, we need to mark the empty result as "-".

    """
    if dict_data and len(dict_data) > idx:
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
    return {"task_key": task_key, "task": task}


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


def summarize_non_anomaly_data(task: str, task_key: str, json_data: dict, result_data: dict) -> dict:
    """Make DataFrame by gathering all results."""
    for label_type in LABEL_TYPES:
        for train_type in TRAIN_TYPES:
            task_data = json_data[task_key][label_type][train_type]

            train_data = task_data.get("train")
            if train_data is None:
                raise ValueError("Train data can't be empty.")
            export_data = task_data.get("export", None)
            deploy_data = task_data.get("deploy", None)
            nncf_data = task_data.get("nncf", None)
            pot_data = task_data.get("pot", None)

            for i, per_model_data in enumerate(train_data):
                for model in per_model_data:
                    train_items = get_metric_items(get_metric_dict(train_data, i, model))
                    export_items = get_metric_items(get_metric_dict(export_data, i, model))
                    deploy_items = get_metric_items(get_metric_dict(deploy_data, i, model))
                    nncf_items = get_metric_items(get_metric_dict(nncf_data, i, model))
                    pot_items = get_metric_items(get_metric_dict(pot_data, i, model))

                    result_data["Task type"].append(task)
                    result_data["Train type"].append(train_type)
                    result_data["Label type"].append(label_type)
                    result_data["Model"].append(model)

                    fill_model_performance(train_items, "train", result_data)
                    fill_model_performance(export_items, "export", result_data)
                    fill_model_performance(deploy_items, "deploy", result_data)
                    fill_model_performance(nncf_items, "nncf", result_data)
                    fill_model_performance(pot_items, "pot", result_data)


def summarize_anomaly_data(task: str, task_key: str, json_data: dict, result_data: dict) -> dict:
    """Make DataFrame by gathering all results."""
    task_data = json_data[task_key]

    train_data = task_data.get("train")
    if train_data is None:
        raise ValueError("Train data can't be empty.")
    export_data = task_data.get("export")
    deploy_data = task_data.get("deploy")
    nncf_data = task_data.get("nncf")
    pot_data = task_data.get("pot")

    for anomaly_category in ANOMALY_DATASET_CATEGORIES:
        train_cat_data = train_data.get(anomaly_category)
        export_cat_data = export_data.get(anomaly_category)
        deploy_cat_data = deploy_data.get(anomaly_category)
        nncf_cat_data = nncf_data.get(anomaly_category)
        pot_cat_data = pot_data.get(anomaly_category)

        for i, per_model_data in enumerate(train_cat_data):
            for model in per_model_data:
                train_items = get_metric_items(get_metric_dict(train_cat_data, i, model))
                export_items = get_metric_items(get_metric_dict(export_cat_data, i, model))
                deploy_items = get_metric_items(get_metric_dict(deploy_cat_data, i, model))
                nncf_items = get_metric_items(get_metric_dict(nncf_cat_data, i, model))
                pot_items = get_metric_items(get_metric_dict(pot_cat_data, i, model))

                result_data["Task type"].append(task)
                result_data["MVTec Category"].append(anomaly_category)
                result_data["Model"].append(model)

                fill_model_performance(train_items, "train", result_data)
                fill_model_performance(export_items, "export", result_data)
                fill_model_performance(deploy_items, "deploy", result_data)
                fill_model_performance(nncf_items, "nncf", result_data)
                fill_model_performance(pot_items, "pot", result_data)


def save_file(result_data: dict, output_path: str):
    df = pd.DataFrame(result_data)
    df.to_csv(output_path)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_path", default="/tmp/regression_test_results", type=str)
    parser.add_argument("--output_path", default="/tmp", type=str)
    return parser.parse_args()


def summarize_data(args):
    input_path = args.input_path

    for root, _, files in os.walk(input_path):
        for result_file in files:
            task_dict = filter_task(root)
            task_key, task = task_dict["task_key"], task_dict["task"]

            json_file_path = os.path.join(root, result_file)
            with open(json_file_path, "r") as f:
                json_data = json.load(f)

            if is_anomaly_task(task) is True:
                summarize_anomaly_data(task, task_key, json_data, ANOMALY_DATA)
            else:
                summarize_non_anomaly_data(task, task_key, json_data, NON_ANOMALY_DATA)

    save_file(ANOMALY_DATA, f"{args.output_path}/anomaly_results.csv")
    save_file(NON_ANOMALY_DATA, f"{args.output_path}/non_anomaly_results.csv")


if __name__ == "__main__":
    args = parse_args()
    summarize_data(args)
