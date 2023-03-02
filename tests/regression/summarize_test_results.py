import json
import pandas as pd
import os
import sys
import argparse
from typing import Union, Optional, Dict, List, Any

from tests.regression.regression_test_helpers import TRAIN_TYPES, LABEL_TYPES, TEST_TYPES

result_data = {
    'Task type': [], 
    'Train type': [], 
    'Label type': [], 
    'Model': [], 
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
    "pot Eval Time (Sec.)": []
}

def get_metric_dict(dict_type, idx: int, model: str):
    if dict_type:
        return dict_type[idx][model]
    else:
        return "-"

def get_metric_items(input_data):
    if isinstance(input_data, dict):
        return_list = []
        for k, v in input_data.items():
            return_list.append((k, v))
        return return_list
    else:
        return "-"

def filter_task(root: str) -> Dict[str, str]:
    """Find prpoer task and task_key."""
    task = root.split('/')[-1]
    if "tiling" in task:
        task_key = '_'.join(task.split('_')[1:])
    else:
        task_key = task
    return {'task_key': task_key, 'task': task}

def is_anomaly_task(task: str) -> bool:
    """Returns True if task is anomaly."""
    return "anomaly" in task

def fill_train_task_label_model_data(task: str, train_type: str, label_type: str, model: str):
    """Fill the task, lable, model filed of result_data."""
    result_data['Task type'].append(task)
    result_data['Train type'].append(train_type)
    result_data['Label type'].append(label_type)
    result_data['Model'].append(model)

def fill_model_performance(items: Union[list, str], test_type: str):
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
         

def set_test_type_data(task: str, task_data: Dict[str, Any], train_type: str, label_type: str, test_type: str):
    """Get the proper data according to test type."""
    test_type_data = task_data.get(test_type, None)
    
    if test_type_data is not None:
        for i, per_model_data in enumerate(test_type_data):
            for model in per_model_data:
                items = get_metric_items(get_metric_dict(test_type_data, i, model))
                print(result_data, test_type)
                fill_train_task_label_model_data(task, train_type, label_type, model)
                fill_model_performance(items, test_type)
                
    raise
def summarize_data(task:str, task_key:str, result_data: dict):
    """Make DataFrame by gathering all results."""
    for label_type in LABEL_TYPES:
        for train_type in TRAIN_TYPES:
            if is_anomaly_task(task) is False: 
                task_data = result_data[task_key][label_type][train_type]
                for test_type in TEST_TYPES:
                    set_test_type_data(task, task_data, train_type, label_type, test_type)
    
def save_file(output_path):
    df = pd.DataFrame(result_data)
    df.to_csv(output_path)

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_path", default="/tmp/regression_test_results", type=str)
    parser.add_argument("--output_path", default="regression_results.csv", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input_path
    
    for root, _, files in os.walk(input_path):
        for result_file in files:
            task_dict = filter_task(root)
            task_key, task = task_dict['task_key'], task_dict['task']
            
            json_file_path = os.path.join(root, result_file)
            with open(json_file_path, "r") as f:
                result_data = json.load(f)
                
            summarize_data(task, task_key, result_data)    
    raise 
    save_file(args.output_path) 

if __name__ == '__main__':
    main()