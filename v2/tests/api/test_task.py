import os

import pytest

from otx import OTXConstants
from otx.api.dataset import Dataset
from otx.api.registry import find_task_types, find_tasks
from otx.api.task import Task


def test_task_create():
    task_types = find_task_types()
    assert task_types is not None
    print(f"list of task types = {task_types}")

    task_type = task_types[0]
    tasks = find_tasks(task_types[0])
    assert tasks is not None
    print(f"list of tasks for {task_type} = {tasks}")

    task_yaml = tasks[0]
    assert task_yaml is not None
    print(f"selected task yaml = {task_yaml}")
    task = Task.create(task_yaml)
    assert isinstance(task, Task)


@pytest.fixture(scope="session")
def fixture_task():
    task_types = find_task_types()
    assert task_types is not None
    print(f"list of task types = {task_types}")

    task_type = task_types[0]
    tasks = find_tasks(task_types[0])
    assert tasks is not None
    print(f"list of tasks for {task_type} = {tasks}")

    task_yaml = tasks[0]
    assert task_yaml is not None
    print(f"selected task yaml = {task_yaml}")
    task = Task.create(task_yaml)
    assert isinstance(task, Task)
    return task

@pytest.fixture(scope="session")
def fixture_dataset():
    data_path = os.path.join(OTXConstants.PACKAGE_ROOT, "..", "data")
    dataset = Dataset.create(os.path.join(data_path, "classification/train"), "imagenet", subset="train")
    assert isinstance(dataset, Dataset)
    print(dataset.categories())
    dataset_val = Dataset.create(os.path.join(data_path, "classification/val"), "imagenet", subset="val")
    assert isinstance(dataset_val, Dataset)

    dataset = dataset.update(dataset_val)
    print(f"categories = {dataset.categories()}")
    print(f"subsets = {dataset.subsets()}")

    return dataset


def test_task_apis(fixture_task, fixture_dataset):
    ret = fixture_task.train(fixture_dataset)
    assert isinstance(ret, dict)

    ret = fixture_task.infer(fixture_dataset)
    assert isinstance(ret, dict)

    ret = fixture_task.eval(fixture_dataset, "top-1")
    assert isinstance(ret, dict)

    ret = fixture_task.optimize("pot")
    assert isinstance(ret, dict)

    ret = fixture_task.export("mo")
    assert isinstance(ret, dict)
