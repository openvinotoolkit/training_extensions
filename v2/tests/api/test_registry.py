import pytest
from otx.api.registry import find_task_types, find_tasks, find_models, find_backbones


def test_registry():
    task_types = find_task_types()
    assert task_types is not None
    print(f"list of task types = {task_types}")

    task_type = task_types[0]
    tasks = find_tasks(task_types[0])
    assert tasks is not None
    print(f"list of tasks for {task_type} = {tasks}")

    task = tasks[0]
    models = find_models(task)
    assert models is not None
    print(f"list of models for {task} = {models}")

    model = models[0]
    backbones = find_backbones(model)
    assert backbones is not None
    print(f"list of backbones for {model} = {backbones}")
