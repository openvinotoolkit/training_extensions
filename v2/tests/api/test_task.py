import pytest

from otx.api.task import Task
from otx.api.registry import find_task_types, find_tasks

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

def test_task_apis(fixture_task):
    fixture_task.train(None)