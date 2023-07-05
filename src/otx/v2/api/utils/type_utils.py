from otx.v2.api.entities.subset import Subset
from otx.v2.api.entities.task_type import TaskType, TrainType

map_task_type = {str(task_type.name).upper(): task_type for task_type in TaskType}
map_train_type = {str(train_type.name).upper(): train_type for train_type in TrainType}
map_subset_type = {str(subset_type.name).upper(): subset_type for subset_type in Subset}


def str_to_task_type(task_type: str):
    if task_type.upper() in map_task_type:
        return map_task_type[task_type.upper()]
    raise ValueError(f"{task_type.upper()} is not supported task.")


def str_to_train_type(train_type: str):
    if train_type.upper() in map_train_type:
        return map_train_type[train_type.upper()]
    raise ValueError(f"{train_type.upper()} is not supported train type.")


def str_to_subset_type(subset: str):
    map_short_str = {"train": "training", "val": "validation", "test": "testing", "unlabel": "unlabeled"}
    if subset in map_short_str:
        subset = map_short_str[subset]
    if subset.upper() in map_subset_type:
        return map_subset_type[subset.upper()]
    raise ValueError(f"{subset.upper()} is not supported subset type.")
