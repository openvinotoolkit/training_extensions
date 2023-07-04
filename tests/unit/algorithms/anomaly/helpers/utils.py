"""Helper utils for tests."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from otx.api.configuration.helper import create as create_hyper_parameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model_template import (
    ModelTemplate,
    TaskType,
    parse_model_template,
)
from otx.api.entities.task_environment import TaskEnvironment


def get_model_template(model_name: str, task_type: str = "classification") -> ModelTemplate:
    template_file_root = Path("src", "otx", "algorithms", "anomaly", "configs", task_type, model_name)
    template_file_path = (
        template_file_root / "template.yaml"
        if (template_file_root / "template.yaml").exists()
        else template_file_root / "template_experimental.yaml"
    )
    model_template: ModelTemplate = parse_model_template(str(template_file_path))
    return model_template


def create_task_environment(dataset: DatasetEntity, task_type: TaskType) -> TaskEnvironment:
    # get padim model template
    padim_template = get_model_template("padim")
    padim_template.task_type = task_type
    hyper_parameters = create_hyper_parameters(padim_template.hyper_parameters.data)
    labels = dataset.get_labels()
    label_schema = LabelSchemaEntity.from_labels(labels)

    return TaskEnvironment(
        model_template=padim_template, model=None, hyper_parameters=hyper_parameters, label_schema=label_schema
    )
