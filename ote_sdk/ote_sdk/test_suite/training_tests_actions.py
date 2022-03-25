# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Type

import pytest

from ote_sdk.configuration.helper import create as ote_sdk_configuration_helper_create
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.model import ModelEntity, ModelFormat, ModelOptimizationType
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.utils.importing import get_impl_class

from .e2e_test_system import DataCollector
from .logging import get_logger
from .training_tests_common import (
    KEEP_CONFIG_FIELD_VALUE,
    performance_to_score_name_value,
)

logger = get_logger()


class BaseOTETestAction(ABC):
    _name: Optional[str] = None
    _with_validation = False
    _depends_stages_names: List[str] = []

    def __init__(*args, **kwargs):
        pass

    @property
    def name(self):
        return type(self)._name

    @property
    def with_validation(self):
        return type(self)._with_validation

    @property
    def depends_stages_names(self):
        return type(self)._depends_stages_names

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"name={self.name}, "
            f"with_validation={self.with_validation}, "
            f"depends_stages_names={self.depends_stages_names})"
        )

    def _check_result_prev_stages(self, results_prev_stages, list_required_stages):
        for stage_name in list_required_stages:
            if not results_prev_stages or stage_name not in results_prev_stages:
                raise RuntimeError(
                    f"The action {self.name} requires results of the stage {stage_name}, "
                    f"but they are absent"
                )

    @abstractmethod
    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        raise NotImplementedError("The main action method is not implemented")


def create_environment_and_task(params, labels_schema, model_template):
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template,
    )
    logger.info("Create base Task")
    task_impl_path = model_template.entrypoints.base
    task_cls = get_impl_class(task_impl_path)
    task = task_cls(task_environment=environment)
    return environment, task


class OTETestTrainingAction(BaseOTETestAction):
    _name = "training"

    def __init__(
        self, dataset, labels_schema, template_path, num_training_iters, batch_size
    ):
        self.dataset = dataset
        self.labels_schema = labels_schema
        self.template_path = template_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size

    def _get_training_performance_as_score_name_value(self):
        training_performance = getattr(self.output_model, "performance", None)
        if training_performance is None:
            raise RuntimeError("Cannot get training performance")
        return performance_to_score_name_value(training_performance)

    def _run_ote_training(self, data_collector):
        logger.debug(f"self.template_path = {self.template_path}")

        print(f"train dataset: {len(self.dataset.get_subset(Subset.TRAINING))} items")
        print(
            f"validation dataset: "
            f"{len(self.dataset.get_subset(Subset.VALIDATION))} items"
        )

        logger.debug("Load model template")
        self.model_template = parse_model_template(self.template_path)

        logger.debug("Set hyperparameters")
        params = ote_sdk_configuration_helper_create(
            self.model_template.hyper_parameters.data
        )
        if self.num_training_iters != KEEP_CONFIG_FIELD_VALUE:
            params.learning_parameters.num_iters = int(self.num_training_iters)
            logger.debug(
                f"Set params.learning_parameters.num_iters="
                f"{params.learning_parameters.num_iters}"
            )
        else:
            logger.debug(
                f"Keep params.learning_parameters.num_iters="
                f"{params.learning_parameters.num_iters}"
            )

        if self.batch_size != KEEP_CONFIG_FIELD_VALUE:
            params.learning_parameters.batch_size = int(self.batch_size)
            logger.debug(
                f"Set params.learning_parameters.batch_size="
                f"{params.learning_parameters.batch_size}"
            )
        else:
            logger.debug(
                f"Keep params.learning_parameters.batch_size="
                f"{params.learning_parameters.batch_size}"
            )

        logger.debug("Setup environment")
        self.environment, self.task = create_environment_and_task(
            params, self.labels_schema, self.model_template
        )

        logger.debug("Train model")
        self.output_model = ModelEntity(
            self.dataset,
            self.environment.get_model_configuration(),
        )

        self.copy_hyperparams = deepcopy(self.task._hyperparams)

        try:
            self.task.train(self.dataset, self.output_model)
        except Exception as ex:
            raise RuntimeError("Training failed") from ex

        score_name, score_value = self._get_training_performance_as_score_name_value()
        logger.info(f"performance={self.output_model.performance}")
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._run_ote_training(data_collector)
        results = {
            "model_template": self.model_template,
            "task": self.task,
            "dataset": self.dataset,
            "environment": self.environment,
            "output_model": self.output_model,
        }
        return results


def is_nncf_enabled():
    return importlib.util.find_spec("nncf") is not None


def run_evaluation(dataset, task, model):
    logger.debug("Evaluation: Get predictions on the dataset")
    predicted_dataset = task.infer(
        dataset.with_empty_annotations(), InferenceParameters(is_evaluation=True)
    )
    resultset = ResultSetEntity(
        model=model,
        ground_truth_dataset=dataset,
        prediction_dataset=predicted_dataset,
    )
    logger.debug("Evaluation: Estimate quality on dataset")
    task.evaluate(resultset)
    evaluation_performance = resultset.performance
    logger.info(f"Evaluation: performance={evaluation_performance}")
    score_name, score_value = performance_to_score_name_value(evaluation_performance)
    return score_name, score_value


class OTETestTrainingEvaluationAction(BaseOTETestAction):
    _name = "training_evaluation"
    _with_validation = True
    _depends_stages_names = ["training"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_evaluation(self, data_collector, dataset, task, trained_model):
        logger.info("Begin evaluation of trained model")
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset, task, trained_model
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info(
            f"End evaluation of trained model, results: {score_name}: {score_value}"
        )
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "dataset": results_prev_stages["training"]["dataset"],
            "task": results_prev_stages["training"]["task"],
            "trained_model": results_prev_stages["training"]["output_model"],
        }

        score_name, score_value = self._run_ote_evaluation(data_collector, **kwargs)
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


def run_export(environment, dataset, task, action_name, expected_optimization_type):
    logger.debug(
        f'For action "{action_name}": Copy environment for evaluation exported model'
    )

    environment_for_export = deepcopy(environment)

    logger.debug(f'For action "{action_name}": Create exported model')
    exported_model = ModelEntity(
        dataset,
        environment_for_export.get_model_configuration(),
    )
    logger.debug("Run export")

    try:
        task.export(ExportType.OPENVINO, exported_model)
    except Exception as ex:
        raise RuntimeError("Export to OpenVINO failed") from ex

    assert (
        exported_model.model_format == ModelFormat.OPENVINO
    ), f"In action '{action_name}': Wrong model format after export. " \
       f"exported_model.model_format=exported_model.model_format"
    assert (
        exported_model.optimization_type == expected_optimization_type
    ), f"In action '{action_name}': Wrong optimization type. " \
       f"exported_model.optimization_type={exported_model.optimization_type}"

    logger.debug(
        f'For action "{action_name}": Set exported model into environment for export'
    )
    environment_for_export.model = exported_model
    return environment_for_export, exported_model


class OTETestExportAction(BaseOTETestAction):
    _name = "export"
    _depends_stages_names = ["training"]

    def _run_ote_export(self, data_collector, environment, dataset, task):
        self.environment_for_export, self.exported_model = run_export(
            environment,
            dataset,
            task,
            action_name=self.name,
            expected_optimization_type=ModelOptimizationType.MO,
        )

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "environment": results_prev_stages["training"]["environment"],
            "dataset": results_prev_stages["training"]["dataset"],
            "task": results_prev_stages["training"]["task"],
        }

        self._run_ote_export(data_collector, **kwargs)
        results = {
            "environment": self.environment_for_export,
            "exported_model": self.exported_model,
        }
        return results


def create_openvino_task(model_template, environment):
    logger.debug("Create OpenVINO Task")
    openvino_task_impl_path = model_template.entrypoints.openvino
    openvino_task_cls = get_impl_class(openvino_task_impl_path)
    openvino_task = openvino_task_cls(environment)
    return openvino_task


class OTETestExportEvaluationAction(BaseOTETestAction):
    _name = "export_evaluation"
    _with_validation = True
    _depends_stages_names = ["training", "export", "training_evaluation"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_export_evaluation(
        self,
        data_collector,
        model_template,
        dataset,
        environment_for_export,
        exported_model,
    ):
        logger.info("Begin evaluation of exported model")
        self.openvino_task = create_openvino_task(
            model_template, environment_for_export
        )
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset, self.openvino_task, exported_model
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info("End evaluation of exported model")
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "model_template": results_prev_stages["training"]["model_template"],
            "dataset": results_prev_stages["training"]["dataset"],
            "environment_for_export": results_prev_stages["export"]["environment"],
            "exported_model": results_prev_stages["export"]["exported_model"],
        }

        score_name, score_value = self._run_ote_export_evaluation(
            data_collector, **kwargs
        )
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


class OTETestPotAction(BaseOTETestAction):
    _name = "pot"
    _depends_stages_names = ["export"]

    def __init__(self, pot_subset=Subset.TRAINING):
        self.pot_subset = pot_subset

    def _run_ote_pot(
        self, data_collector, model_template, dataset, environment_for_export
    ):
        logger.debug("Creating environment and task for POT optimization")
        self.environment_for_pot = deepcopy(environment_for_export)
        self.openvino_task_pot = create_openvino_task(
            model_template, environment_for_export
        )

        self.optimized_model_pot = ModelEntity(
            dataset,
            self.environment_for_pot.get_model_configuration(),
        )
        logger.info("Run POT optimization")

        try:
            self.openvino_task_pot.optimize(
                OptimizationType.POT,
                dataset.get_subset(self.pot_subset),
                self.optimized_model_pot,
                OptimizationParameters(),
            )
        except Exception as ex:
            raise RuntimeError("POT optimization failed") from ex

        assert (
            self.optimized_model_pot.model_format == ModelFormat.OPENVINO
        ), "Wrong model format after pot"
        assert (
            self.optimized_model_pot.optimization_type == ModelOptimizationType.POT
        ), "Wrong optimization type"
        logger.info("POT optimization is finished")

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "model_template": results_prev_stages["training"]["model_template"],
            "dataset": results_prev_stages["training"]["dataset"],
            "environment_for_export": results_prev_stages["export"]["environment"],
        }

        self._run_ote_pot(data_collector, **kwargs)
        results = {
            "openvino_task_pot": self.openvino_task_pot,
            "optimized_model_pot": self.optimized_model_pot,
        }
        return results


class OTETestPotEvaluationAction(BaseOTETestAction):
    _name = "pot_evaluation"
    _with_validation = True
    _depends_stages_names = ["training", "pot", "export_evaluation"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_pot_evaluation(
        self, data_collector, dataset, openvino_task_pot, optimized_model_pot
    ):
        logger.info("Begin evaluation of pot model")
        validation_dataset_pot = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset_pot, openvino_task_pot, optimized_model_pot
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info("End evaluation of pot model")
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "dataset": results_prev_stages["training"]["dataset"],
            "openvino_task_pot": results_prev_stages["pot"]["openvino_task_pot"],
            "optimized_model_pot": results_prev_stages["pot"]["optimized_model_pot"],
        }

        score_name, score_value = self._run_ote_pot_evaluation(data_collector, **kwargs)
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


class OTETestNNCFAction(BaseOTETestAction):
    _name = "nncf"
    _depends_stages_names = ["training"]

    def _run_ote_nncf(
        self, data_collector, model_template, dataset, trained_model, environment
    ):
        logger.debug("Get predictions on the validation set for exported model")
        self.environment_for_nncf = deepcopy(environment)

        logger.info("Create NNCF Task")
        nncf_task_class_impl_path = model_template.entrypoints.nncf
        if not nncf_task_class_impl_path:
            pytest.skip("NNCF is not enabled for this template")

        if not is_nncf_enabled():
            pytest.skip("NNCF is not installed")

        logger.info("Creating NNCF task and structures")
        self.nncf_model = ModelEntity(
            dataset,
            self.environment_for_nncf.get_model_configuration(),
        )
        self.nncf_model.set_data("weights.pth", trained_model.get_data("weights.pth"))

        self.environment_for_nncf.model = self.nncf_model

        nncf_task_cls = get_impl_class(nncf_task_class_impl_path)
        self.nncf_task = nncf_task_cls(task_environment=self.environment_for_nncf)

        logger.info("Run NNCF optimization")
        try:
            self.nncf_task.optimize(
                OptimizationType.NNCF, dataset, self.nncf_model, None
            )
        except Exception as ex:
            raise RuntimeError("NNCF optimization failed") from ex

        assert (
            self.nncf_model.optimization_type == ModelOptimizationType.NNCF
        ), "Wrong optimization type"
        assert (
            self.nncf_model.model_format == ModelFormat.BASE_FRAMEWORK
        ), "Wrong model format"

        logger.info("NNCF optimization is finished")

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "model_template": results_prev_stages["training"]["model_template"],
            "dataset": results_prev_stages["training"]["dataset"],
            "trained_model": results_prev_stages["training"]["output_model"],
            "environment": results_prev_stages["training"]["environment"],
        }

        self._run_ote_nncf(data_collector, **kwargs)
        results = {
            "nncf_task": self.nncf_task,
            "nncf_model": self.nncf_model,
            "nncf_environment": self.environment_for_nncf,
        }
        return results


# TODO: think about move to special file
def check_nncf_model_graph(model, path_to_dot):
    import networkx as nx

    logger.info(f"Reference graph: {path_to_dot}")
    load_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)

    graph = model.get_graph()
    nx_graph = graph.get_graph_for_structure_analysis()

    for _, node in nx_graph.nodes(data=True):
        if "scope" in node:
            node.pop("scope")

    for k, attrs in nx_graph.nodes.items():
        attrs = {k: str(v) for k, v in attrs.items()}
        load_attrs = {k: str(v).strip('"') for k, v in load_graph.nodes[k].items()}
        if "scope" in load_attrs:
            load_attrs.pop("scope")
        if attrs != load_attrs:
            logger.info("ATTR: {} : {} != {}".format(k, attrs, load_attrs))
            return False

    return (
        load_graph.nodes.keys() == nx_graph.nodes.keys()
        and nx.DiGraph(load_graph).edges == nx_graph.edges
    )


class OTETestNNCFGraphAction(BaseOTETestAction):
    _name = "nncf_graph"

    def __init__(
        self,
        dataset,
        labels_schema,
        template_path,
        reference_dir,
        fn_get_compressed_model,
    ):
        self.dataset = dataset
        self.labels_schema = labels_schema
        self.template_path = template_path
        self.reference_dir = reference_dir
        self.fn_get_compressed_model = fn_get_compressed_model

    def _run_ote_nncf_graph(self, data_collector):
        # pylint:disable=protected-access
        logger.debug("Load model template")
        model_template = parse_model_template(self.template_path)
        nncf_task_class_impl_path = model_template.entrypoints.nncf

        if not nncf_task_class_impl_path:
            pytest.skip("NNCF is not enabled for this template")

        if not is_nncf_enabled():
            pytest.skip("NNCF is not installed")

        if not os.path.exists(self.reference_dir):
            pytest.skip("Reference directory does not exist")

        params = ote_sdk_configuration_helper_create(
            model_template.hyper_parameters.data
        )
        environment, task = create_environment_and_task(
            params, self.labels_schema, model_template
        )
        output_model = ModelEntity(
            self.dataset,
            environment.get_model_configuration(),
        )
        # Save model without training to create nncf_task
        task.save_model(output_model)

        logger.info("Create NNCF Task")
        environment_for_nncf = deepcopy(environment)

        logger.info("Creating NNCF task and structures")
        nncf_model = ModelEntity(
            self.dataset,
            environment_for_nncf.get_model_configuration(),
        )
        nncf_model.set_data("weights.pth", output_model.get_data("weights.pth"))

        environment_for_nncf.model = nncf_model

        nncf_task_cls = get_impl_class(nncf_task_class_impl_path)
        nncf_task = nncf_task_cls(task_environment=environment_for_nncf)

        path_to_ref_dot = os.path.join(
            self.reference_dir, "nncf", f"{nncf_task._nncf_preset}.dot"
        )
        if not os.path.exists(path_to_ref_dot):
            pytest.skip("Reference file does not exist: {}".format(path_to_ref_dot))

        compressed_model = self.fn_get_compressed_model(nncf_task)

        assert check_nncf_model_graph(
            compressed_model, path_to_ref_dot
        ), "Compressed model differs from the reference"

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)
        self._run_ote_nncf_graph(data_collector)
        return {}


class OTETestNNCFEvaluationAction(BaseOTETestAction):
    _name = "nncf_evaluation"
    _with_validation = True
    _depends_stages_names = ["training", "nncf", "training_evaluation"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_nncf_evaluation(self, data_collector, dataset, nncf_task, nncf_model):
        logger.info("Begin evaluation of nncf model")
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset, nncf_task, nncf_model
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info("End evaluation of nncf model")
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "dataset": results_prev_stages["training"]["dataset"],
            "nncf_task": results_prev_stages["nncf"]["nncf_task"],
            "nncf_model": results_prev_stages["nncf"]["nncf_model"],
        }

        score_name, score_value = self._run_ote_nncf_evaluation(
            data_collector, **kwargs
        )
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


class OTETestNNCFExportAction(BaseOTETestAction):
    _name = "nncf_export"
    _depends_stages_names = ["training", "nncf"]

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_nncf_export(
        self, data_collector, nncf_environment, dataset, nncf_task
    ):
        logger.info("Begin export of nncf model")
        self.environment_nncf_export, self.nncf_exported_model = run_export(
            nncf_environment,
            dataset,
            nncf_task,
            action_name=self.name,
            expected_optimization_type=ModelOptimizationType.NNCF,
        )
        logger.info("End export of nncf model")

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "nncf_environment": results_prev_stages["nncf"]["nncf_environment"],
            "dataset": results_prev_stages["training"]["dataset"],
            "nncf_task": results_prev_stages["nncf"]["nncf_task"],
        }

        self._run_ote_nncf_export(data_collector, **kwargs)
        results = {
            "environment": self.environment_nncf_export,
            "exported_model": self.nncf_exported_model,
        }
        return results


class OTETestNNCFExportEvaluationAction(BaseOTETestAction):
    _name = "nncf_export_evaluation"
    _with_validation = True
    _depends_stages_names = ["training", "nncf_export", "nncf_evaluation"]

    def __init__(self, subset=Subset.TESTING):
        self.subset = subset

    def _run_ote_nncf_export_evaluation(
        self,
        data_collector,
        model_template,
        dataset,
        nncf_environment_for_export,
        nncf_exported_model,
    ):
        logger.info("Begin evaluation of NNCF exported model")
        self.openvino_task = create_openvino_task(
            model_template, nncf_environment_for_export
        )
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(
            validation_dataset, self.openvino_task, nncf_exported_model
        )
        data_collector.log_final_metric("metric_name", self.name + "/" + score_name)
        data_collector.log_final_metric("metric_value", score_value)
        logger.info("End evaluation of NNCF exported model")
        return score_name, score_value

    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
        self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)

        kwargs = {
            "model_template": results_prev_stages["training"]["model_template"],
            "dataset": results_prev_stages["training"]["dataset"],
            "nncf_environment_for_export": results_prev_stages["nncf_export"][
                "environment"
            ],
            "nncf_exported_model": results_prev_stages["nncf_export"]["exported_model"],
        }

        score_name, score_value = self._run_ote_nncf_export_evaluation(
            data_collector, **kwargs
        )
        results = {"metrics": {"accuracy": {score_name: score_value}}}
        return results


def get_default_test_action_classes() -> List[Type[BaseOTETestAction]]:
    return [
        OTETestTrainingAction,
        OTETestTrainingEvaluationAction,
        OTETestExportAction,
        OTETestExportEvaluationAction,
        OTETestPotAction,
        OTETestPotEvaluationAction,
        OTETestNNCFAction,
        OTETestNNCFEvaluationAction,
        OTETestNNCFExportAction,
        OTETestNNCFExportEvaluationAction,
        OTETestNNCFGraphAction,
    ]
