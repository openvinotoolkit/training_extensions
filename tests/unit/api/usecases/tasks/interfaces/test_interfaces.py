# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from unittest.mock import patch

import pytest

from otx.api.configuration import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity, ModelPrecision
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import (
    IInferenceTask,
    IRawInference,
)
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.api.usecases.tasks.interfaces.training_interface import ITrainingTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIEvaluationTask:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        "otx.api.usecases.tasks.interfaces.evaluate_interface.IEvaluationTask.__abstractmethods__",
        set(),
    )
    def test_evaluate_interface(self):
        """
        <b>Description:</b>
        Check IEvaluationTask class object initialization

        <b>Input data:</b>
        IEvaluationTask object

        <b>Expected results:</b>
        Test passes if IEvaluationTask object evaluate method raises NotImplementedError exception
        """
        dataset = DatasetEntity()
        configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="Test Header"),
            label_schema=LabelSchemaEntity(),
        )
        model_entity = ModelEntity(configuration=configuration, train_dataset=dataset)
        with pytest.raises(NotImplementedError):
            IEvaluationTask().evaluate(
                ResultSetEntity(
                    model=model_entity,
                    ground_truth_dataset=dataset,
                    prediction_dataset=dataset,
                )
            )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIInferenceTask:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        "otx.api.usecases.tasks.interfaces.inference_interface.IInferenceTask.__abstractmethods__",
        set(),
    )
    def test_i_inference_task(self):
        """
        <b>Description:</b>
        Check IInferenceTask class object initialization

        <b>Input data:</b>
        IInferenceTask object

        <b>Expected results:</b>
        Test passes if IInferenceTask object infer method raises NotImplementedError exception
        """
        dataset = DatasetEntity()
        inference_parameters = InferenceParameters()
        with pytest.raises(NotImplementedError):
            IInferenceTask().infer(dataset=dataset, inference_parameters=inference_parameters)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIRawInference:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        "otx.api.usecases.tasks.interfaces.inference_interface.IRawInference.__abstractmethods__",
        set(),
    )
    def test_i_raw_inference(self):
        """
        <b>Description:</b>
        Check TestIRawInference class object initialization

        <b>Input data:</b>
        TestIRawInference object

        <b>Expected results:</b>
        Test passes if TestIRawInference object raw_infer method raises NotImplementedError exception
        """
        with pytest.raises(NotImplementedError):
            IRawInference().raw_infer(input_tensors={}, output_tensors={})


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestOptimizationType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_optimization_type(self):
        """
        <b>Description:</b>
        Check OptimizationType Enum class elements

        <b>Expected results:</b>
        Test passes if OptimizationType Enum class length is equal to expected value and its elements have expected
        sequence numbers
        """
        assert len(OptimizationType) == 2
        assert OptimizationType.POT.value == 1
        assert OptimizationType.NNCF.value == 2


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIOptimizationTask:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        "otx.api.usecases.tasks.interfaces.optimization_interface.IOptimizationTask.__abstractmethods__",
        set(),
    )
    def test_optimization_interface(self):
        """
        <b>Description:</b>
        Check IOptimizationTask class object initialization

        <b>Input data:</b>
        IOptimizationTask object

        <b>Expected results:</b>
        Test passes if IOptimizationTask object optimize method raises NotImplementedError exception
        """
        dataset = DatasetEntity()
        configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="Test Header"),
            label_schema=LabelSchemaEntity(),
        )
        model_entity = ModelEntity(configuration=configuration, train_dataset=dataset)
        optimization_parameters = OptimizationParameters()
        with pytest.raises(NotImplementedError):
            IOptimizationTask().optimize(
                optimization_type=OptimizationType.POT,
                dataset=dataset,
                output_model=model_entity,
                optimization_parameters=optimization_parameters,
            )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestITrainingTask:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        "otx.api.usecases.tasks.interfaces.training_interface.ITrainingTask.__abstractmethods__",
        set(),
    )
    def test_training_interface(self):
        """
        <b>Description:</b>
        Check ITrainingTask class object initialization

        <b>Input data:</b>
        ITrainingTask object

        <b>Expected results:</b>
        Test passes if ITrainingTask object methods raise NotImplementedError exception
        """
        i_training_task = ITrainingTask()
        dataset = DatasetEntity()
        configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="Test Header"),
            label_schema=LabelSchemaEntity(),
        )
        model_entity = ModelEntity(configuration=configuration, train_dataset=dataset)
        train_parameters = TrainParameters()

        with pytest.raises(NotImplementedError):
            i_training_task.save_model(model_entity)
        with pytest.raises(NotImplementedError):
            i_training_task.train(
                dataset=dataset,
                output_model=model_entity,
                train_parameters=train_parameters,
            )
        with pytest.raises(NotImplementedError):
            i_training_task.cancel_training()


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIUnload:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        "otx.api.usecases.tasks.interfaces.unload_interface.IUnload.__abstractmethods__",
        set(),
    )
    def test_unload_interface(self):
        """
        <b>Description:</b>
        Check IUnload class object initialization

        <b>Input data:</b>
        IUnload object

        <b>Expected results:</b>
        Test passes if IUnload object unload method raises NotImplementedError exception
        """
        with pytest.raises(NotImplementedError):
            IUnload().unload()


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestExportType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_export_type(self):
        """
        <b>Description:</b>
        Check ExportType Enum class elements

        <b>Expected results:</b>
        Test passes if ExportType Enum class length is equal to expected value and its elements have expected
        sequence numbers
        """
        assert len(ExportType) == 1
        assert ExportType.OPENVINO.value == 1


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIExportTask:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    @patch(
        "otx.api.usecases.tasks.interfaces.export_interface.IExportTask.__abstractmethods__",
        set(),
    )
    def test_export_interface(self):
        """
        <b>Description:</b>
        Check IExportTask class object initialization

        <b>Input data:</b>
        IExportTask object

        <b>Expected results:</b>
        Test passes if IExportTask object export method raises NotImplementedError exception
        """
        dataset = DatasetEntity()
        configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(header="Test Header"),
            label_schema=LabelSchemaEntity(),
        )
        model_entity = ModelEntity(configuration=configuration, train_dataset=dataset)
        with pytest.raises(NotImplementedError):
            IExportTask().export(
                export_type=ExportType.OPENVINO,
                output_model=model_entity,
                precision=ModelPrecision.FP32,
                dump_features=False,
            )
