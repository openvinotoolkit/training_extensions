# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from otx.api.configuration import ConfigurableParameters, cfg_helper
from otx.api.entities.annotation import NullAnnotationSceneEntity
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.metrics import NullPerformance, Performance, ScoreMetric
from otx.api.entities.model import (
    ModelConfiguration,
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.model_template import TargetDevice, parse_model_template
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.api.utils.time_utils import now
from tests.test_helpers import generate_random_single_image
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelPrecision:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_precision(self):
        """
        <b>Description:</b>
        Check that ModelPrecision correctly returns the precision name

        <b>Expected results:</b>
        Test passes if ModelPrecision correctly returns the precision name

        <b>Steps</b>
        1. Check precisions in the ModelPrecision
        """

        model_precision = ModelPrecision
        assert len(model_precision) == 4


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelFormat:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_format(self):
        """
        <b>Description:</b>
        Check that ModelFormat correctly returns the format name

        <b>Expected results:</b>
        Test passes if ModelFormat correctly returns the format name

        <b>Steps</b>
        1. Check formats in the ModelFormat
        """

        model_format = ModelFormat
        assert len(model_format) == 3


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelOptimizationType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_optimization_type(self):
        """
        <b>Description:</b>
        Check that ModelOptimizationType correctly returns the optimization type name

        <b>Expected results:</b>
        Test passes if ModelOptimizationType correctly returns the optimization type name

        <b>Steps</b>
        1. Check optimization types in the ModelOptimizationType
        """

        model_optimization_type = ModelOptimizationType
        assert len(model_optimization_type) == 5


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestOptimizationMethod:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_optimization_method(self):
        """
        <b>Description:</b>
        Check that OptimizationMethod correctly returns the optimization method name

        <b>Expected results:</b>
        Test passes if OptimizationMethod correctly returns the optimization method name

        <b>Steps</b>
        1. Check optimization methods in the OptimizationMethod
        """

        optimization_method = OptimizationMethod
        assert len(optimization_method) == 2


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelConfiguration:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_configuration(self):
        """
        <b>Description:</b>
        Check that ModelConfiguration correctly returns the configuration

        <b>Input data:</b>
        ConfigurableParameters, LabelSchemaEntity

        <b>Expected results:</b>
        Test passes if ModelConfiguration correctly returns the configuration

        <b>Steps</b>
        1. Check configuration params in the ModelConfiguration
        """
        parameters = ConfigurableParameters(header="Test header")
        label_schema = LabelSchemaEntity()
        model_configuration = ModelConfiguration(configurable_parameters=parameters, label_schema=label_schema)
        assert model_configuration.configurable_parameters == parameters
        assert model_configuration.get_label_schema() == label_schema


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelEntity:
    creation_date = now()

    def generate_random_image(self):
        with generate_random_single_image() as path:
            image = Image(file_path=path)
            return DatasetItemEntity(media=image, annotation_scene=NullAnnotationSceneEntity())

    def dataset(self):
        return DatasetEntity(items=[self.generate_random_image()])

    def configuration(self):
        parameters = ConfigurableParameters(header="Test header")
        label_schema = LabelSchemaEntity()
        return ModelConfiguration(configurable_parameters=parameters, label_schema=label_schema)

    def other_configuration(self):
        parameters = ConfigurableParameters(header="Other test header")
        label_schema = LabelSchemaEntity()
        return ModelConfiguration(configurable_parameters=parameters, label_schema=label_schema)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_entity_default_values(self):
        """
        <b>Description:</b>
        Check that ModelEntity correctly returns the default values

        <b>Expected results:</b>
        Test passes if ModelEntity correctly returns the default values

        <b>Steps</b>
        1. Check default values in the ModelEntity
        """

        model_entity = ModelEntity(train_dataset=self.dataset(), configuration=self.configuration())

        assert model_entity.id_ == ID()
        assert type(model_entity.configuration) == ModelConfiguration
        assert type(model_entity.creation_date) == datetime
        assert type(model_entity.train_dataset) == DatasetEntity
        assert model_entity.version == 1
        assert model_entity.model_format == ModelFormat.OPENVINO
        assert model_entity.precision == [ModelPrecision.FP32]
        assert model_entity.target_device == TargetDevice.CPU
        assert model_entity.optimization_type == ModelOptimizationType.NONE
        assert model_entity.performance == NullPerformance()

        for default_val_none in [
            "previous_trained_revision",
            "previous_revision",
            "target_device_type",
        ]:
            assert getattr(model_entity, default_val_none) is None

        for default_val_0_0 in ["training_duration", "model_size_reduction"]:
            assert getattr(model_entity, default_val_0_0) == 0.0

        for default_val_empty_list in ["tags", "optimization_methods"]:
            assert getattr(model_entity, default_val_empty_list) == []

        for default_val_empty_dict in [
            "model_adapters",
            "optimization_objectives",
            "performance_improvement",
        ]:
            assert getattr(model_entity, default_val_empty_dict) == {}

        for default_val_zero in ["latency", "fps_throughput"]:
            assert getattr(model_entity, default_val_zero) == 0

        assert model_entity.is_optimized() is False

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_entity_sets_values(self):
        """
        <b>Description:</b>
        Check that ModelEntity correctly returns the set values

        <b>Expected results:</b>
        Test passes if ModelEntity correctly returns the set values

        <b>Steps</b>
        1. Check set values in the ModelEntity
        """

        def __get_path_to_file(filename: str):
            """
            Return the path to the file named 'filename', which lives in the tests/entities directory
            """
            return str(Path(__file__).parent / Path(filename))

        car = LabelEntity(name="car", domain=Domain.DETECTION)
        labels_list = [car]
        dummy_template = __get_path_to_file("./dummy_template.yaml")
        model_template = parse_model_template(dummy_template)
        hyper_parameters = model_template.hyper_parameters.data
        params = cfg_helper.create(hyper_parameters)
        labels_schema = LabelSchemaEntity.from_labels(labels_list)
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )

        item = self.generate_random_image()
        dataset = DatasetEntity(items=[item])
        score_metric = ScoreMetric(name="Model accuracy", value=0.5)

        model_entity = ModelEntity(train_dataset=self.dataset(), configuration=self.configuration())

        set_params = {
            "configuration": environment.get_model_configuration(),
            "train_dataset": dataset,
            "id": ID(1234567890),
            "creation_date": self.creation_date,
            "previous_trained_revision": 5,
            "previous_revision": 2,
            "version": 2,
            "tags": ["tree", "person"],
            "model_format": ModelFormat.BASE_FRAMEWORK,
            "performance": Performance(score_metric),
            "training_duration": 5.8,
            "precision": [ModelPrecision.INT8],
            "latency": 328,
            "fps_throughput": 20,
            "target_device": TargetDevice.GPU,
            "target_device_type": "notebook",
            "optimization_methods": [OptimizationMethod.QUANTIZATION],
            "optimization_type": ModelOptimizationType.MO,
            "optimization_objectives": {"param": "Test param"},
            "performance_improvement": {"speed", 0.5},
            "model_size_reduction": 1.0,
        }

        for key, value in set_params.items():
            setattr(model_entity, key, value)
            assert getattr(model_entity, key) == value

        assert model_entity.is_optimized() is True

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_entity_model_adapters(self):
        """
        <b>Description:</b>
        Check that ModelEntity correctly returns the adapters

        <b>Expected results:</b>
        Test passes if ModelEntity correctly returns the adapters

        <b>Steps</b>
        1. Create a ModelEntity with adapters
        2. Change data source for an adapter
        3. Remove an adapter
        """

        data_source_0 = b"{0: binaryrepo://localhost/repo/data_source/0}"
        data_source_1 = b"binaryrepo://localhost/repo/data_source/1"
        data_source_2 = b"binaryrepo://localhost/repo/data_source/2"
        data_source_3 = b"binaryrepo://localhost/repo/data_source/3"

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "data_source_0")

        with open(temp_file, "wb") as tmp:
            tmp.write(data_source_0)

        model_adapters = {
            "0": ModelAdapter(data_source=data_source_0),
            "1": ModelAdapter(data_source=data_source_1),
            "2": ModelAdapter(data_source=data_source_2),
        }

        model_entity = ModelEntity(
            train_dataset=self.dataset(),
            configuration=self.configuration(),
            model_adapters=model_adapters,
        )

        # Adapter with key 0 not from file
        assert model_entity.model_adapters["0"].from_file_storage is False

        model_entity.set_data("0", temp_file)

        for adapter in model_entity.model_adapters:
            if adapter == "0":
                # Adapter with key 0 from file
                assert model_entity.model_adapters[adapter].from_file_storage is True
            else:
                assert model_entity.model_adapters[adapter].from_file_storage is False

        assert model_entity.get_data("1") == data_source_1

        model_entity.set_data("2", data_source_1)
        assert model_entity.get_data("2") == data_source_1
        assert len(model_entity.model_adapters) == 3

        model_entity.set_data("3", data_source_3)
        assert model_entity.get_data("3") == data_source_3
        assert len(model_entity.model_adapters) == 4

        model_entity.delete_data("3")
        assert len(model_entity.model_adapters) == 3

        # Attempt to retrieve a missing and deleted key
        with pytest.raises(KeyError):
            model_entity.get_data("5")

        with pytest.raises(KeyError):
            model_entity.get_data("3")

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_entity__eq__(self):
        """
        <b>Description:</b>
        Check that ModelEntity __eq__ method

        <b>Expected results:</b>
        Test passes if ModelEntity equal ModelEntity and not equal another type
        """
        dataset = self.dataset()
        other_model_entity = ModelEntity(train_dataset=dataset, configuration=self.configuration())
        model_entity = ModelEntity(train_dataset=dataset, configuration=self.configuration())
        third_model_entity = ModelEntity(train_dataset=self.dataset(), configuration=self.other_configuration())
        assert model_entity.__eq__("") is False
        assert model_entity == other_model_entity
        assert model_entity != third_model_entity
