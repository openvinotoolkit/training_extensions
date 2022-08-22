# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import pytest

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelGroup, LabelSchemaEntity
from otx.api.entities.model import ModelConfiguration, ModelEntity
from otx.api.usecases.reporting.callback import Callback
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestCallback:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_callback_attributes(self):
        """
        <b>Description:</b>
        Check "Callback" class object initialization

        <b>Input data:</b>
        "Callback" class object

        <b>Expected results:</b>
        Test passes if "params" and "model" attributes of initialized "Callback" class object are equal to expected

        <b>Steps</b>
        1. Check "model" attribute of "Callback" object after "set_model" method
        2. Check "params" attribute of "Callback" object after "set_params" method
        """
        callback = Callback()
        # Checking "params" of "Callback" object after "set_params"
        params = {"parameter_1": 1, "parameter_2": 4, "parameter_3": 9}
        callback.set_params(params)
        assert callback.params == params
        # Checking "model" of "Callback" after "set_model"
        configurable_params = ConfigurableParameters(header="Test model configurable params")
        labels_group = LabelGroup(name="model_group", labels=[])
        model_configuration = ModelConfiguration(configurable_params, LabelSchemaEntity(label_groups=[labels_group]))
        model = ModelEntity(train_dataset=DatasetEntity(), configuration=model_configuration)
        callback.set_model(model)
        assert callback.model == model

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_callback_abstract_methods(self):
        """
        <b>Description:</b>
        Check "Callback" class object abstract methods

        <b>Input data:</b>
        "Callback" class object

        <b>Expected results:</b>
        Test passes if none of "Callback" class abstract methods raised an exception
        """
        callback = Callback()
        callback.on_epoch_begin(epoch=1, logs="on_epoch_begin logs")
        callback.on_epoch_end(epoch=1, logs="on_epoch_end logs")
        callback.on_batch_begin(batch=1, logs="on_batch_begin logs")
        callback.on_batch_end(batch=1, logs="on_batch_end logs")
        callback.on_train_begin(logs="on_train_begin logs")
        callback.on_train_end(logs="on_train_end logs")
        callback.on_train_batch_begin(batch=1, logs="on_train_batch_begin logs")
        callback.on_train_batch_end(batch=1, logs="on_train_batch_end logs")
        callback.on_test_begin("on_test_begin logs")
        callback.on_test_end("on_epoch_begin logs")
        callback.on_test_batch_begin(batch=1, logs="on_test_batch_begin logs")
        callback.on_test_batch_end(batch=1, logs="on_epoch_begin logs")
