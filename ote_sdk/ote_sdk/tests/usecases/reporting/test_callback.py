# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.


import pytest

from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label_schema import LabelGroup, LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.usecases.reporting.callback import Callback


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestCallback:
    @pytest.mark.priority_medium
    @pytest.mark.component
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
        configurable_params = ConfigurableParameters(
            header="Test model configurable params"
        )
        labels_group = LabelGroup(name="model_group", labels=[])
        model_configuration = ModelConfiguration(
            configurable_params, LabelSchemaEntity(label_groups=[labels_group])
        )
        model = ModelEntity(
            train_dataset=DatasetEntity(), configuration=model_configuration
        )
        callback.set_model(model)
        assert callback.model == model

    @pytest.mark.priority_medium
    @pytest.mark.component
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
