"""
Utils for dynamically importing stuff
"""

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

import io
import pickle

from ote_sdk.serialization.label_mapper import LabelSchemaMapper


def load_model_weights(path):
    """
    Loads binary weights of a model.

        Args:
            path: A path where to load model from.
    """

    with open(path, "rb") as read_file:
        return read_file.read()


def read_label_schema(model_bytes):
    """
    Reads serialized representation from binary snapshot and returns deserialized LabelSchema.
    """

    return LabelSchemaMapper().backward(
        pickle.load(io.BytesIO(model_bytes))["label_schema"]
    )
