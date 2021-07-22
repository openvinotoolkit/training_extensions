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


def apply_template_configurable_parameters(params, template: dict):

    def xset(obj, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                xset(getattr(obj, k), v)
            else:
                # if hasattr(getattr(obj, k), 'value'):
                #     getattr(obj, k).value = type(getattr(obj, k).value)(v)
                # else:
                setattr(obj, k, v)

    hyper_params = template['hyper_parameters']['params']
    xset(params, hyper_params)
    params.algo_backend.model_name = template['name']
