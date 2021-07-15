# Copyright (C) 2020-2021 Intel Corporation
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

from common.image_classification_test_case import create_image_classification_nncf_test_case # pylint: disable=import-error

kwargs = dict(
    problem_name='imagenet-classification',
    ann_file=' \' \' ',
    img_root=os.path.dirname(__file__) + '/../../../data/classification/'
)


class CustomClassificationNNCFTestCaseSmall(
        create_image_classification_nncf_test_case(
            model_name='mobilenet-v3-small',
            compression_cmd_line_parameters='--nncf-quantization',
            **kwargs,
            compression_cfg_update_dict={
                'nncf_quantization.train.batch_size': 2,
                'nncf_quantization.test.batch_size': 2,
                'nncf_quantization.train.max_epoch': 11,
                'nncf_quantization.changes_aux_config.train.batch_size': 2,
                'nncf_quantization.changes_aux_config.test.batch_size': 2,
                'nncf_quantization.train.seed': 260,
            },
            field_value_changes_in_nncf_config={
                'num_init_samples': 8,
                'num_bn_adaptation_samples': 8
            }
        )
):
    """ NNCF test case for mobilenet v3 small model. """
