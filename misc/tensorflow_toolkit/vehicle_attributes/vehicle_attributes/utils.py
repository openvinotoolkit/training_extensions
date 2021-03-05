# Copyright (C) 2019 Intel Corporation
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

def get_checkpoint_variable_names(ckpt):
  from tensorflow.python import pywrap_tensorflow
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  vars_dict = reader.get_variable_to_shape_map()
  return [v for v in vars_dict.keys()]
