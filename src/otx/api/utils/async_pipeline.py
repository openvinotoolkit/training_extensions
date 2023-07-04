"""OTX AsyncPipeline of OTX Detection."""

# Copyright (C) 2023 Intel Corporation
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
import copy
from time import perf_counter

from openvino.model_api.pipelines import AsyncPipeline


class OTXDetectionAsyncPipeline(AsyncPipeline):
    """OTX AsyncPipeline of OTX Detection."""

    def get_result(self, id):  # pylint: disable=redefined-builtin
        """Get result of inference by index.

        Args:
            id (int): index of inference

        Returns:
            result (tuple): tuple of inference result and meta information
        """
        result = self.get_raw_result(id)
        if result:
            raw_result, meta, preprocess_meta, infer_start_time = result
            self.inference_metrics.update(infer_start_time)

            postprocessing_start_time = perf_counter()
            result = self.model.postprocess(raw_result, preprocess_meta), {**meta, **preprocess_meta}
            self.postprocess_metrics.update(postprocessing_start_time)
            features = (None, None)
            if "feature_vector" in raw_result or "saliency_map" in raw_result:
                features = (
                    copy.deepcopy(raw_result["feature_vector"].reshape(-1)),
                    copy.deepcopy(raw_result["saliency_map"][0]),
                )
            return *result, features
        return None
