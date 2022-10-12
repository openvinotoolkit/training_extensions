"""Dataset for MPA Training of Detection Task."""

# Copyright (C) 2022 Intel Corporation
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

from mmdet.datasets.builder import DATASETS
from mpa.utils.logger import get_logger

from otx.algorithms.common.utils.data_utils import get_old_new_img_indices

from .mmdataset import OTXDataset

logger = get_logger()


@DATASETS.register_module()
class MPADetDataset(OTXDataset):
    """MPADetDataset for Class-Incremental Learning."""

    def __init__(self, **kwargs):
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop("org_type", None)
        new_classes = dataset_cfg.pop("new_classes", [])
        super().__init__(**dataset_cfg)

        test_mode = kwargs.get("test_mode", False)
        if test_mode is False:
            self.img_indices = get_old_new_img_indices(self.labels, new_classes, self.otx_dataset)

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.get_ann_info(idx)["labels"].astype(int).tolist()
