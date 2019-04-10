"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
import pycocotools.mask as mask_util


def polys_to_mask_wrt_box(polygons, box, resolution):
    if not isinstance(resolution, (list, tuple)):
        resolution = (resolution, resolution)

    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    for poly in polygons:
        p = np.array(poly, dtype=np.float32).flatten()
        p[0::2] = (p[0::2] - box[0]) * resolution[0] / w
        p[1::2] = (p[1::2] - box[1]) * resolution[1] / h
        polygons_norm.append(p)

    if len(polygons_norm) > 0:
        rle = mask_util.frPyObjects(polygons_norm, resolution[0], resolution[1])
        mask = np.array(mask_util.decode(rle), dtype=np.float32)
        # Flatten in case polygons was a list
        mask = np.sum(mask, axis=2)
        mask = np.array(mask > 0, dtype=np.float32)
    else:
        mask = np.full(resolution, -1, dtype=np.float32)
    return mask
