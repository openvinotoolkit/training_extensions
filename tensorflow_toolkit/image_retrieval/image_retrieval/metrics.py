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


import collections
import numpy as np
from image_retrieval.common import central_crop
from image_retrieval.frames_provider import FramesProvider, CvatFramesProvider, VideoFramesProvider
from image_retrieval.image_retrieval import ImageRetrieval


def test_model(model_path, model_backend, model, gallery_path, test_data_path, test_data_type,
               test_annotation_path, input_size):
    img_retrieval = ImageRetrieval(model_path, model_backend, model, gallery_path, input_size)

    if test_data_type == 'crops':
        frames = FramesProvider(test_data_path)
    elif test_data_type == 'cvat_annotation':
        assert test_annotation_path
        frames = CvatFramesProvider(test_annotation_path, test_data_path)
    elif test_data_type == 'videos':
        assert test_annotation_path
        frames = VideoFramesProvider(test_annotation_path, test_data_path)

    top1_counters = []
    top5_counters = []
    top10_counters = []
    mean_positions = []

    results = collections.defaultdict(list)

    for image, probe_class, view_frame in frames.frames_gen():
        if image is not None:
            image = central_crop(image, divide_by=5, shift=1)

            probe_embedding = img_retrieval.compute_embedding(image)

            sorted_indexes, distances = img_retrieval.search_in_gallery(probe_embedding)

            sorted_classes = [img_retrieval.gallery_classes[i] for i in sorted_indexes]
            position = sorted_classes.index(img_retrieval.text_label_to_class_id[probe_class])
            results[probe_class].append(position)

    for probe_class in sorted(results.keys()):
        top1, top5, top10 = 0, 0, 0
        for p in results[probe_class]:
            if p < 1:
                top1 += 1
            if p < 5:
                top5 += 1
            if p < 10:
                top10 += 1
        top1 /= len(results[probe_class])
        top5 /= len(results[probe_class])
        top10 /= len(results[probe_class])
        mean_position = np.mean(results[probe_class])

        print('{0}\t{1:4.2f}\t{2:4.2f}\t{3:4.2f}\t{4:4.2f}'.format(probe_class, top1, top5, top10,
                                                                   mean_position))

        top1_counters.append(top1)
        top5_counters.append(top5)
        top10_counters.append(top10)

        mean_positions.append(mean_position)

    print(
        'AVERAGE: top1: {0:4.3f}    top5: {1:4.3f}    top10: {2:4.3f}    mean_index: {3:4.3f}'.format(
            np.mean(top1_counters),
            np.mean(top5_counters),
            np.mean(top10_counters),
            np.mean(mean_positions)
        ))

    return np.mean(top1_counters), np.mean(top5_counters), np.mean(top10_counters), np.mean(
        mean_positions)
