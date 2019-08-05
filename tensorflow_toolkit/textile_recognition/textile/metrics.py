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


import time
import collections
import numpy as np
import cv2
from textile.common import central_crop
from textile.frames_provider import FramesProvider, CvatFramesProvider, VideoFramesProvider
from textile.image_retrieval import ImageRetrieval

# pylint: disable=R0913
def visualize(image, target_pos, impaths, distances, input_size, compute_embedding_time,
              search_in_gallery_time, imshow_delay):
    size = 200

    input_image = image
    input_image = cv2.resize(input_image, (size * 4, size * 3))

    border = 30
    result = np.ones(
        (input_image.shape[0] + size * 2 + border * 4, size * 5 + border * 6, 3),
        dtype=np.uint8) * 200

    result[border:border + input_image.shape[0], border:border + input_image.shape[1]] = input_image

    text_size = 1.5

    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    LWD = 2

    if target_pos is not None and target_pos >= 0:
        target = cv2.imread(impaths[target_pos])
        target = cv2.resize(target, (size, size))

        target_loc = border + input_image.shape[0] - size, border * 2 + input_image.shape[1]
        result[target_loc[0]:target_loc[0] + size, target_loc[1]:target_loc[1] + size] = target

        cv2.putText(result, 'Pattern:', (target_loc[1], target_loc[0] - 10), 1, text_size,
                    BLACK, LWD)
        cv2.putText(result, 'Position: {}'.format(target_pos),
                    (border * 2 + input_image.shape[1], border * 2), 1, text_size, BLACK, LWD)

    cv2.putText(result, 'Gallery size: {}'.format(len(impaths)),
                (border * 2 + input_image.shape[1], border * 2 + 30), 1, text_size, BLACK, LWD)
    cv2.putText(result, 'Embbeding (ms): {}'.format(int(compute_embedding_time * 10000) / 10.0),
                (border * 2 + input_image.shape[1], border * 2 + 60), 1, text_size, BLACK, LWD)
    cv2.putText(result,
                'Gallery search (ms): {}'.format(int(search_in_gallery_time * 10000) / 10.0),
                (border * 2 + input_image.shape[1], border * 2 + 90), 1, text_size, BLACK, LWD)

    cv2.putText(result, 'Inp. res: {0}x{0}'.format(input_size),
                (border * 2 + input_image.shape[1], border * 2 + 120), 1, text_size, BLACK, LWD)

    for index, impath in enumerate(impaths[:10]):
        image = cv2.imread(impath)
        image = cv2.resize(image, (size, size))

        h_shift = 2 * border + input_image.shape[0] + (size + border) * (index // 5)
        w_shift = border + (index % 5) * (size + border)

        result[h_shift: h_shift + size, w_shift: w_shift + size] = image

        cv2.putText(result, '{}:{}'.format(index, int(distances[index] * 100) / 100),
                    (w_shift - border, h_shift - 5), 1,
                    text_size, BLACK, LWD)

        if target_pos == index:
            cv2.rectangle(result, (w_shift, h_shift), (w_shift + size, h_shift + size),
                          GREEN, border // 3)

    cv2.imshow('result', result)

    key = cv2.waitKey(imshow_delay) & 0xff

    return result, key

#pylint: disable=R0912,R0913,R0914,R0915
def test_model(model_path, model_backend, model, gallery_path, test_data_path, test_data_type,
               test_annotation_path, input_size, imshow_delay=-1):
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

    compute_embeddings_times = []
    search_in_gallery_times = []

    for image, probe_class, view_frame in frames.frames_gen():
        if image is not None:
            image = central_crop(image, divide_by=5, shift=1)

            t = time.perf_counter()
            probe_embedding = img_retrieval.compute_embedding(image)
            elapsed = time.perf_counter() - t
            compute_embeddings_times.append(elapsed)

            t = time.perf_counter()
            sorted_indexes, distances = img_retrieval.search_in_gallery(probe_embedding)
            elapsed = time.perf_counter() - t
            search_in_gallery_times.append(elapsed)

            sorted_classes = [img_retrieval.gallery_classes[i] for i in sorted_indexes]
            position = sorted_classes.index(img_retrieval.text_label_to_class_id[probe_class])
            results[probe_class].append(position)
        else:
            sorted_indexes = []
            position = None
        mean_compute_embeddings_times = np.mean(
            compute_embeddings_times) if compute_embeddings_times else -1
        mean_search_in_gallery_times = np.mean(
            search_in_gallery_times) if search_in_gallery_times else -1

        if imshow_delay >= 0:
            sorted_distances = distances[sorted_indexes] if position is not None else None

            _, key = visualize(view_frame, position,
                               [img_retrieval.impaths[i] for i in sorted_indexes],
                               sorted_distances,
                               input_size,
                               mean_compute_embeddings_times, mean_search_in_gallery_times,
                               imshow_delay)

            if key == 27:
                exit(0)
            elif key == ord('n'):
                frames.go_to_next_video()

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
