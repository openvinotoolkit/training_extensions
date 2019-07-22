import argparse
import collections
import time

import cv2

from common import *
from frames_provider import FramesProvider, CvatFramesProvider, VideoFramesProvider
from image_retrieval import ImageRetrieval
import logging as log


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model_weights', required=True, help='Path to model weights.')
    args.add_argument('--gallery', required=True, help='Gallery images folder.')
    args.add_argument('--test_data_path', required=True, help='Test images folder.')
    args.add_argument('--test_data_type', choices=['crops', 'cvat_annotation', 'videos'],
                      required=True)
    args.add_argument('--test_annotation_path')
    args.add_argument('--input_size', default=224, type=int, help='Input image size.')
    args.add_argument('--model', choices=['resnet50', 'mobilenet_v2'], default='mobilenet_v2')
    args.add_argument('--imshow_delay', type=int, default=-1)
    args.add_argument('--ie', choices=['tf', 'ie'], required=True)

    return args.parse_args()


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

    cv2.putText(result, 'Inp. res: {}x{}'.format(input_size, input_size),
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


def test(model_path, model_backend, model, gallery_path, test_data_path, test_data_type,
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

    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # out = cv2.VideoWriter('output.avi', fourcc, 15.0, (980, 920))

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

            result, key = visualize(view_frame, position,
                                    [img_retrieval.impaths[i] for i in sorted_indexes],
                                    sorted_distances,
                                    input_size,
                                    mean_compute_embeddings_times, mean_search_in_gallery_times,
                                    imshow_delay)
            #           out.write(result)
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


if __name__ == '__main__':
    LOG_FORMAT = '%(levelno)s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s|%(message)s'
    log.basicConfig(format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    log.getLogger().setLevel(log.WARN)

    args = parse_args()

    test(model_path=args.model_weights,
         model_backend=args.ie,
         model=None,
         gallery_path=args.gallery,
         test_data_path=args.test_data_path,
         test_data_type=args.test_data_type,
         test_annotation_path=args.test_annotation_path,
         input_size=args.input_size,
         imshow_delay=args.imshow_delay)
