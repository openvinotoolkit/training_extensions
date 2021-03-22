"""
 Copyright (c) 2020 Intel Corporation

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

# pylint: disable=too-many-statements

import re
import argparse
from os.path import exists, join, abspath, isfile, dirname, basename
from os import listdir, walk, makedirs
from shutil import rmtree

import cv2
import numpy as np
import mmcv
from tqdm import tqdm

from openvino.inference_engine import IECore # pylint: disable=no-name-in-module


DETECTOR_OUTPUT_SHAPE = -1, 5
MIN_DET_CONF = 0.1
MIN_AREA_SIZE = 0.15


def load_ie_core(device, cpu_extension=None):
    ie = IECore()
    if device == "CPU" and cpu_extension:
        ie.add_extension(cpu_extension, "CPU")

    return ie


class IEModel:
    def __init__(self, model_path, device, ie_core, num_requests, output_shape=None):
        if model_path.endswith((".xml", ".bin")):
            model_path = model_path[:-4]
        self.net = ie_core.read_network(model_path + ".xml", model_path + ".bin")
        assert len(self.net.inputs.keys()) == 1, "One input is expected"

        supported_layers = ie_core.query_network(self.net, device)
        not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) > 0:
            raise RuntimeError("Following layers are not supported by the {} plugin:\n {}"
                               .format(device, ', '.join(not_supported_layers)))

        self.exec_net = ie_core.create_network(network=self.net,
                                               device_name=device,
                                               num_requests=num_requests)

        self.input_name = next(iter(self.net.inputs))
        if len(self.net.outputs) > 1:
            if output_shape is not None:
                candidates = []
                for candidate_name in self.net.outputs:
                    candidate_shape = self.exec_net.requests[0].outputs[candidate_name].shape
                    if len(candidate_shape) != len(output_shape):
                        continue

                    matches = [src == trg or trg < 0
                               for src, trg in zip(candidate_shape, output_shape)]
                    if all(matches):
                        candidates.append(candidate_name)

                if len(candidates) != 1:
                    raise Exception("One output is expected")

                self.output_name = candidates[0]
            else:
                raise Exception("One output is expected")
        else:
            self.output_name = next(iter(self.net.outputs))

        self.input_size = self.net.inputs[self.input_name].shape
        self.output_size = self.exec_net.requests[0].outputs[self.output_name].shape
        self.num_requests = num_requests

    def infer(self, data):
        input_data = {self.input_name: data}
        infer_result = self.exec_net.infer(input_data)
        return infer_result[self.output_name]


class PersonDetector(IEModel):
    def __init__(self, model_path, device, ie_core, num_requests, output_shape=None):
        super().__init__(model_path, device, ie_core, num_requests, output_shape)

        _, _, h, w = self.input_size
        self.input_height = h
        self.input_width = w

        self.last_scales = None
        self.last_sizes = None

    def _prepare_frame(self, frame):
        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.input_height), initial_w / float(self.input_width)

        in_frame = cv2.resize(frame, (self.input_width, self.input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)

        return in_frame, initial_h, initial_w, scale_h, scale_w

    def _process_output(self, result, initial_h, initial_w, scale_h, scale_w, ):
        if result.shape[-1] == 5:  # format: [xmin, ymin, xmax, ymax, conf]
            return np.array([[scale_w, scale_h, scale_w, scale_h, 1.0]]) * result

        # format: [image_id, label, conf, xmin, ymin, xmax, ymax]
        scale_w *= self.input_width
        scale_h *= self.input_height
        out = np.array([[1.0, scale_w, scale_h, scale_w, scale_h]]) * result[0, 0, :, 2:]

        return np.concatenate((out[:, 1:], out[:, 0].reshape([-1, 1])), axis=1)

    def __call__(self, frame):
        in_frame, initial_h, initial_w, scale_h, scale_w = self._prepare_frame(frame)
        result = self.infer(in_frame)
        out = self._process_output(result, initial_h, initial_w, scale_h, scale_w)

        return out


def create_dirs(dir_path, override=False):
    if override:
        if exists(dir_path):
            rmtree(dir_path)
        makedirs(dir_path)
    elif not exists(dir_path):
        makedirs(dir_path)


def parse_relative_paths(data_dir, dir_pattern=''):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    pattern = None
    if dir_pattern is not None and dir_pattern != '':
        pattern = re.compile(dir_pattern)

    relative_paths = []
    for root, sub_dirs, files in tqdm(walk(data_dir)):
        if len(sub_dirs) == 0 and len(files) > 0:
            valid = True
            if pattern is not None:
                valid = pattern.search(basename(root)) is not None

            if valid:
                relative_paths.append(root[skip_size:])

    return relative_paths


def prepare_tasks(relative_paths, input_dir, output_dir):
    out_tasks = []
    for relative_path in tqdm(relative_paths):
        images = join(input_dir, relative_path)
        assert exists(images)

        input_files = [f for f in listdir(images) if isfile(join(images, f))]
        input_num_frames = len(input_files)

        output_path = join(output_dir, relative_path)
        if exists(output_path):
            existed_files = [f for f in listdir(output_path) if isfile(join(output_path, f))]
            existed_frame_ids = [int(f.split('.')[0]) for f in existed_files]
            existed_num_frames = len(existed_frame_ids)
            if min(existed_frame_ids) != 1 or \
               existed_num_frames != max(existed_frame_ids) or \
               existed_num_frames != input_num_frames:
                rmtree(output_path)
            else:
                continue

        clip_tasks = []
        for image_file in input_files:
            full_input_path = join(images, image_file)
            full_output_path = join(output_path, '{}.jpg'.format(image_file.split('.')[0]))
            clip_tasks.append((full_input_path, full_output_path))

        if len(clip_tasks) == 0:
            continue

        out_tasks.append((clip_tasks, output_path))

    return out_tasks


def parse_detection_output(result, class_ids, threshold=0.4):
    bbox_result, _ = result

    out_data = list()
    for class_id in class_ids:
        bboxes = bbox_result[class_id]
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            if bbox[-1] > threshold:
                out_data.append(bbox[:-1])

    return out_data


def get_main_detection(objects, img_size, min_det_conf, min_area):
    img_height, img_width = img_size[:2]

    areas = []
    for bbox in objects:
        conf = bbox[-1]
        if conf < min_det_conf:
            continue

        bbox_width = max(0, bbox[2] - bbox[0])
        bbox_height = max(0, bbox[3] - bbox[1])
        areas.append(float(bbox_width * bbox_height) / float(img_height * img_width))

    areas = np.array(areas, dtype=np.float32)
    if len(areas) == 0:
        return None

    best_match = np.argmax(areas)
    best_area = areas[best_match]
    if best_area < min_area:
        return None

    return objects[best_match][:4]


def get_glob_bbox(bboxes):
    bboxes = np.array(bboxes)
    x_min = np.percentile(bboxes[:, 0], 5.0)
    y_min = np.percentile(bboxes[:, 1], 5.0)
    x_max = np.percentile(bboxes[:, 2], 95.0)
    y_max = np.percentile(bboxes[:, 3], 95.0)

    return x_min, y_min, x_max, y_max


def crop_image(image, bbox, trg_size, scale):
    def _fix_bbox():
        frame_height, frame_width = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox

        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)

        scaled_width = scale * (x_max - x_min)
        scaled_height = scale * (y_max - y_min)

        src_ar = float(scaled_height) / float(scaled_width)
        trg_ar = float(trg_size[0]) / float(trg_size[1])

        if src_ar < trg_ar:
            out_width = scaled_width
            out_height = trg_ar * out_width
        else:
            out_height = scaled_height
            out_width = out_height / trg_ar

        out_x_min = np.maximum(0, int(center_x - 0.5 * out_width))
        out_y_min = np.maximum(0, int(center_y - 0.5 * out_height))
        out_x_max = np.minimum(frame_width, int(center_x + 0.5 * out_width))
        out_y_max = np.minimum(frame_height, int(center_y + 0.5 * out_height))

        return out_x_min, out_y_min, out_x_max, out_y_max

    roi = _fix_bbox()

    cropped_image = image[roi[1]:roi[3], roi[0]:roi[2]]
    resized_image = cv2.resize(cropped_image, (trg_size[1], trg_size[0]))

    return resized_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help='checkpoint file')
    parser.add_argument('--input-dir', '-i', type=str, required=True, help='input dir')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='output dir')
    parser.add_argument('--dir_pattern', type=str, required=False, default='')
    parser.add_argument('--trg_image_size', type=str, required=False, default='300,256')
    parser.add_argument('--scale', type=float, required=False, default=1.2)
    parser.add_argument('--device', '-d', type=str, default='CPU')
    parser.add_argument('--cpu_extension', '-l', type=str, default=None)
    parser.add_argument('--clear_dumped', '-c', action='store_true', required=False)
    args = parser.parse_args()

    assert exists(args.input_dir)

    override = args.clear_dumped
    create_dirs(args.output_dir, override=override)

    print('\nLoading model ...')
    ie_core = load_ie_core(args.device, args.cpu_extension)
    person_detector = PersonDetector(args.model, args.device, ie_core,
                                     num_requests=1, output_shape=DETECTOR_OUTPUT_SHAPE)
    print('Finished.')

    print('\nPreparing tasks ...')
    relative_paths = parse_relative_paths(args.input_dir, args.dir_pattern)
    tasks = prepare_tasks(relative_paths, args.input_dir, args.output_dir)
    print('Finished. Prepared {} / {} tasks.'.format(len(tasks), len(relative_paths)))

    trg_size = [int(v) for v in args.trg_image_size.split(',')]

    print('\nCropping frames ...')
    invalid_sources = []
    for clip_level_tasks, clip_out_path in tqdm(tasks):
        clip_bboxes = []
        frames = []
        for in_image_path, out_image_path in clip_level_tasks:
            img = mmcv.imread(in_image_path)
            if img is None:
                continue

            img_height, img_width = img.shape[:2]
            if img_height < 5 or img_width < 5:
                continue

            frames.append((np.copy(img), out_image_path))

            det_output = person_detector(img)
            obj_bbox = get_main_detection(det_output, img.shape,
                                          min_det_conf=MIN_DET_CONF,
                                          min_area=MIN_AREA_SIZE)
            if obj_bbox is None:
                continue

            clip_bboxes.append(obj_bbox)

        if len(clip_bboxes) == 0:
            invalid_source = dirname(clip_level_tasks[0][0])
            invalid_sources.append((invalid_source, 'no detections'))
            print('[WARNING] Skipped video: {}'.format(invalid_source))
            continue

        if len(frames) != len(clip_level_tasks):
            invalid_source = dirname(clip_level_tasks[0][0])
            num_skipped_frames = len(clip_level_tasks) - len(frames)
            message = 'skipped {} / {} frames'.format(num_skipped_frames, len(clip_level_tasks))
            invalid_sources.append((invalid_source, message))
            print('[WARNING] Invalid source: {}'.format(invalid_source))
            continue

        glob_bbox = get_glob_bbox(clip_bboxes)

        makedirs(clip_out_path)
        for img, out_image_path in frames:
            cropped_img = crop_image(img, glob_bbox, trg_size, args.scale)
            cv2.imwrite(out_image_path, cropped_img)
    print('Finished.')

    if len(invalid_sources) > 0:
        print('\nInvalid sources:')
        for invalid_source, message in invalid_sources:
            print('   - {}: {}'.format(invalid_source, message))
        print('Total invalid sources: {}'.format(len(invalid_sources)))


if __name__ == '__main__':
    main()
