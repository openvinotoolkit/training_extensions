#!/usr/bin/env python2
#
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

from __future__ import print_function

from os import makedirs, listdir
from os.path import exists, join, basename, isfile
from json import dump as json_dump
from shutil import rmtree
from argparse import ArgumentParser

import yaml
import numpy as np
from tqdm import tqdm
from lxml import etree


VALID_EXTENSIONS = ['png', 'jpg']
IDS_SHIFT_SCALE = 1000000


def _parse_tasks(file_path):
    """Parse input text file where each row is [path_to_annotation img_h,img_w path_to_images_dir]
       into list of task.

    :param file_path: Path to file
    :return: List of tasks
    """

    tasks = []
    with open(file_path, 'r') as file_stream:
        for line in file_stream:
            if line.endswith('\n'):
                line = line[:-len('\n')]

            if len(line) == 0:
                continue

            annotation_path, video_resolution, images_dir = line.split(' ')
            video_resolution = [int(x) for x in video_resolution.split(',')]

            if not exists(annotation_path) or not exists(images_dir):
                continue

            tasks.append((annotation_path, images_dir, video_resolution))

    return tasks


def _parse_images(images_dir):
    """Parse valid image names from the specified directory path.

    :param images_dir: Path directory with images
    :return: List of valid image paths
    """

    all_files = [join(images_dir, f) for f in listdir(images_dir) if isfile(join(images_dir, f))]

    data = {}
    for file_path in all_files:
        file_name, extension = basename(file_path).split('.')
        if extension not in VALID_EXTENSIONS:
            continue

        frame_id = int(file_name.split('_')[-1])
        if frame_id < 0:
            continue

        data[frame_id] = file_path

    return data


def _calc_iou(box_a, box_b):
    """Calculates IoU metric between two boxes.

    :param box_a: First box coordinates
    :param box_b: Second box coordinates
    :return: Scalar IoU value
    """

    intersect_ymin = np.maximum(box_a['ymin'], box_b['ymin'])
    intersect_xmin = np.maximum(box_a['xmin'], box_b['xmin'])
    intersect_ymax = np.minimum(box_a['ymax'], box_b['ymax'])
    intersect_xmax = np.minimum(box_a['xmax'], box_b['xmax'])

    intersect_height = np.maximum(0.0, intersect_ymax - intersect_ymin)
    intersect_width = np.maximum(0.0, intersect_xmax - intersect_xmin)

    intersect_area = intersect_height * intersect_width
    area_a = (box_a['ymax'] - box_a['ymin']) * (box_a['xmax'] - box_a['xmin'])
    area_b = (box_b['ymax'] - box_b['ymin']) * (box_b['xmax'] - box_b['xmin'])

    union_area = area_a + area_b - intersect_area

    overlap_ratio = intersect_area / union_area if union_area > 0.0 else 0.0

    return overlap_ratio


def _try_to_merge(all_tracks, candidate, id_shift, min_iou_overlap, max_time_overlap):
    """Merges overlapped tracks into single if possible.

    :param all_tracks: List of merged tracks
    :param candidate: List of candidate tracks
    :param id_shift: ID shift
    :param min_iou_overlap: Min IoU value to merge tracks
    :param max_time_overlap: Max time overlap to merge
    :return: Returns True if success
    """

    candidate_frames_ids = list(candidate)
    candidate_start_id = np.min(candidate_frames_ids)
    candidate_end_id = np.max(candidate_frames_ids) + 1

    best_bbox_overlap = 0.0
    best_bbox_overlap_track_id = -1
    for anchor_track_id in all_tracks:
        anchor_track = all_tracks[anchor_track_id]
        anchor_frames_ids = list(anchor_track)
        anchor_start_id = np.min(anchor_frames_ids)
        anchor_end_id = np.max(anchor_frames_ids) + 1

        time_overlap_start = np.maximum(candidate_start_id, anchor_start_id)
        time_overlap_end = np.minimum(candidate_end_id, anchor_end_id)
        time_overlap = np.maximum(0, time_overlap_end - time_overlap_start)

        if 0 < time_overlap <= max_time_overlap:
            for overlap_frame_id in xrange(time_overlap_start, time_overlap_end):
                if overlap_frame_id not in anchor_track or overlap_frame_id not in candidate:
                    continue

                anchor_bbox = anchor_track[overlap_frame_id]
                ref_bbox = candidate[overlap_frame_id]

                bbox_overlap = _calc_iou(anchor_bbox, ref_bbox)
                if bbox_overlap > best_bbox_overlap:
                    best_bbox_overlap = bbox_overlap
                    best_bbox_overlap_track_id = anchor_track_id

    if best_bbox_overlap_track_id >= 0 and best_bbox_overlap > min_iou_overlap:
        anchor_track = all_tracks[best_bbox_overlap_track_id]
        for frame_id in candidate:
            anchor_track[frame_id] = candidate[frame_id]
            anchor_track[frame_id]['track_id'] = best_bbox_overlap_track_id + id_shift
        return True
    else:
        return False


def _read_annotation(annotation_path, image_size, id_shift, action_names_map, ignore_occluded=False,
                     valid_action_ids=None, min_iou_overlap=0.8, max_time_overlap=5, allow_extend_map=False):
    """Loads annotation from the specified file and merges overlapped tracks.

    :param annotation_path: Path to annotation file
    :param image_size: Size of image
    :param id_shift: ID shift
    :param action_names_map: Map of input actions
    :param ignore_occluded: Whether to ignore occluded boxes
    :param valid_action_ids: List of valid action IDs
    :param min_iou_overlap: Min IoU value to merge tracks
    :param max_time_overlap: Max time overlap to merge
    :param allow_extend_map: Whether to allow extending labels map
    :return: Loaded annotation
    """

    tree = etree.parse(annotation_path)
    root = tree.getroot()

    converted_tracks = {}
    for track_id, track in enumerate(root):
        if 'label' not in track.attrib or track.attrib['label'] != 'person':
            continue

        if 'id' in track.attrib:
            track_id = int(track.attrib['id'])
            assert track_id < IDS_SHIFT_SCALE, 'Invalid ID: {}'.format(track_id)

        is_new_track = track_id not in converted_tracks
        converted_track = {} if is_new_track else converted_tracks[track_id]

        for bbox in track:
            if len(bbox) < 1:
                continue

            frame_id = int(bbox.attrib['frame'])
            if frame_id < 0:
                continue

            action_name = None
            for bbox_attr_id, _ in enumerate(bbox):
                attribute_name = bbox[bbox_attr_id].attrib['name']
                if attribute_name != 'action':
                    continue

                action_name = bbox[bbox_attr_id].text

            if action_name is None:
                continue

            if action_name not in action_names_map:
                if allow_extend_map:
                    action_names_map[action_name] = len(action_names_map)
                else:
                    continue

            action_name_id = action_names_map[action_name]
            if valid_action_ids is not None and action_name_id not in valid_action_ids:
                continue

            is_occluded = bbox.attrib['occluded'] == '1'
            if ignore_occluded and is_occluded:
                action_name_id = action_names_map['__undefined__']

            bbox_desc = {'label': action_name_id,
                         'track_id': track_id + id_shift,
                         'occluded': is_occluded,
                         'xmin': float(bbox.attrib['xtl']) / float(image_size[1]),
                         'ymin': float(bbox.attrib['ytl']) / float(image_size[0]),
                         'xmax': float(bbox.attrib['xbr']) / float(image_size[1]),
                         'ymax': float(bbox.attrib['ybr']) / float(image_size[0])}

            converted_track[frame_id] = bbox_desc

        if is_new_track and len(converted_track) > 0:
            is_merged = _try_to_merge(converted_tracks, converted_track, id_shift, min_iou_overlap, max_time_overlap)
            if not is_merged:
                converted_tracks[track_id] = converted_track

    detections_by_frame_id = {}
    for track_id in converted_tracks:
        out_track = converted_tracks[track_id]
        for frame_id in out_track:
            detections_by_frame_id[frame_id] = detections_by_frame_id.get(frame_id, []) + [out_track[frame_id]]

    return detections_by_frame_id


def create_dir(dir_path):
    """Creates directory if needed.

    :param dir_path: Path to new directory
    """

    if exists(dir_path):
        rmtree(dir_path)

    makedirs(dir_path)


def save_data_paths(data, out_path):
    """Saves paths to data into specified file.

    :param data: data to save
    :param out_path: Path to save
    """

    with open(out_path, 'w') as input_stream:
        for image_path, annot_path in tqdm(data, desc='Dumping image paths'):
            input_stream.write('{} {}\n'.format(image_path, annot_path))


def save_class_mapping(class_mapping, out_path):
    """Saves class mapping from ID to name onto disk.

    :param class_mapping: Class mapping
    :param out_path: path to save
    """

    with open(out_path, 'w') as output_stream:
        for class_name in class_mapping:
            class_id = class_mapping[class_name]

            output_stream.write('\"{}\": {}\n'.format(class_name, class_id))


def main():
    """Main function.
    """

    parser = ArgumentParser()
    parser.add_argument('--tasks_path', '-t', type=str, required=True, help='Path to source LMDB')
    parser.add_argument('--out_dir', '-o', type=str, required=True, help='Path to save data')
    parser.add_argument('--input_map', '-i', type=str, required=False, default='', help='Path to class names')
    args = parser.parse_args()

    assert exists(args.tasks_path)

    if exists(args.input_map):
        with open(args.input_map, 'r') as config_file:
            glob_action_names_map = yaml.load(config_file, Loader=yaml.FullLoader)
        print('Loaded map with {} actions.'.format(len(glob_action_names_map)))
    else:
        glob_action_names_map = {}
        print('No class map is specified. Action names will be collected.')
    allow_extend_map = len(glob_action_names_map) == 0

    annotation_dir = join(args.out_dir, 'annotation')
    create_dir(annotation_dir)

    tasks = _parse_tasks(args.tasks_path)
    print('Found {} tasks:'.format(len(tasks)))
    for task_id, task in enumerate(tasks):
        print('   {}: {}'.format(task_id, task[0]))

    dumped_paths = []
    total_num_frames = 0
    total_num_bboxes = 0
    max_num_bboxes = 0

    for task_id, task in enumerate(tasks):
        print('Loading task {}...'.format(task[0]))

        annotation_path = task[0]
        images_dir = task[1]
        video_resolution = task[2]

        image_paths = _parse_images(images_dir)
        if len(image_paths) == 0:
            continue

        task_annotation_dir = join(annotation_dir, basename(annotation_path.replace('.xml', '')))
        makedirs(task_annotation_dir)

        id_shift = (task_id + 1) * IDS_SHIFT_SCALE
        annotation = _read_annotation(annotation_path, video_resolution, id_shift, glob_action_names_map,
                                      min_iou_overlap=0.5, max_time_overlap=5, allow_extend_map=allow_extend_map)

        for frame_id in tqdm(annotation, desc='Dumping'):
            if frame_id not in image_paths:
                continue

            image_path = image_paths[frame_id]
            gt_objects = annotation[frame_id]
            if len(gt_objects) == 0:
                continue

            sample_name = 'sample_{:06}'.format(frame_id)
            annot_path = join(task_annotation_dir, '{}.json'.format(sample_name))
            with open(annot_path, 'w') as out_stream:
                json_dump(gt_objects, out_stream)

            dumped_paths.append((image_path, annot_path))
            total_num_frames += 1
            total_num_bboxes += len(gt_objects)
            max_num_bboxes = max(max_num_bboxes, len(gt_objects))

    out_data_path = join(args.out_dir, 'data.txt')
    save_data_paths(dumped_paths, out_data_path)
    print('\nLoaded frames: {} with {} boxes. Max number bboxes on image: {}'
          .format(total_num_frames, total_num_bboxes, max_num_bboxes))

    if allow_extend_map:
        out_class_map_path = join(args.out_dir, 'class_map.txt')
        save_class_mapping(glob_action_names_map, out_class_map_path)
        print('\nAction names map is stored to: {}'.format(out_class_map_path))


if __name__ == '__main__':
    main()
