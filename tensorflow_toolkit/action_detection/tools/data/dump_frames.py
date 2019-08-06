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

from argparse import ArgumentParser
from os import makedirs, listdir
from os.path import exists, basename, join, isdir
from shutil import rmtree

import cv2
from lxml import etree
from tqdm import tqdm


def clear_or_create(path):
    """Clears directory if exists or creates new one.

    :param path: Path to directory
    """

    if exists(path):
        rmtree(path)
    makedirs(path)


def create_next_empty_dir(root, default_name='dumped'):
    """Creates folder structure.

    :param root: Path to root directory
    :param default_name: Name of directory with dumped frames
    :return: Path to new directory
    """

    if exists(root):
        candidates = [join(root, o) for o in listdir(root)
                      if isdir(join(root, o)) and o.startswith(default_name)]

        if len(candidates) > 0:
            max_index = 0
            for candidate in candidates:
                parts = candidate.split('_')
                if len(parts) > 1:
                    max_index = max(max_index, int(parts[1]))
            empty_dir_path = join(root, '{}_{}'.format(default_name, max_index + 1))
        else:
            empty_dir_path = join(root, default_name)
    else:
        empty_dir_path = join(root, default_name)

    makedirs(empty_dir_path)

    return empty_dir_path


def parse_tasks(file_path):
    """Parse input file with tasks, where each task is presented by line in file: [path_to_annotation path_to_video]

    :param file_path: Path to file with tasks
    :return: List of decoded tasks
    """

    print('Found tasks:')

    tasks = []
    with open(file_path, 'r') as input_stream:
        for line in input_stream:
            if line.endswith('\n'):
                line = line[:-len('\n')]

            if len(line) == 0:
                continue

            annotation_path, video_path = line.split(' ')

            if not exists(annotation_path) or not exists(video_path):
                continue

            tasks.append((annotation_path, video_path))
            print('   #{}: {} {}'.format(len(tasks), annotation_path, video_path))

    return tasks


def parse_frame_ids(annotation_path):
    """Reads unique frames IDs from the specified annotation file.

    :param annotation_path:
    :return: List of unique frame IDs
    """

    tree = etree.parse(annotation_path)
    root = tree.getroot()

    out_frame_ids = []
    for track in tqdm(root, desc='Extracting annotation'):
        if 'label' not in track.attrib or track.attrib['label'] != 'person':
            continue

        for bbox in track:
            if len(bbox) < 1:
                continue

            frame_id = int(bbox.attrib['frame'])

            action_name = None
            for bbox_attr_id in xrange(len(bbox)):
                attribute_name = bbox[bbox_attr_id].attrib['name']
                if attribute_name != 'action':
                    continue

                action_name = bbox[bbox_attr_id].text

            if action_name is None and action_name == '':
                continue

            out_frame_ids.append(frame_id)

    unique_frame_ids = set(out_frame_ids)

    print('Loaded {} annotated frames.'.format(len(unique_frame_ids)))

    return unique_frame_ids


def dump_frames(video_path, frame_ids, trg_height, trg_width, out_dir):
    """Dumps frames from the specified video file.

    :param video_path: Path to video file
    :param frame_ids: List of frame IDs to dump
    :param trg_height: Target image height
    :param trg_width: Target image width
    :param out_dir: Output directory to dump frames
    :return: Number of dumped frames
    """

    vidcap = cv2.VideoCapture(video_path)

    pbar = tqdm(total=len(frame_ids), desc='Dumping frames')

    success = True
    frame_id = -1
    num_collected_frames = 0

    while success:
        success, frame = vidcap.read()
        frame_id += 1

        if success:
            if frame_id not in frame_ids:
                continue

            image_path = join(out_dir, 'frame_{:06}.jpg'.format(frame_id))
            out_image = cv2.resize(frame, (trg_width, trg_height))
            cv2.imwrite(image_path, out_image)

            pbar.update(1)

            num_collected_frames += 1
            if num_collected_frames >= len(frame_ids):
                break

    pbar.close()

    return frame_id


def main():
    """Main function to dump frames for the specified tasks.
    """

    parser = ArgumentParser()
    parser.add_argument('--tasks', '-t', type=str, required=True, help='Path to the file with tasks')
    parser.add_argument('--out_dir', '-o', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--img_size', type=str, required=False, default='1080,1920', help='Target image size: HxW')
    args = parser.parse_args()

    assert exists(args.tasks)

    tasks = parse_tasks(args.tasks)
    trg_height, trg_width = map(int, args.img_size.split(','))

    working_dir = create_next_empty_dir(args.out_dir)

    out_tasks_file_path = join(working_dir, 'tasks.txt')
    out_tasks_stream = open(out_tasks_file_path, 'w')

    num_frames = 0
    for task_id, task in enumerate(tasks):
        task_name = basename(task[0]).split('.')[0]
        print('\nProcessing task {} / {}: \'{}\''.format(task_id + 1, len(tasks), task_name))

        valid_frame_ids = parse_frame_ids(task[0])

        output_task_dir = join(working_dir, task_name)
        clear_or_create(output_task_dir)

        task_num_frames = dump_frames(task[1], valid_frame_ids, trg_height, trg_width, output_task_dir)

        if task_num_frames <= 0:
            print('No frames are found to dump')
            continue

        print('Dumped {} frames'.format(task_num_frames))

        num_frames += task_num_frames

        out_tasks_stream.write('{} {}\n'.format(task[0], output_task_dir))

    out_tasks_stream.close()

    print('\nTotal dumped {} frames'.format(num_frames))
    print('Out list of tasks is saved to: {}'.format(out_tasks_file_path))


if __name__ == '__main__':
    main()
