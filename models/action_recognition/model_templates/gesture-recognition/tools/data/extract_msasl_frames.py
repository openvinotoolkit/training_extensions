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

import json
from os import makedirs, listdir
from os.path import exists, join, isfile, basename
from argparse import ArgumentParser

import numpy as np
import cv2
from tqdm import tqdm


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def get_valid_sources(all_sources):
    return [s for s in all_sources if exists(s)]


def print_data_sources_stat(data_sources):
    print('Specified {} valid data sources:'.format(len(data_sources)))
    for data_source in data_sources:
        print('   - {}'.format(data_source))


def load_sign_durations(data_path):
    with open(data_path) as input_stream:
        data = json.load(input_stream)

    return {int(k): float(v) for k, v in data.items()}


def get_video_names(videos_dir, extension):
    return [f.split('.{}'.format(extension))[0]
            for f in listdir(videos_dir) if isfile(join(videos_dir, f)) and f.endswith(extension)]


def collect_records(data_sources, valid_video_names, mean_sign_duration=0.8, speed_factor=1.1, preparation_time=0.8):
    num_skipped_records = 0
    out_records = list()
    for data_source in data_sources:
        with open(data_source) as input_stream:
            data = json.load(input_stream)

        data_type = basename(data_source).split('.')[0].split('_')[-1]

        for record in data:
            url = record['url']
            video_name = url.split('?v=')[-1]

            if video_name not in valid_video_names:
                num_skipped_records += 1
                continue

            bbox = record['box']
            start_frame = int(record['start'])
            end_frame = int(record['end'])
            assert start_frame >= 0 and end_frame >= 0
            if end_frame - start_frame <= 1:
                num_skipped_records += 1
                continue

            label = record['label']
            expected_sign_duration = speed_factor * mean_sign_duration * float(record['fps'])
            expected_clip_duration = expected_sign_duration + preparation_time * float(record['fps'])
            real_clip_duration = float(end_frame - start_frame)

            if real_clip_duration > expected_clip_duration:
                left_duration = real_clip_duration - preparation_time * float(record['fps'])

                ratio = left_duration / expected_sign_duration
                num_repeats = int(np.round(ratio))

                num_segments = num_repeats + 1 if num_repeats > 0 else 2
                segment_length = real_clip_duration / float(num_segments)

                center_frame = start_frame + segment_length  # get the first most probable position
                start_frame = int(center_frame - 0.5 * expected_sign_duration)
                end_frame = int(center_frame + 0.5 * expected_sign_duration)
                out_limits = [(start_frame, end_frame)]
            else:
                center_frame = 0.5 * (start_frame + end_frame)
                trg_sign_duration = np.minimum(real_clip_duration, expected_sign_duration)
                start_frame = int(center_frame - 0.5 * trg_sign_duration)
                end_frame = int(center_frame + 0.5 * trg_sign_duration)
                out_limits = [(start_frame, end_frame)]

            for fixed_start_frame, fixed_end_frame in out_limits:
                out_records.append(dict(label=label,
                                        signer_id=record['signer_id'],
                                        start=fixed_start_frame,
                                        end=fixed_end_frame,
                                        bbox=dict(xmin=bbox[1], ymin=bbox[0], xmax=bbox[3], ymax=bbox[2]),
                                        video_name=video_name,
                                        fps=float(record['fps']),
                                        type=data_type))

    if num_skipped_records > 0:
        print('Warning. Skipped {} records.'.format(num_skipped_records))
    else:
        print('All records are parsed successfully.')

    return out_records


def order_by_video_name(records):
    out_records = dict()
    for record in records:
        video_name = record['video_name']
        if video_name not in out_records:
            out_records[video_name] = []

        out_records[video_name].append(record)

    return out_records


def validate_and_sort_records(records):
    out_records = dict()
    for video_name in records:
        video_records = records[video_name]
        video_records.sort(key=lambda r: r['start'])

        out_video_records = list()
        for record_id in range(len(video_records) - 1):
            cur_record = video_records[record_id]
            next_record = video_records[record_id + 1]

            if cur_record['end'] > next_record['start']:
                cur_record['end'] = next_record['start']

            if cur_record['start'] < cur_record['end']:
                out_video_records.append(cur_record)

        if len(video_records) > 0:
            out_video_records.append(video_records[-1])

        out_records[video_name] = out_video_records

    return out_records


def print_records_stat(records):
    num_records = np.sum([len(video_records) for video_records in records.values()])
    clip_lengths = [record['end'] - record['start'] for video_records in records.values() for record in video_records]

    print('Stat for {} records:'.format(num_records))
    print('   - min: {} max: {}'.format(np.min(clip_lengths),
                                        np.max(clip_lengths)))
    print('   - p@5: {} p@50: {} p@95: {}'.format(np.percentile(clip_lengths, 5.0),
                                                  np.percentile(clip_lengths, 50.0),
                                                  np.percentile(clip_lengths, 95.0)))


def crop_image(image, bbox, trg_size, scale):
    def _fix_bbox():
        frame_height, frame_width = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

        center_x = 0.5 * (x_min + x_max) * frame_width
        center_y = 0.5 * (y_min + y_max) * frame_height

        scaled_width = scale * (x_max - x_min) * frame_width
        scaled_height = scale * (y_max - y_min) * frame_height

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


def extract_frames(records, videos_dir, video_name_template, out_dir, image_name_template,
                   target_num_frames, trg_size, scale, trg_fps=30.):
    pbar = tqdm(total=len(records), desc='Dumping')

    video_names = list(records)
    for video_name in video_names:
        video_records = records[video_name]
        video_path = join(videos_dir, video_name_template.format(video_name))
        video_capture = cv2.VideoCapture(video_path)
        num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_ids_map = dict()
        for record_id, _ in enumerate(video_records):
            record = video_records[record_id]

            left_limit = video_records[record_id - 1]['end'] if record_id > 0 else 0
            right_limit = video_records[record_id + 1]['start'] if record_id + 1 < len(video_records) else num_frames

            fps_factor = trg_fps / record['fps']
            time_delta = np.maximum(int(target_num_frames / fps_factor), record['end'] - record['start'])
            record['video_start'] = np.maximum(left_limit, record['end'] - time_delta)
            record['video_end'] = np.minimum(right_limit, record['start'] + time_delta)
            for i in range(record['video_start'], record['video_end']):
                if i not in frame_ids_map:
                    frame_ids_map[i] = []
                frame_ids_map[i].append(record_id)

            record['rel_images_dir'] = join(video_name, 'clip_{:04}'.format(record_id))
            abs_images_dir = join(out_dir, record['rel_images_dir'])
            ensure_dir_exists(abs_images_dir)

        success = True
        read_frame_id = -1
        while success:
            success, frame = video_capture.read()
            read_frame_id += 1

            if not success or read_frame_id not in frame_ids_map:
                continue

            for record_id in frame_ids_map[read_frame_id]:
                record = video_records[record_id]

                cropped_frame = crop_image(frame, record['bbox'], trg_size, scale)

                images_dir = join(out_dir, record['rel_images_dir'])
                out_image_path = join(images_dir, image_name_template.format(read_frame_id - record['video_start'] + 1))
                cv2.imwrite(out_image_path, cropped_frame)

        video_capture.release()
        pbar.update(1)

    pbar.close()

    return records


def split_data(records):
    out_data = dict()
    for video_records in records.values():
        for record in video_records:
            record_type = record['type']
            if record_type not in out_data:
                out_data[record_type] = []

            out_data[record_type].append(record)

    return out_data


def dump_paths(data, out_dir):
    for data_type in data:
        records = data[data_type]

        out_path = join(out_dir, '{}.txt'.format(data_type))
        with open(out_path, 'w') as out_stream:
            record_ids = list(range(len(records)))
            if data_type == 'train':
                np.random.shuffle(record_ids)

            for record_id in record_ids:
                record = records[record_id]

                converted_record = (
                    record['rel_images_dir'],
                    str(record['label']),
                    str(record['start'] - record['video_start']),
                    str(record['end'] - record['video_start']),
                    str(0),
                    str(record['video_end'] - record['video_start']),
                    str(record['fps'])
                )
                out_stream.write('{}\n'.format(' '.join(converted_record)))


def main():
    parser = ArgumentParser()
    parser.add_argument('--sources', '-s', nargs='+', type=str, required=True)
    parser.add_argument('--videos_dir', '-v', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--video_extension', '-ve', type=str, required=False, default='mp4')
    parser.add_argument('--image_extension', '-ie', type=str, required=False, default='jpg')
    parser.add_argument('--target_num_frames', '-l', type=int, required=False, default=64)
    parser.add_argument('--trg_image_size', type=str, required=False, default='300,256')
    parser.add_argument('--scale', type=float, required=False, default=1.2)
    args = parser.parse_args()

    assert exists(args.videos_dir)

    images_out_dir = join(args.output_dir, 'global_crops')
    ensure_dir_exists(images_out_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    available_video_names = get_video_names(args.videos_dir, args.video_extension)
    records = order_by_video_name(collect_records(data_sources, available_video_names))
    valid_records = validate_and_sort_records(records)
    print_records_stat(valid_records)

    trg_size = [int(v) for v in args.trg_image_size.split(',')]
    video_name_template = '{}' + '.{}'.format(args.video_extension)
    image_name_template = 'img_{:05}' + '.{}'.format(args.image_extension)
    extended_records = extract_frames(valid_records, args.videos_dir, video_name_template,
                                      images_out_dir, image_name_template, args.target_num_frames,
                                      trg_size, args.scale)

    data_splits = split_data(extended_records)
    dump_paths(data_splits, args.output_dir)


if __name__ == '__main__':
    main()
