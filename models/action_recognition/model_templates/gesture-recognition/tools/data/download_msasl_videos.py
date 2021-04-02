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

import os
import json
import subprocess
from os import makedirs, listdir
from os.path import exists, join, isfile
from argparse import ArgumentParser

from joblib import delayed, Parallel # pylint: disable=import-error
import cv2


class VideoDownloader:
    def __init__(self, num_jobs, num_attempts=5):
        self.num_jobs = num_jobs
        assert self.num_jobs > 0

        self.num_attempts = num_attempts
        assert self.num_attempts > 0

    @staticmethod
    def _log(message_tuple):
        output_filename, status, msg = message_tuple
        str_template = '   - {}: {}' if status else '   - {}: Error: {}'
        print(str_template.format(output_filename, msg))

    @staticmethod
    def _get_number_of_frames(video_path):
        # Get number of frame that can be read
        # cap.get(cv2.CAP_PROP_FRAME_COUNT) can return incorrect value
        cap = cv2.VideoCapture(video_path)
        n = 0
        s = True
        while True:
            s, _ = cap.read()
            if not s:
                break
            n = n + 1
        return n

    def _download_video(self, video_data, output_filename, num_attempts=5):
        status = False

        command = ['youtube-dl',
                   '--quiet', '--no-warnings',
                   '-f', 'mp4',
                   '-o', '"%s"' % output_filename,
                   '"%s"' % video_data['url']]
        command = ' '.join(command)

        attempts = 0
        while True:
            try:
                _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                num_frame = self._get_number_of_frames(output_filename)
                if num_frame < video_data['end_frame']:
                    os.remove(output_filename)
                    if attempts == num_attempts:
                        return video_data['url'], status, 'Corrupted file or incorrect annotation'
                    attempts += 1
                    continue
            except subprocess.CalledProcessError as err:
                attempts += 1
                if attempts == num_attempts:
                    return video_data['url'], status, err.output
            else:
                break

        status = exists(output_filename)

        message_tuple = video_data['url'], status, 'Downloaded'
        self._log(message_tuple)

        return message_tuple

    def __call__(self, tasks):
        if len(tasks) == 0:
            return []

        if self.num_jobs == 1:
            status_lst = []
            for data, out_video_path in tasks:
                status_lst.append(self._download_video(data, out_video_path))
        else:
            status_lst = Parallel(n_jobs=self.num_jobs)(
                delayed(self._download_video)(data, out_video_path)
                for data, out_video_path in tasks
            )

        return status_lst


def ensure_dir_exists(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


def get_valid_sources(all_sources):
    return [s for s in all_sources if exists(s)]


def print_data_sources_stat(data_sources):
    print('Specified {} valid data sources:'.format(len(data_sources)))
    for data_source in data_sources:
        print('   - {}'.format(data_source))


def collect_videos(data_sources):
    out_videos = dict()
    for data_source in data_sources:
        with open(data_source) as input_stream:
            data = json.load(input_stream)

        for record in data:
            url = record['url']
            end_frame = record['end']
            video_name = url.split('?v=')[-1]
            if video_name not in out_videos:
                out_videos[video_name] = {'url': url, 'end_frame': end_frame}
            else:
                assert out_videos[video_name]['url'] == url

    return out_videos


def prepare_tasks(video_sources, videos_dir, extension):
    downloaded_videos = [join(videos_dir, f) for f in listdir(videos_dir)
                         if isfile(join(videos_dir, f)) and f.endswith(extension)]

    out_tasks = []
    for video_name, video_data in video_sources.items():
        video_path = join(videos_dir, '{}.{}'.format(video_name, extension))

        if video_path not in downloaded_videos:
            out_tasks.append((video_data, video_path))

    return out_tasks


def print_status(status_lst):
    if len(status_lst) == 0:
        return

    print('Status:')
    for status in status_lst:
        str_template = '   - {}: {}' if status[1] else '   - {}: Error: {}'
        print(str_template.format(status[0], status[2]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--sources', '-s', nargs='+', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--extension', '-e', type=str, required=False, default='mp4')
    parser.add_argument('--num_jobs', '-n', type=int, required=False, default=24)
    args = parser.parse_args()

    ensure_dir_exists(args.output_dir)

    data_sources = get_valid_sources(args.sources)
    print_data_sources_stat(data_sources)
    assert len(data_sources) > 0

    all_videos = collect_videos(data_sources)
    print('Found {} unique videos.'.format(len(all_videos)))

    tasks = prepare_tasks(all_videos, args.output_dir, args.extension)
    print('Prepared {} tasks for downloading.'.format(len(tasks)))

    downloader = VideoDownloader(args.num_jobs)
    status_lst = downloader(tasks)
    print_status(status_lst)


if __name__ == '__main__':
    main()
