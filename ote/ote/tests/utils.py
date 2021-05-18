# Copyright (C) 2020-2021 Intel Corporation
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

import glob
import re
import os


def collect_ap(path):
    ap = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        for line in content:
            if line.startswith(beginning):
                ap.append(float(line.replace(beginning, '')))
    return ap


def download_if_not_yet(output_folder, url):
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, os.path.basename(url))
    if not os.path.exists(path):
        os.system(f'wget --no-verbose {url} -P {output_folder}')
    return path


def relative_abs_error(expected, actual):
    return abs(expected - actual) / expected

def find_files_by_pattern(folder_path, pattern):
    found_files = glob.glob(os.path.join(folder_path, '**', pattern), recursive=True)
    found_files = sorted(found_files)
    return found_files

def extract_last_lines_by_pattern(file_path, regex, num=1):
    res = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if re.search(regex, line):
                res.append(line)
    res = res[-num:]
    return res
