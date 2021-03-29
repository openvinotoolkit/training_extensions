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

from os.path import exists
from argparse import ArgumentParser


class RawFramesSegmentedRecord:
    def __init__(self, row):
        self._data = row

        assert self.label >= 0

    @property
    def label(self):
        return int(self._data[1])

    @property
    def data(self):
        return self._data


def load_records(ann_file):
    return [RawFramesSegmentedRecord(x.strip().split(' ')) for x in open(ann_file)]


def filter_records(records, k):
    return [record for record in records if record.label < k]


def dump_records(records, out_file_path):
    with open(out_file_path, 'w') as out_stream:
        for record in records:
            out_stream.write('{}\n'.format(' '.join(record.data)))


def main():
    parser = ArgumentParser()
    parser.add_argument('--annot', '-a', nargs='+', type=str, required=True)
    parser.add_argument('--topk', '-k', nargs='+', type=int, required=True)
    args = parser.parse_args()

    records = dict()
    for annot_path in args.annot:
        assert exists(annot_path)

        records[annot_path] = load_records(annot_path)

    for k in args.topk:
        for annot_path in records:
            filtered_records = filter_records(records[annot_path], k)

            out_path = '{}{}.txt'.format(annot_path[:-len('.txt')], k)
            dump_records(filtered_records, out_path)


if __name__ == '__main__':
    main()
