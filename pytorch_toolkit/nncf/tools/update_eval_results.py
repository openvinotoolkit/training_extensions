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

This is a helper script which parses a metrics.json file containing currently measured accuracy values for checkpoints
registered in tests/sota_checkpoints_eval.json, and produces:
a) a file with exactly the same contents as tests/sota_checkpoints_eval.json, but with target accuracy scores
 updated to reflect what was currently measured and reported in metrics.json, and
b) a set of .MD file snippets containing updated results tables, ready for copy-and-paste into corresponding
 README.md files (a common frontpage README and sample-specific readmes).

Usage:
python update_eval_results.py --results path/to/metrics.json --config path/to/sota_checkpoints_eval.json
-o path/to/new_config.json
"""


import os
import json
from collections import OrderedDict
from mdutils import MdUtils
import argparse


def examples_table_maker(sample_type, sota_checkpoints_dict, metrics):
    examples_row = ["Model", "Compression algorithm", "Dataset", "PyTorch compressed accuracy", "Config path",
                    "PyTorch Checkpoint"]
    common_row = ["Model", "Compression algorithm", "Dataset", "PyTorch FP32 baseline", "PyTorch compressed accuracy"]
    for sample_ in sota_checkpoints_dict:
        datasets = sota_checkpoints_dict[sample_]
        for data_name_ in datasets:
            dataset_name = data_conversion(data_name_)
            model_name = datasets[data_name_]
            for configs in model_name:
                conf_file = model_name[configs].get('config', {})
                reference = None
                if model_name[configs].get('reference', {}):
                    reference = model_name[configs].get('reference', {})
                if model_name[configs].get('resume', {}):
                    resume = model_name[configs].get('resume', {})
                else:
                    resume = os.path.basename(conf_file).replace(".json", ".pth")
                table_name = model_name[configs].get('model_description', {})
                if model_name[configs].get('compression_description', {}):
                    compression = model_name[configs].get('compression_description', {})
                else:
                    compression = None
                if compression is None and sample_type == "classification" and ("mobilenet_v2" not in model_name):
                    checkpoint_link = "-"
                else:
                    checkpoint_link = 'https://download.01.org/opencv/openvino_training_extensions/models/nncf/'\
                                      + resume
                if sample_ == sample_type:
                    row_ext = [str(table_name), str(compression), str(dataset_name), str(metrics[configs]),
                               str(conf_file), str(checkpoint_link)]
                    examples_row.extend(row_ext)
                    md_table_writer(sample_, examples_row)
                fp32_ref_metric = fp32_ref(reference, metrics)
                common_row_ext = [str(table_name), str(compression), str(dataset_name), str(fp32_ref_metric),
                                  str(metrics[configs])]
                common_row.extend(common_row_ext)
                md_table_writer("common", common_row)
                if args.output is not None:
                    model_name[configs]['target'] = measured_metrics[configs]


def fp32_ref(reference_, metrics_):
    if reference_ is not None:
        fp32_ref_metric = metrics_[reference_]
    else:
        fp32_ref_metric = "-"
    return fp32_ref_metric


def data_conversion(data_name):
    if data_name == 'imagenet':
        dataset_name = 'ImageNet'
    elif data_name == 'camvid':
        dataset_name = 'CamVid'
    elif data_name == 'VOCdevkit':
        dataset_name = 'VOC12+07'
    else:
        dataset_name = "Mapillary"
    return dataset_name


def lines_delete(file_name):
    with open(file_name) as input_data:
        lines = input_data.readlines()
    with open(file_name, 'w') as out:
        out.writelines(lines[4:])


def md_table_writer(name, rows):
    if name == 'common':
        mdfile = MdUtils(file_name='results_common')
        mdfile.new_table(columns=5, rows=int(len(rows)/5), text=rows, text_align='center')
    else:
        mdfile = MdUtils(file_name='results_{}'.format(name))
        mdfile.new_table(columns=6, rows=int(len(rows)/6), text=rows, text_align='center')
    mdfile.create_md_file()
    lines_delete('results_{}.md'.format(name))


parser = argparse.ArgumentParser()
parser.add_argument('--results', '-r', help='Metrics file')
parser.add_argument('--config', '-c', help='Config .json file')
parser.add_argument('--output', '-o', help="Added new target value")
args = parser.parse_args()
results = args.results
config = args.config
output = args.output

measured_metrics = json.load(open(results, 'r'))
sota_checkpoints_eval = json.load(open(config), object_pairs_hook=OrderedDict)

for sample_type_ in sota_checkpoints_eval:
    examples_table_maker(sample_type_, sota_checkpoints_eval, measured_metrics)

if args.output is not None:
    with open(output, "w") as write_file:
        json.dump(sota_checkpoints_eval, write_file, indent=8)
