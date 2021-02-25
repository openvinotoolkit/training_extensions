#!/usr/bin/env python3
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

""" This script allows you to freeze Text Recognition model. """

import argparse
import os

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

from text_recognition.model import TextRecognition
from text_recognition.dataset import Dataset
from tfutils.helpers import execute_mo

def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Pretrained model path.')
    parser.add_argument('--data_type', default='FP32', choices=['FP32', 'FP16'], help='Data type of IR')
    parser.add_argument('--output_dir', default=None, help='Output Directory')
    return parser.parse_args()


def freezing_graph(sess, graph_file, output_node_names):
    """ Saves model as frozen graph."""

    assert graph_file.endswith('.pb')

    directory = os.path.dirname(graph_file)
    base = os.path.basename(graph_file)
    ckpt = graph_file.replace('.pb', '.ckpt')
    frozen = graph_file.replace('.pb', '.pb.frozen')

    os.system('mkdir -p {}'.format(directory))
    print('>> Saving `{}`... '.format(graph_file), end='')
    tf.train.write_graph(sess.graph, directory, base, as_text=False)
    print('Done')

    print('>> Saving `{}`... '.format(ckpt), end='')
    tf.train.Saver().save(sess, ckpt, write_meta_graph=False)
    print('Done')

    print('>> Running `freeze_graph.py`... ')
    print('Outputs:\n  {}'.format(', '.join(output_node_names)))

    freeze_graph(input_graph=graph_file,
                 input_saver='',
                 input_binary=True,
                 input_checkpoint=ckpt,
                 output_node_names=','.join(output_node_names),
                 restore_op_name='save/restore_all',
                 filename_tensor_name='save/Const:0',
                 output_graph=frozen,
                 clear_devices=True,
                 initializer_nodes='',
                 saved_model_tags='serve')

    return frozen


def main():
    """ Main freezing function. """

    args = parse_args()

    image_width = 120
    image_height = 32

    _, _, num_classes = Dataset.create_character_maps()

    model = TextRecognition(is_training=False, num_classes=num_classes)
    model_out = model(inputdata=tf.placeholder(tf.float32, [1, image_height, image_width, 1]))

    output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(args.checkpoint), 'export')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=args.checkpoint)
        graph_file = os.path.join(output_dir, 'graph.pb')
        frozen_graph = freezing_graph(sess, graph_file, output_node_names=[model_out.name[:-2]])

    mo_params = {
        'model_name': 'text_recognition',
        'data_type': args.data_type,
    }

    export_ir_dir = os.path.join(output_dir, 'IR', args.data_type)
    execute_mo(mo_params, frozen_graph, export_ir_dir)

if __name__ == '__main__':
    main()
