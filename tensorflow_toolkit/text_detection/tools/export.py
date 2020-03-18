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

""" This module performs freezing of text detection neural network. """

import argparse
import os

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

from text_detection.model import pixel_link_model
from text_detection.common import load_config


tf.compat.v1.disable_v2_behavior()


def arg_parser():
    """ Returns argument parser. """

    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', help='Path to trained weights.')
    parser.add_argument('--resolution', nargs=2, type=int, default=(1280, 768))
    parser.add_argument('--config', required=True, help='Path to training configuration file.')
    parser.add_argument('--output_dir', default=None, help='Output Directory')
    return parser

def print_flops(graph):
    """ Prints information about FLOPs. """

    with graph.as_default():
        flops = tf.compat.v1.profiler.profile(
            graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        print('')
        if flops.total_float_ops > 10 ** 9:
            print('Operations number: {} GFlops'.format(flops.total_float_ops / 10 ** 9))
        elif flops.total_float_ops > 10 ** 6:
            print('Operations number: {} MFlops'.format(flops.total_float_ops / 10 ** 6))
        elif flops.total_float_ops > 10 ** 3:
            print('Operations number: {} KFlops'.format(flops.total_float_ops / 10 ** 3))

    return flops


def load_frozen_graph(frozen_graph_filename):
    """ Loads and returns frozen graph. """

    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(file.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph


def freeze(args, config):
    """ Exports model to TF 1.x saved_model (simple_save) and freezes graph. """
    tf.keras.backend.set_learning_phase(0)
    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32,
                                            shape=[1, None, None, 3])
    model = pixel_link_model(tf.keras.Input(tensor=input_tensor), config=config)
    segm_logits, link_logits = model(input_tensor, training=False)

    link_logits = tf.reshape(link_logits, tf.concat([tf.shape(link_logits)[0:3], [config['num_neighbours'] * 2]], -1))

    export_folder = args.output_dir if args.output_dir else os.path.join(os.path.dirname(args.weights), 'export')

    with tf.compat.v1.Session() as sess:
        model.load_weights(args.weights)

        tf.compat.v1.saved_model.simple_save(sess, export_folder,
                                             inputs={'input': input_tensor},
                                             outputs={segm_logits.name[:-2]: segm_logits,
                                                      link_logits.name[:-2]: link_logits})

        frozen_graph_path = os.path.join(export_folder, 'frozen_graph.pb')

        output_node_names = (segm_logits.name[:-2], link_logits.name[:-2])
        freeze_graph(
            input_graph=None,
            input_saver='',
            input_binary=True,
            input_checkpoint='',
            output_node_names=','.join(output_node_names),
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=frozen_graph_path,
            clear_devices=True,
            initializer_nodes='',
            input_meta_graph=None,
            input_saved_model_dir=export_folder,
        )

        graph = load_frozen_graph(frozen_graph_path)
        print_flops(graph)

        print('')
        print('Output tensor names for using in InferenceEngine:')
        print('     model/link_logits_/add')
        print('     model/segm_logits/add')
        print('')
        print('Run model_optimizer to get IR: mo.py --input_model {} --framework tf'.format(
            frozen_graph_path))


def main():
    """ Main function. """
    args = arg_parser().parse_args()
    config = load_config(args.config)
    freeze(args, config)


if __name__ == '__main__':
    main()
