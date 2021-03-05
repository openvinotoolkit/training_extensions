"""
 Copyright (c) 2019 Intel Corporation

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

import argparse
import os

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

from image_retrieval.model import keras_applications_mobilenetv2, keras_applications_resnet50

tf.compat.v1.disable_v2_behavior()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', required=True, help='Path to model weights.')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size.')
    parser.add_argument('--model', choices=['resnet50', 'mobilenet_v2'], required=True)
    parser.add_argument('--data_type', default='FP32', choices=['FP32', 'FP16'],
                        help='Data type of IR')
    parser.add_argument('--output_dir', default=None, help='Output Directory')
    return parser.parse_args()


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

    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph


def main():
    args = parse_args()

    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32,
                                            shape=[1, args.input_size, args.input_size, 3])
    if args.model == 'resnet50':
        model = keras_applications_resnet50(
            tf.keras.layers.Input(tensor=input_tensor))
    elif args.model == 'mobilenet_v2':
        model = keras_applications_mobilenetv2(
            tf.keras.layers.Input(tensor=input_tensor))
    else:
        raise Exception('unknown model')

    embedding = model(input_tensor, training=False)

    with tf.compat.v1.Session() as sess:
        model.load_weights(args.model_weights)
        model_dir = os.path.dirname(args.model_weights)
        export_folder = args.output_dir if args.output_dir else os.path.join(model_dir, 'export')

        tf.compat.v1.saved_model.simple_save(sess, export_folder,
                                             inputs={'input': input_tensor},
                                             outputs={embedding.name[:-2]: embedding})

        frozen_graph_path = os.path.join(export_folder, 'frozen_graph.pb')

        freeze_graph(
            input_graph=None,
            input_saver='',
            input_binary=True,
            input_checkpoint='',
            output_node_names=embedding.name[:-2],
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

        print("Path to frozen graph: %s" % frozen_graph_path)

if __name__ == '__main__':
    main()
