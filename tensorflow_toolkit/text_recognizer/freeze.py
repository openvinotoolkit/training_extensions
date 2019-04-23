""" This script allows you to freeze Text Recognition model. """

import argparse
import os

import tensorflow as tf

from model import TextRecognition
from dataset import Dataset


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', required=True, help='Pretrained model path.')

    return parser.parse_args()


def dump_for_tfmo(sess, graph_file, output_node_names):
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

    from tensorflow.python.tools.freeze_graph import freeze_graph
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

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=args.weights_path)
        graph_file = 'dump/graph.pb'
        dump_for_tfmo(sess, graph_file, output_node_names=[model_out.name[:-2]])


if __name__ == '__main__':
    main()
