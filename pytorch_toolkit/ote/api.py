import argparse

import yaml


def train_args_parser(template_path):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    with open(template_path, 'r') as model_definition:
        config = yaml.safe_load(model_definition)

        parser.add_argument('--train-ann-files', required=True,
                            help='Comma-separated paths to training annotation files.')
        parser.add_argument('--train-img-roots', required=True,
                            help='Comma-separated paths to training images folders.')
        parser.add_argument('--val-ann-files', required=True,
                            help='Comma-separated paths to validation annotation files.')
        parser.add_argument('--val-img-roots', required=True,
                            help='Comma-separated paths to validation images folders.')
        parser.add_argument('--resume-from', default='',
                            help='Resume training from previously saved checkpoint')
        parser.add_argument('--load-weights', default='',
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--save-checkpoints-to', default='/tmp/checkpoints',
                            help='Location where checkpoints will be stored')
        parser.add_argument('--epochs', type=int,
                            default=config['hyper_parameters']['epochs'],
                            help='Number of epochs during training')
        parser.add_argument('--batch-size', type=int,
                            default=config['hyper_parameters']['batch_size'],
                            help='Size of a single batch during training per GPU.')
        parser.add_argument('--base-learning-rate', type=float,
                            default=config['hyper_parameters']['base_learning_rate'],
                            help='Starting value of learning rate that might be changed during '
                                 'training according to learning rate schedule that is usually '
                                 'defined in detailed training configuration.')
        parser.add_argument('--gpu-num', type=int,
                            default=config['gpu_num'],
                            help='Number of GPUs that will be used in training, 0 is for CPU mode.')
        parser.add_argument('--config', default=config['config'], help=argparse.SUPPRESS)

    return parser


def test_args_parser(template_path):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    with open(template_path, 'r') as model_definition:
        config = yaml.safe_load(model_definition)

        parser.add_argument('--test-ann-files', required=True,
                            help='Comma-separated paths to test annotation files.')
        parser.add_argument('--test-img-roots', required=True,
                            help='Comma-separated paths to test images folders.')
        parser.add_argument('--load-weights', required=True,
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--save-metrics-to', required=True,
                            help='Location where evaluated metrics values will be stored (yaml file).')
        parser.add_argument('--save-output-images-to', default='',
                            help='Location where output images (with displayed result of model work) will be stored.')
        parser.add_argument('--config', default=config['config'], help=argparse.SUPPRESS)

    return parser


def export_args_parser(template_path):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    with open(template_path, 'r') as model_definition:
        config = yaml.safe_load(model_definition)
        parser.add_argument('--load-weights', required=True,
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--save-model-to', required='True',
                            help='Location where exported model will be stored.')
        parser.add_argument('--onnx', action='store_true', default=config['output_format']['onnx']['default'],
                            help='Enable onnx export.')
        parser.add_argument('--openvino', action='store_true',
                            default=config['output_format']['openvino']['default'],
                            help='Enable OpenVINO export.')
        parser.add_argument('--openvino-input-format',
                            default=config['output_format']['openvino']['input_format'],
                            help='Format of an input image for OpenVINO exported model.')
        parser.add_argument('--openvino-mo-args',
                            help='Additional args to OpenVINO Model Optimizer.')
        parser.add_argument('--config', default=config['config'], help=argparse.SUPPRESS)

    return parser
