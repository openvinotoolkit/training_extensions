import argparse

import yaml


def train_args_parser(template_path):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    with open(template_path, 'r') as model_definition:
        config = yaml.safe_load(model_definition)

        parser.add_argument('--train_ann_files', required=True,
                            help='Comma-separated paths to training annotation files.')
        parser.add_argument('--train_img_roots', required=True,
                            help='Comma-separated paths to training images folders.')
        parser.add_argument('--val_ann_files', required=True,
                            help='Comma-separated paths to validation annotation files.')
        parser.add_argument('--val_img_roots', required=True,
                            help='Comma-separated paths to validation images folders.')
        parser.add_argument('--resume_from', default='',
                            help='Resume training from previously saved checkpoint')
        parser.add_argument('--load_weights', default='',
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--save_checkpoints_to', default='/tmp/checkpoints',
                            help='Location where checkpoints will be stored')
        parser.add_argument('--epochs', type=int,
                            default=config['training_parameters']['epochs'],
                            help='Number of epochs during training')
        parser.add_argument('--batch_size', type=int,
                            default=config['training_parameters']['batch_size'],
                            help='Size of a single batch during training per GPU.')
        parser.add_argument('--gpu_num', type=int,
                            default=config['training_parameters']['gpu_num'],
                            help='Number of GPUs that will be used in training, 0 is for CPU mode.')
        parser.add_argument('--base_learning_rate', type=float,
                            default=config['training_parameters']['base_learning_rate'],
                            help='Starting value of learning rate that might be changed during '
                                 'training according to learning rate schedule that is usually '
                                 'defined in detailed training configuration.')
        parser.add_argument('--config', default=config['config'],
                            help='Location of a file describing detailed model configuration.')

    return parser


def test_args_parser(template_path):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    with open(template_path, 'r') as model_definition:
        config = yaml.safe_load(model_definition)

        parser.add_argument('--test_ann_files', required=True,
                            help='Comma-separated paths to test annotation files.')
        parser.add_argument('--test_img_roots', required=True,
                            help='Comma-separated paths to test images folders.')
        parser.add_argument('--load_weights', required=True,
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--save_metrics_to', required=True,
                            help='Location where evaluated metrics values will be stored (yaml file).')
        parser.add_argument('--config', default=config['config'],
                            help='Location of a file describing detailed model configuration.')
        parser.add_argument('--save_output_images_to', default='',
                            help='Location where output images (with displayed result of model work) will be stored.')

    return parser


def export_args_parser(template_path):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    with open(template_path, 'r') as model_definition:
        config = yaml.safe_load(model_definition)
        parser.add_argument('--config', default=config['config'],
                            help='Location of a file describing detailed model configuration.')
        parser.add_argument('--load_weights', default='',
                            help='Load only weights from previously saved checkpoint')
        parser.add_argument('--export_format', choices=['openvino', 'onnx'], default='openvino',
                            help='Export format.')
        parser.add_argument('--save_exported_model_to', required='True',
                            help='Location where exported model will be stored.')

    return parser
