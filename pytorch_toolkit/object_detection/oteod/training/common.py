import logging
import sys

import torch
import yaml
from oteod import MMDETECTION_TOOLS
from oteod.misc import run_with_termination


def train_internal(config, gpu_num, update_config):
    training_info = {'training_gpu_num': 0}
    if torch.cuda.is_available():
        logging.info('Training on GPUs started ...')
        available_gpu_num = torch.cuda.device_count()
        if available_gpu_num < gpu_num:
            logging.warning(f'available_gpu_num < args.gpu_num: {available_gpu_num} < {gpu_num}')
            logging.warning(f'decreased number of gpu to: {available_gpu_num}')
            gpu_num = available_gpu_num
            sys.stdout.flush()
        run_with_termination(f'{MMDETECTION_TOOLS}/dist_train.sh'
                             f' {config}'
                             f' {gpu_num}'
                             f'{update_config}'.split(' '))
        training_info['training_gpu_num'] = gpu_num
        logging.info('... training on GPUs completed.')
    else:
        logging.info('Training on CPU started ...')
        run_with_termination(f'python {MMDETECTION_TOOLS}/train.py'
                             f' {config}'
                             f'{update_config}'.split(' '))
        logging.info('... training on CPU completed.')

    return training_info


def train(config, gpu_num, out, update_config):
    """ Main function. """

    logging.basicConfig(level=logging.INFO)

    logging.info(f'Commandline:\n{" ".join(sys.argv)}')

    update_config = f' --update_config {update_config}' if update_config else ''

    logging.info('Training started ...')
    training_info = train_internal(config, gpu_num, update_config)
    logging.info('... training completed.')

    with open(out, 'a+') as dst_file:
        yaml.dump(training_info, dst_file)
