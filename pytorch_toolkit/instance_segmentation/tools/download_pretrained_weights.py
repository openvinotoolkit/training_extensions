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
import logging
import os
import os.path as osp
from pydoc import locate
from subprocess import call

import requests
import yaml
from termcolor import colored
from tqdm import tqdm

from segmentoly.utils.logging import setup_logging
from segmentoly.utils.onnx import onnx_export
from segmentoly.utils.weights import load_checkpoint


def download_file_from_web(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    save_response_content(response, destination)


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    URL = 'https://docs.google.com/uc?export=download'
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    response.raise_for_status()
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        response.raise_for_status()
    save_response_content(response, destination)


def save_response_content(response, destination):
    CHUNK_SIZE = 32 * 1024

    file_size = None
    if 'Content-Length' in response.headers:
        file_size = int(response.headers['Content-Length'])

    with open(destination, 'wb') as f, \
            tqdm(desc='Downloading...', total=file_size, unit='B', unit_scale=True,
                 unit_divisor=1024, leave=True) as pbar:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


def mkdir_for_file(file_path):
    dir_path = osp.dirname(file_path)
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_zoo', help='File with models to download and convert.',
                        default=osp.join(osp.dirname(osp.abspath(__file__)), 'model_zoo.yaml'))
    parser.add_argument('--output_dir', help='Directory to save downloaded weights files.',
                        default=os.path.join('data', 'pretrained_models'))
    parser.add_argument('-mo', '--model_optimizer', dest='model_optimizer',
                        help='Model optimizer executable.', default='mo.py')
    return parser.parse_args()


def main(args):
    args.output_dir = osp.abspath(args.output_dir)

    downloaders = dict(web=download_file_from_web,
                       google_drive=download_file_from_google_drive)

    with open(args.model_zoo, 'rt') as f:
        model_zoo = yaml.load(f)['models']

    logging.info('Models to be fetched:')
    for target in model_zoo:
        logging.info('\t{} ({}, {})'.format(colored(target['name'], 'green'),
                                           target['dataset'], target['framework']))
    logging.info('')

    for target in model_zoo:
        output_file = osp.join(args.output_dir, 'raw', target['dst_file'])
        mkdir_for_file(output_file)
        logging.info('Fetching {} ({}, {})'.format(colored(target['name'], 'green'),
                                                  target['dataset'], target['framework']))
        try:
            # Download weights.
            if target['storage_type'] in downloaders:
                downloaders[target['storage_type']](target['url'], output_file)
            else:
                logging.warning('No downloaders available for storage {}'.format(target['storage_type']))
                continue

            # Convert weights.
            logging.info('Downloaded to {}'.format(output_file))
            if target.get('weights_converter', None):
                logging.info('Converting weights...')
                converter = locate(target['weights_converter'])
                if converter is None:
                    logging.warning('Invalid weights converter {}'.format(target['weights_converter']))
                    continue
                output_converted_file = osp.join(args.output_dir, 'converted', target['dst_file'])
                output_converted_file = osp.splitext(output_converted_file)[0] + '.pth'
                mkdir_for_file(output_converted_file)
                try:
                    converter(output_file, output_converted_file)
                    logging.info('Converted weights file saved to {}'.format(output_converted_file))
                except Exception as ex:
                    logging.warning('Failed to convert weights.')
                    logging.warning(ex)
                    continue

                if target.get('convert_to_ir', False):
                    # Convert to ONNX.
                    logging.info('Exporting to ONNX...')
                    output_onnx_file = osp.join(args.output_dir, 'onnx', target['dst_file'])
                    output_onnx_file = osp.splitext(output_onnx_file)[0] + '.onnx'
                    mkdir_for_file(output_onnx_file)
                    net = locate(target['model'])(81, fc_detection_head=False)
                    load_checkpoint(net, output_converted_file, verbose=False)
                    onnx_export(net, target['input_size'], output_onnx_file)
                    logging.info('ONNX file is saved to {}'.format(output_onnx_file))

                    # Convert to IR.
                    logging.info('Converting to IR...')
                    output_ir_dir = osp.join(args.output_dir, 'ir', target['dst_file'])
                    mkdir_for_file(output_ir_dir)
                    output_ir_dir = osp.dirname(output_ir_dir)
                    status = call([args.model_optimizer,
                                   '--framework', 'onnx',
                                   '--input_model', output_onnx_file,
                                   '--output_dir', output_ir_dir,
                                   '--input', 'im_data,im_info',
                                   '--output', 'boxes,scores,classes,batch_ids,raw_masks',
                                   '--mean_values',
                                   'im_data{},im_info[0,0,0]'.format(str(target['mean_pixel']).replace(' ', ''))
                                   ])
                    if status:
                        logging.warning('Failed to convert model to IR.')
                    else:
                        logging.info('IR files saved to {}'.format(output_ir_dir))

        except Exception as ex:
            logging.warning(repr(ex))


if __name__ == '__main__':
    setup_logging()
    args = parse_args()
    main(args)
