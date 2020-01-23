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

import argparse
import json
import logging

from tqdm import tqdm

import torch
from segmentoly.data.dataparallel import collate, ShallowDataParallel
from segmentoly.data.transforms import Resize, Normalize, ToTensor, Compose
from segmentoly.utils.logging import setup_logging
from segmentoly.utils.profile import Timer
from segmentoly.utils.stats import add_flops_counting_methods, flops_to_string, \
    print_model_with_flops
from segmentoly.utils.weights import load_checkpoint
from text_spotting.data.alphabet import AlphabetDecoder
from text_spotting.datasets.factory import get_dataset
from text_spotting.models.openvino_net import TextMaskRCNNOpenVINO
from text_spotting.models.text_detectors import make_text_detector
from text_spotting.utils.postprocess import postprocess_batch


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(description='Evaluate instance segmentation')
    subparsers = parser.add_subparsers(help='Backend', dest='backend')

    parser.add_argument('--dataset', help='Dataset name.', metavar='"<name>"', )
    parser.add_argument('--nw', dest='num_workers', default=8, type=int, metavar='<num>',
                        help='Number of data loading workers.')
    parser.add_argument('-pt', '--prob_threshold', default=0.8, type=float,
                        help='Probability threshold for detections filtering.')
    parser.add_argument('--mean_pixel', default=(0.0, 0.0, 0.0), type=float, nargs=3,
                        metavar='<num>',
                        help='Mean pixel value to subtract from image.')
    parser.add_argument('--std_pixel', default=(1.0, 1.0, 1.0), type=float, nargs=3,
                        metavar='<num>',
                        help='STD pixel value to divide an image.')
    parser.add_argument('--rgb', action='store_true',
                        help='Use RGB instead of BGR.')
    parser.add_argument('--size', dest='size', default=None, type=int, nargs=2,
                        metavar='<num>',
                        help='Input resolution in a (height, width) format.')
    parser.add_argument('--visualize', action='store_true', help='Visualize text spotting results.')

    openvino_parser = subparsers.add_parser('openvino')
    openvino_parser.add_argument('--detector_model', dest='openvino_detector_model_path', type=str,
                                 required=True, metavar='"<path>"',
                                 help='XML file with model description for text detection part '
                                      '(OpenVINO format)')
    openvino_parser.add_argument('--encoder_model', dest='openvino_encoder_model_path', type=str,
                                 required=True, metavar='"<path>"',
                                 help='XML file with model description for text detection part '
                                      '(OpenVINO format)')
    openvino_parser.add_argument('--decoder_model', dest='openvino_decoder_model_path', type=str,
                                 required=True, metavar='"<path>"',
                                 help='XML file with model description for text detection part '
                                      '(OpenVINO format)')
    openvino_parser.add_argument('--show_pc', '--show_performance_counters',
                                 dest='show_performance_counters',
                                 help='Show OpenVINO performance counters.',
                                 action='store_true')

    pytorch_parser = subparsers.add_parser('pytorch')
    pytorch_parser.add_argument('--model', dest='pytorch_model_class', type=str, required=True,
                                help='Path to module and class implementing Mask R-CNN instance '
                                     'segmentation model.')
    pytorch_parser.add_argument('--weights', dest='checkpoint_file_path', required=True,
                                help='File with models weights')
    pytorch_parser.add_argument('--show_flops',
                                help='Show FLOPs.',
                                action='store_true')
    pytorch_parser.add_argument('--show_layers_flops',
                                help='Show FLOPs for all modules.',
                                action='store_true')

    return parser.parse_args()


def main(args):
    """ Tests text spotter. """

    transforms = Compose(
        [
            Resize(size=args.size),
            ToTensor(),
            Normalize(mean=args.mean_pixel, std=args.std_pixel, rgb=args.rgb),
        ]
    )
    dataset = get_dataset(args.dataset, False, False, transforms,
                          alphabet_decoder=AlphabetDecoder())
    logging.info(dataset)
    num_workers = args.num_workers

    inference_timer = Timer()

    logging.info('Using {} backend'.format(args.backend))

    logging.info('Loading network...')
    batch_size = 1
    if args.backend == 'pytorch':
        with open(args.pytorch_model_class) as file:
            config = json.load(file)
        net = make_text_detector(**config['model'])(dataset.classes_num,
                                                    force_max_output_size=False, shape=args.size)
        net.eval()
        load_checkpoint(net, args.checkpoint_file_path)
        if torch.cuda.is_available():
            net = net.cuda()
        net = add_flops_counting_methods(net)
        net.reset_flops_count()
        net.start_flops_count()
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            net = net.cuda()
            net = ShallowDataParallel(net)
    elif args.backend == 'openvino':
        net = TextMaskRCNNOpenVINO(args.openvino_detector_model_path,
                                   args.openvino_encoder_model_path,
                                   args.openvino_decoder_model_path,
                                   collect_perf_counters=args.show_performance_counters)
    else:
        raise ValueError('Unknown backend "{}"'.format(args.backend))

    logging.info('Using batch size {}'.format(batch_size))
    logging.info('Number of prefetching processes {}'.format(num_workers))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate
    )

    logging.info('Processing dataset...')
    boxes_all = []
    masks_all = []
    classes_all = []
    scores_all = []
    text_probs_all = []
    for data_batch in tqdm(iter(data_loader)):
        batch_meta = data_batch['meta']
        actual_batch_size = len(batch_meta)
        with torch.no_grad(), inference_timer:
            boxes, classes, scores, batch_ids, masks, text_probs = net(**data_batch)

        im_heights = [meta['original_size'][0] for meta in batch_meta]
        im_widths = [meta['original_size'][1] for meta in batch_meta]
        im_scale_y = [meta['processed_size'][0] / meta['original_size'][0] for meta in batch_meta]
        im_scale_x = [meta['processed_size'][1] / meta['original_size'][1] for meta in batch_meta]
        scores, classes, boxes, masks, text_probs = postprocess_batch(
            batch_ids, scores, classes, boxes, masks, text_probs, actual_batch_size,
            im_h=im_heights,
            im_w=im_widths,
            im_scale_y=im_scale_y,
            im_scale_x=im_scale_x,
            full_image_masks=True, encode_masks=True,
            confidence_threshold=args.prob_threshold)

        boxes_all.extend(boxes)
        masks_all.extend(masks)
        classes_all.extend(classes)
        scores_all.extend(scores)
        text_probs_all.extend(text_probs)

    try:
        del data_loader
    except ConnectionResetError:
        pass

    logging.info('Evaluating results...')
    evaluation_results = dataset.evaluate(scores_all, classes_all, boxes_all, masks_all,
                                          text_probs_all, dump='dump', visualize=args.visualize)
    logging.info(evaluation_results)

    logging.info('Average inference time {}'.format(inference_timer.average_time))

    if args.backend == 'pytorch':
        if torch.cuda.is_available():
            net = net.module
        net.stop_flops_count()
        if args.show_flops:
            logging.info(
                'Average FLOPs:  {}'.format(flops_to_string(net.compute_average_flops_cost())))
        if args.show_layers_flops:
            logging.info('Thorough computational complexity statistics:')
            print_model_with_flops(net)
    else:
        if args.show_performance_counters:
            net.print_performance_counters()

    del net


if __name__ == '__main__':
    setup_logging()
    main(parse_args())
