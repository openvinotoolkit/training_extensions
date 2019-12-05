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
import sys

import cv2
from tqdm import tqdm

import torch
from segmentoly.data.dataparallel import collate
from segmentoly.data.transforms import Resize, Normalize, ToTensor, Compose
from segmentoly.datasets.images import ImagesDataset
from segmentoly.datasets.video import VideoDataset
from segmentoly.utils.logging import setup_logging
from segmentoly.utils.profile import Timer
from segmentoly.utils.stats import add_flops_counting_methods, flops_to_string, \
    print_model_with_flops
from segmentoly.utils.tracker import StaticIOUTracker
from segmentoly.utils.weights import load_checkpoint
from text_spotting.models.openvino_net import TextMaskRCNNOpenVINO
from text_spotting.models.text_detectors import make_text_detector
from text_spotting.utils.postprocess import postprocess
from text_spotting.utils.visualizer import TextVisualizer


def parse_args():
    """ Parses input arguments. """

    parser = argparse.ArgumentParser(description='Run instance segmentation live')
    subparsers = parser.add_subparsers(help='Backend', dest='backend')

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

    data_source_parser = parser.add_mutually_exclusive_group(required=True)
    data_source_parser.add_argument('-v', '--video', default=None, type=str,
                                    help='Path to a video file or numeric camera ID.')
    data_source_parser.add_argument('-i', '--images', default=None, type=str,
                                    help='Path to an image or a folder with images.')

    parser.add_argument('--show_scores', action='store_true',
                        help='Show detection scores.')
    parser.add_argument('--show_boxes', action='store_true',
                        help='Show bounding boxes.')
    parser.add_argument('--show_fps', action='store_true',
                        help='Show FPS.')
    parser.add_argument('--delay', default=0, type=int)

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
    """ Demonstrates text spotter. """

    transforms = Compose(
        [
            Resize(size=args.size),
            ToTensor(),
            Normalize(mean=args.mean_pixel, std=args.std_pixel, rgb=args.rgb),
        ]
    )

    classes = ['', 'text']

    logging.info('Using {} backend'.format(args.backend))

    logging.info('Loading network...')
    batch_size = 1
    if args.backend == 'pytorch':
        with open(args.pytorch_model_class) as file:
            config = json.load(file)
        net = make_text_detector(**config['model'])(len(classes),
                                                    force_max_output_size=False, shape=args.size)
        net.eval()
        load_checkpoint(net, args.checkpoint_file_path)
        if torch.cuda.is_available():
            net = net.cuda()
        net = add_flops_counting_methods(net)
        net.reset_flops_count()
        net.start_flops_count()
    elif args.backend == 'openvino':
        net = TextMaskRCNNOpenVINO(args.openvino_detector_model_path,
                                   args.openvino_encoder_model_path,
                                   args.openvino_decoder_model_path,
                                   collect_perf_counters=args.show_performance_counters)
    else:
        raise ValueError('Unknown backend "{}"'.format(args.backend))

    viz = TextVisualizer(0.9, classes, confidence_threshold=args.prob_threshold,
                         show_boxes=args.show_boxes, show_scores=args.show_scores)

    inference_timer = Timer(cuda_sync=True, warmup=1)
    timer = Timer(cuda_sync=False, warmup=1)
    timer.tic()

    logging.info('Configuring data source...')
    if args.video:
        try:
            args.video = int(args.video)
        except ValueError:
            pass
        demo_dataset = VideoDataset(args.video, labels=classes, transforms=transforms)
        num_workers = 0
        tracker = StaticIOUTracker()
    else:
        demo_dataset = ImagesDataset(args.images, labels=classes, transforms=transforms)
        num_workers = 1
        tracker = None

    data_loader = torch.utils.data.DataLoader(
        demo_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate
    )

    logging.info('Processing data...')
    frames_num = len(demo_dataset)
    for data_batch in tqdm(iter(data_loader), total=frames_num if frames_num != sys.maxsize else 0):
        im_data = data_batch['im_data']
        im_info = data_batch['im_info']
        if torch.cuda.is_available():
            im_data = [i.cuda() for i in im_data]
            im_info = [i.cuda() for i in im_info]
        with torch.no_grad(), inference_timer:
            boxes, classes, scores, _, masks, text_probs = net(im_data, im_info)

        meta = data_batch['meta'][0]
        scores, classes, boxes, masks, text_probs = postprocess(
            scores, classes, boxes, masks, text_probs,
            im_h=meta['original_size'][0],
            im_w=meta['original_size'][1],
            im_scale_y=meta['processed_size'][0] / meta['original_size'][0],
            im_scale_x=meta['processed_size'][1] / meta['original_size'][1],
            full_image_masks=True,
            encode_masks=False,
            confidence_threshold=args.prob_threshold)

        masks_ids = tracker(masks, classes) if tracker is not None else None
        image = data_batch['original_image'][0]
        visualization = viz(image, boxes, classes, scores, segms=masks, ids=masks_ids,
                            text_log_softmax=text_probs)
        fps = 1 / timer.toc()
        if args.show_fps:
            visualization = cv2.putText(visualization, 'FPS: {:>2.2f}'.format(fps),
                                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('result', visualization)
        key = cv2.waitKey(args.delay)
        if key == 27:
            break
        timer.tic()

    if inference_timer.average_time > 0:
        logging.info('Average inference FPS: {:3.2f}'.format(1 / inference_timer.average_time))

    if args.backend == 'pytorch':
        net.stop_flops_count()
        if args.show_flops:
            logging.info(
                'Average FLOPs:  {}'.format(flops_to_string(net.compute_average_flops_cost())))
        if args.show_layers_flops:
            logging.info('Thorough computational complexity statistics:')
            print_model_with_flops(net)
        if torch.cuda.is_available():
            logging.info('GPU memory footprint:')
            logging.info(
                '\tMax allocated: {:.2f} MiB'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
            logging.info(
                '\tMax cached:    {:.2f} MiB'.format(torch.cuda.max_memory_cached() / 1024 ** 2))
    else:
        if args.show_performance_counters:
            net.print_performance_counters()

    cv2.destroyAllWindows()
    del net


if __name__ == '__main__':
    setup_logging()
    main(parse_args())
