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
from pydoc import locate

from tqdm import tqdm

from segmentoly.data.dataparallel import collate, ShallowDataParallel
from segmentoly.data.transforms import *
from segmentoly.datasets.factory import get_dataset
from segmentoly.rcnn.openvino_net import MaskRCNNOpenVINO
from segmentoly.utils.logging import setup_logging
from segmentoly.utils.postprocess import postprocess_batch
from segmentoly.utils.stats import add_flops_counting_methods, flops_to_string, print_model_with_flops
from segmentoly.utils.weights import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate instance segmentation')
    subparsers = parser.add_subparsers(help='Backend', dest='backend')

    parser.add_argument('--dataset', help='Dataset name.', metavar='"<name>"',)
    parser.add_argument('--ckpt', dest='checkpoint_file_path', required=True, metavar='"<path>"',
                        help='File with models weights.')
    parser.add_argument('--nw', dest='num_workers', default=8, type=int, metavar='<num>',
                        help='Number of data loading workers.')
    parser.add_argument('--mean_pixel', default=(0.0, 0.0, 0.0), type=float, nargs=3,
                        metavar='<num>',
                        help='Mean pixel value to subtract from image.')
    image_resize_group = parser.add_mutually_exclusive_group(required=True)
    image_resize_group.add_argument('--fit_max', dest='fit_max_image_size', default=None, type=int, nargs=2,
                                    metavar='<num>',
                                    help='Max processed image size in a format '
                                         '(max short side, max long side).')
    image_resize_group.add_argument('--fit_window', dest='fit_window_size', default=None, type=int, nargs=2,
                                    metavar='<num>',
                                    help='Max processed image size in a format (max height, max width).')

    openvino_parser = subparsers.add_parser('openvino')
    openvino_parser.add_argument('--model', dest='openvino_model_path', type=str, required=True, metavar='"<path>"',
                                 help='XML file with model description (OpenVINO format)')
    openvino_parser.add_argument('-d', '--device',
                                 help='Optional. Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD. '
                                      'The demo will look for a suitable plugin for device specified '
                                      '(by default, it is CPU).',
                                 default='CPU', type=str, metavar='"<device>"')
    openvino_parser.add_argument('-l', '--cpu_extension',
                                 help='Required for CPU custom layers. '
                                      'Absolute path to a shared library with the kernels implementation.',
                                 default=None, type=str, metavar='"<absolute_path>"')
    openvino_parser.add_argument('-pp', '--plugin_dir',
                                 help='Optional. Path to a plugin folder.',
                                 default=None, type=str, metavar='"<absolute_path>"')
    openvino_parser.add_argument('--show_pc', '--show_performance_counters', dest='show_performance_counters',
                                 help='Show OpenVINO performance counters.',
                                 action='store_true')

    pytorch_parser = subparsers.add_parser('pytorch')
    pytorch_parser.add_argument('--model', dest='pytorch_model_class', type=str, required=True, metavar='"<path>"',
                                help='Path to module and class implementing Mask R-CNN instance segmentation model.')
    pytorch_parser.add_argument('--batch_size', default=1, type=int, metavar='<num>',
                                help='Batch size for all GPUs.')
    pytorch_parser.add_argument('--show_flops',
                                help='Show FLOPs.',
                                action='store_true')
    pytorch_parser.add_argument('--show_layers_flops',
                                help='Show FLOPs for all modules.',
                                action='store_true')

    return parser.parse_args()


def main(args):
    transforms = Compose(
        [
            Resize(max_size=args.fit_max_image_size, window_size=args.fit_window_size),
            ToTensor(),
            Normalize(mean=args.mean_pixel, std=[1., 1., 1.], rgb=False),
        ]
    )
    dataset = get_dataset(args.dataset, False, False, transforms)
    logging.info(dataset)
    num_workers = args.num_workers

    logging.info('Using {} backend'.format(args.backend))

    logging.info('Loading network...')
    if args.backend == 'pytorch':
        net = locate(args.pytorch_model_class)(dataset.classes_num)
        net.eval()
        load_checkpoint(net, args.checkpoint_file_path)
        net = add_flops_counting_methods(net)
        net.reset_flops_count()
        net.start_flops_count()
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            net = net.cuda()
            net = ShallowDataParallel(net)
        batch_size = args.batch_size
    elif args.backend == 'openvino':
        net = MaskRCNNOpenVINO(args.openvino_model_path, args.checkpoint_file_path,
                               device=args.device, plugin_dir=args.plugin_dir,
                               cpu_extension_lib_path=args.cpu_extension,
                               collect_perf_counters=args.show_performance_counters)
        batch_size = 1
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
    for data_batch in tqdm(iter(data_loader)):
        batch_meta = data_batch['meta']
        actual_batch_size = len(batch_meta)
        with torch.no_grad():
            boxes, classes, scores, batch_ids, masks = net(**data_batch)

        im_heights = [meta['original_size'][0] for meta in batch_meta]
        im_widths = [meta['original_size'][1] for meta in batch_meta]
        im_scales = [meta['processed_size'][0] / meta['original_size'][0] for meta in batch_meta]
        scores, classes, boxes, masks = postprocess_batch(batch_ids, scores, classes, boxes, masks, actual_batch_size,
                                                          im_h=im_heights,
                                                          im_w=im_widths,
                                                          im_scale=im_scales,
                                                          full_image_masks=True, encode_masks=True)
        boxes_all.extend(boxes)
        masks_all.extend(masks)
        classes_all.extend(classes)
        scores_all.extend(scores)

    try:
        del data_loader
    except ConnectionResetError:
        pass

    logging.info('Evaluating results...')
    evaluation_results = dataset.evaluate(scores_all, classes_all, boxes_all, masks_all)
    logging.info(evaluation_results)

    if args.backend == 'pytorch':
        if torch.cuda.is_available():
            net = net.module
        net.stop_flops_count()
        if args.show_flops:
            logging.info('Average FLOPs:  {}'.format(flops_to_string(net.compute_average_flops_cost())))
        if args.show_layers_flops:
            logging.info('Thorough computational complexity statistics:')
            print_model_with_flops(net)
    else:
        if args.show_performance_counters:
            net.print_performance_counters()

    del net


if __name__ == '__main__':
    setup_logging()
    args = parse_args()
    main(args)
