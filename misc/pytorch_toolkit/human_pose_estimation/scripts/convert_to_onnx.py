import argparse

import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(net, output_name, single_person, input_size):
    input = torch.randn(1, 3, input_size[0], input_size[1])
    input_layer_names = ['data']
    output_layer_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                          'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']
    if single_person:
        input = torch.randn(1, 3, input_size[0], input_size[1])
        output_layer_names = ['stage_{}_output_1_heatmaps'.format(i) for i in range(len(net.refinement_stages) + 1)]

    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_layer_names,
                      output_names=output_layer_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation.onnx',
                        help='name of output model in ONNX format')
    parser.add_argument('--single-person', action='store_true', help='convert model for single-person pose estimation')
    parser.add_argument('--input-size', nargs='+', type=int,  required=True,
                        help='Size of input image in format: height width')
    parser.add_argument('--mode-interpolation', type=str, required=False, default='bilinear',
                        help='type interpolation <bilinear> or <nearest>')
    parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')

    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    if args.single_person:
        net = SinglePersonPoseEstimationWithMobileNet(mode=args.mode_interpolation, num_refinement_stages=args.num_refinement_stages)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    convert_to_onnx(net, args.output_name, args.single_person, args.input_size)
