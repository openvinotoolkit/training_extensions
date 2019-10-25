import argparse

import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from models.single_person_pose_with_mobilenet import SinglePersonPoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(net, output_name, single_person):
    input = torch.randn(1, 3, 256, 456)
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']
    if single_person:
        input = torch.randn(1, 3, 384, 288)
        output_names = 'output_data'
        output_names = ['stage_0_output_1_heatmaps',
                       # 'stage_1_output_1_heatmaps',
                       # 'stage_2_output_1_heatmaps',
                       # 'stage_3_output_1_heatmaps',
                       # 'stage_4_output_1_heatmaps',
                       ]

    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='single-human-pose-estimation1.onnx',
                        help='name of output model in ONNX format')
    parser.add_argument('--single-person', action='store_true', help='convert model for single-person pose estimation')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    if args.single_person:
        net = SinglePersonPoseEstimationWithMobileNet(num_refinement_stages=1)
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    convert_to_onnx(net, args.output_name, args.single_person)
