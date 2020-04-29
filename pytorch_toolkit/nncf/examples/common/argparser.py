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

from nncf.config import CustomArgumentParser
from nncf.hw_config import HWConfigType


def get_common_argument_parser():
    """Defines command-line arguments, and parses them.

    """
    parser = CustomArgumentParser()

    parser.add_argument('-c', '--config', help='Path to a config file with task/model-specific parameters',
                        required=True)

    parser.add_argument('--hw-config', help='Type of the hardware configuration for compression algorithms',
                        type=str,
                        default=None,
                        dest="hw_config_type",
                        choices=[t.value for t in HWConfigType])

    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test'],
        default='train',
        help=("train: performs training and validation; test: tests the model"
              "found in \"--save_dir\" with name \"--name\" on the validation split of \"--dataset\"; "))

    parser.add_argument('--metrics-dump', type=str, help='Name of metrics collecting .json file')
    model_init_mode = parser.add_mutually_exclusive_group()
    model_init_mode.add_argument(
        "--resume",
        metavar='PATH',
        type=str,
        default=None,
        dest='resuming_checkpoint',
        help="Specifies the .pth file with the saved model to be tested (for \"-m test\""
             "or to be resumed from (for \"-m train\"). The model architecture should "
             "correspond to what is specified in the config file, and the checkpoint file"
             "must have all necessary optimizer/compression algorithm/metric states required.")
    model_init_mode.add_argument(
        "--weights",
        metavar='PATH',
        type=str,
        default=None,
        help="Attempt to load the model state from the specified .pth file. "
             "This allows to start new compression algorithm from scratch with initializing model by given state")

    parser.add_argument(
        "--checkpoint-save-dir",
        metavar='PATH',
        type=str,
        default=None,
        help="Specifies the directory for the trained model checkpoints to be saved to")

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pretrained models from the model zoo",
        action="store_true",
    )

    execution_type = parser.add_mutually_exclusive_group()
    execution_type.add_argument(
        "--gpu-id",
        type=int,
        metavar='N',
        help="The ID of the GPU training will be performed on, without any parallelization"
    )
    execution_type.add_argument(
        "--multiprocessing-distributed",
        action='store_true',
        help="Specifies that the computations should be parallelized using "
             "PyTorch DistributedDataParallel with training launched "
             "in a separate process for each available GPU. This is the "
             "fastest way to use PyTorch for either single-node or "
             "multi-node data parallel training"
    )
    execution_type.add_argument('--cpu-only', action='store_true',
                                help='Specifies that the computation should be performed'
                                     'using CPU only')

    parser.add_argument('--world-size', default=1, type=int,
                        help='Sets the number of nodes participating in training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8899',
                        help='URL used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='Node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='Distributed backend')

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        metavar='N',
        help="Batch size. Will be split equally between multiple GPUs in the "
             "--multiprocessing-distributed mode."
             "Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='Set starting epoch number manually (useful on restarts)')
    parser.add_argument('--seed', default=None, type=int,
                        help='Specific seed for initializing pseudo-random number'
                             'generators.')

    # Dataset
    parser.add_argument(
        "--data",
        dest="dataset_dir",
        type=str,
        help="Path to the root directory of the selected dataset. ")

    # Settings
    parser.add_argument(
        '-j', '--workers',
        type=int,
        metavar='N',
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4")
    parser.add_argument(
        "--print-step",
        action='store_true',
        help="Print loss every step")
    parser.add_argument(
        "--imshow-batch",
        action='store_true',
        help=("Displays batch images when loading the dataset and making "
              "predictions."))

    # Storage settings
    parser.add_argument(
        "--log-dir",
        type=str,
        default='runs',
        help="The directory where models and TensorboardX summaries"
             " are saved. Default: runs")

    parser.add_argument('--save-freq', default=5, type=int,
                        help='Checkpoint save frequency (epochs). Default: 5')

    parser.add_argument('--to-onnx', type=str, metavar='PATH', default=None,
                        help='Export to ONNX model by given path')

    # Display
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='Print frequency (batch iterations). '
                                          'Default: 10)')

    return parser
