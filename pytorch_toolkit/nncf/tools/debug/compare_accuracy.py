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

import os
from functools import partial
from openvino.inference_engine import IENetwork, IEPlugin, get_version

from nncf.config import Config
from nncf.dynamic_graph.graph_builder import create_input_infos
from tools.ir_utils import get_ir_paths


def getExecNet(plugin, net):
    return plugin.load(network=net)


argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--model", help="input IR name", required=True)
argparser.add_argument("--bin", help="Input *.bin file name")
argparser.add_argument("-o", "--output-dir", help="Output directory to dump weights", required=True)
argparser.add_argument("-c", "--config", type=str, help="Model's config", required=True)
argparser.add_argument("--cuda", help="inference PyTorch model on CUDA", action='store_true')
argparser.add_argument('--data', metavar='DIR', help='path to dataset', required=True)
argparser.add_argument('--cpu-plugin-dir', metavar='DIR',
                       help='path to the directory with CPU Plugin and CPU Extension libraries', required=True)
argparser.add_argument("-n", "--num-layers", type=int, default=-1, help="Dump activations for given number of layers")
argparser.add_argument("--dump", action='store_true', help="Enables dump of activations")

args = argparser.parse_args()


def validate_torch_model(output_dir, config, num_layers, dump, val_loader=None, cuda=False):
    from tools.debug.common import load_torch_model, register_print_hooks

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_torch_model(config, cuda)

    model_e = model.eval()
    if dump:
        register_print_hooks(output_dir, model_e, num_layers=num_layers, data_to_compare=None, dump_activations=True)

    validate_general(val_loader, model_e, infer_pytorch_model, cuda)


def main():
    model_bin, model_xml = get_ir_paths(args.model, args.bin)

    config = Config.from_json(args.config)

    input_infos_list = create_input_infos(config)
    image_size = input_infos_list[0].shape[-1]

    size = int(image_size / 0.875)

    print('IE version: {}'.format(get_version()))

    # NOTE: importing torch after loading IE to plugin to avoid issue with built-in MKLDNN of PyTorch
    plugin = IEPlugin(device='CPU',
                      plugin_dirs=args.cpu_plugin_dir)
    plugin.add_cpu_extension(os.path.join(args.cpu_plugin_dir, "libcpu_extension.so"))
    net = IENetwork(model=model_xml, weights=model_bin)
    exec_net = getExecNet(plugin, net)
    from torch.utils.data import DataLoader
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    val_loader = DataLoader(
        datasets.ImageFolder(args.data, transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    config['log_dir'] = args.output_dir

    infer_fn = partial(infer_ie_model, net=net)
    validate_general(val_loader, exec_net, infer_fn)

    validate_torch_model(os.path.join(args.output_dir, "PTH"), config=config, num_layers=args.num_layers,
                         dump=args.dump, val_loader=val_loader, cuda=args.cuda)


def infer_ie_model(exec_net, inputs, net):
    input_cpu = inputs.numpy()
    input_name = next(iter(net.inputs))
    output_name = next(iter(net.outputs))
    res = exec_net.infer(inputs={input_name: input_cpu})
    output = res[output_name]
    import torch
    torch_output = torch.from_numpy(output)
    return torch_output


def infer_pytorch_model(model, inputs):
    return model(inputs)


def validate_general(val_loader, model, infer_model_fn, cuda=False):
    from examples.classification.main import AverageMeter, accuracy
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (input_, target) in enumerate(val_loader):
        # compute output
        output = infer_model_fn(model, input_)

        if cuda:
            target = target.cuda(None, non_blocking=True)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1, input_.size(0))
        top5.update(acc5, input_.size(0))

        if i % 10 == 0:
            print('IE Test : [{0}/{1}]\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
