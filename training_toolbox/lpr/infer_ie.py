#!/usr/bin/env python

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
from trainer import inference, LPRVocab, encode, decode_ie_output
from utils.helpers import load_module


def build_argparser():
  parser = ArgumentParser()
  parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
  parser.add_argument("-l", "--cpu_extension",
                      help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels implementation",
                      type=str, default=None)
  parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
  parser.add_argument("-d", "--device",
                      help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                           "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                      type=str)
  parser.add_argument('path_to_config', help='Path to a config.py')
  parser.add_argument('input_image', help='Image with license plate')
  return parser

def display_license_plate(number, license_plate_img):
  size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
  text_width = size[0][0]
  text_height = size[0][1]

  h_, w_, _ = license_plate_img.shape
  license_plate_img = cv2.copyMakeBorder(license_plate_img, 0, text_height + 10, 0, 0 if text_width < w_ else text_width - w_,
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
  cv2.putText(license_plate_img, number, (0, h_ + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

  return license_plate_img

def main():
  log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
  args = build_argparser().parse_args()
  cfg = load_module(args.path_to_config)
  vocab, r_vocab, num_classes = LPRVocab.create_vocab(cfg.train.train_list_file_path, cfg.eval.file_list_path)
  model_xml = args.model
  model_bin = os.path.splitext(model_xml)[0] + ".bin"
  log.info("Initializing plugin for {} device...".format(args.device))
  plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
  if args.cpu_extension and 'CPU' in args.device:
    plugin.add_cpu_extension(args.cpu_extension)

  # Read IR
  log.info("Reading IR...")
  net = IENetwork.from_ir(model=model_xml, weights=model_bin)

  if "CPU" in plugin.device:
    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
      log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                format(plugin.device, ', '.join(not_supported_layers)))
      log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                "or --cpu_extension command line argument")
      sys.exit(1)
  assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
  assert len(net.outputs) == 1, "Sample supports only single output topologies"
  input_blob = next(iter(net.inputs))
  out_blob = next(iter(net.outputs))
  log.info("Loading IR to the plugin...")
  exec_net = plugin.load(network=net, num_requests=2)
  # Read and pre-process input image
  n, c, h, w = net.inputs[input_blob].shape
  del net

  cur_request_id = 0
  while 1:
    frame = cv2.imread(args.input_image)
    img_to_display = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    in_frame = cv2.resize(frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))

    exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
    if exec_net.requests[cur_request_id].wait(-1) == 0:

      # Parse detection results of the current request
      lp = exec_net.requests[cur_request_id].outputs[out_blob]
      lp_number = decode_ie_output(lp, r_vocab)
      img_to_display = display_license_plate(lp_number, img_to_display)
      cv2.imshow('License Plate', img_to_display)
      key = cv2.waitKey(0)
      if key == 27:
        break

  del exec_net
  del plugin


if __name__ == '__main__':
  sys.exit(main() or 0)