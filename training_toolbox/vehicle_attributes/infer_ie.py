from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import logging as log
import numpy as np
import cv2

from openvino.inference_engine import IENetwork, IEPlugin

def normalized_to_absolute(prediction):
  colorcar = np.zeros((1, 1, 3), dtype=np.uint8)
  for i in range(3):
    if prediction[i] < 0:
      colorcar[0, 0, i] = 0
    elif prediction[i] > 1:
      colorcar[0, 0, i] = 255
    else:
      colorcar[0, 0, i] = prediction[i]*255
  return colorcar

def annotation_to_type2(restype):
  if restype == 0:
    vehtype = 'car'
  elif restype == 1:
    vehtype = 'truck'
  elif restype == 2:
    vehtype = 'van'
  elif restype == 3:
    vehtype = 'bus'
  else:
    vehtype = 'undefined'
  return vehtype

def build_argparser():
  parser = ArgumentParser()
  parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
  parser.add_argument("-l", "--cpu_extension",
                      help="MKLDNN (CPU)-targeted custom layers. \
                        Absolute path to a shared library with the kernels implementation",
                      type=str, default=None)
  parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
  parser.add_argument("-d", "--device",
                      help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                           "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                      type=str)
  parser.add_argument('input_image', help='Image with a vehicle')
  return parser

# pylint: disable=too-many-locals
def main():
  log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
  args = build_argparser().parse_args()

  model_xml = args.model
  model_bin = os.path.splitext(model_xml)[0] + ".bin"
  log.info("Initializing plugin for %s device...", args.device)
  plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
  if args.cpu_extension and 'CPU' in args.device:
    plugin.add_cpu_extension(args.cpu_extension)
  # Read IR
  log.info("Reading IR...")
  net = IENetwork.from_ir(model=model_xml, weights=model_bin)
  if "CPU" in plugin.device:
    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if not_supported_layers:
      log.error("Following layers are not supported by the plugin for specified device %s:\n %s",
                plugin.device, ', '.join(not_supported_layers))
      log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                "or --cpu_extension command line argument")
      sys.exit(1)
  assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
  assert len(net.outputs) == 2, "Sample supports two output topologies"
  input_blob = next(iter(net.inputs))
  log.info("Loading IR to the plugin...")
  exec_net = plugin.load(network=net, num_requests=2)
  # Read and pre-process input image
  _, _, height, width = net.inputs[input_blob].shape
  del net
  frame = cv2.imread(args.input_image)
  in_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
  in_frame = cv2.resize(in_frame, (width, height), interpolation=cv2.INTER_CUBIC)
  in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW

  res = exec_net.infer({input_blob: in_frame})
  vtype = annotation_to_type2(np.argmax(res['resnet_v1_10/type'][0]))

  colorcar = normalized_to_absolute(res['resnet_v1_10/color'][0])
  rgb_color = cv2.cvtColor(colorcar, cv2.COLOR_LAB2BGR)[0, 0].tolist()

  img = cv2.imread(args.input_image)
  cv2.rectangle(img, (0, 0), (30, 30), rgb_color, -1)
  font = cv2.FONT_HERSHEY_PLAIN
  cv2.putText(img, vtype, (0, 15), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
  cv2.imshow('Vehicle_attributes', img)
  _ = cv2.waitKey(0)
  del exec_net
  del plugin
if __name__ == '__main__':
  sys.exit(main() or 0)
