from __future__ import print_function
import logging as log
import sys
import os
from argparse import ArgumentParser
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
from lpr.trainer import decode_ie_output
from utils.helpers import load_module



def build_argparser():
  parser = ArgumentParser()
  parser.add_argument("--model", help="Path to an .xml file with a trained model.", required=True, type=str)
  parser.add_argument("--cpu_extension",
                      help="MKLDNN (CPU)-targeted custom layers. "
                           "Absolute path to a shared library with the kernels implementation", type=str, default=None)
  parser.add_argument("--plugin_dir", help="Path to a plugin folder", type=str, default=None)
  parser.add_argument("--device",
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

  height, width, _ = license_plate_img.shape
  license_plate_img = cv2.copyMakeBorder(license_plate_img, 0, text_height + 10, 0,
                                         0 if text_width < width else text_width - width,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
  cv2.putText(license_plate_img, number, (0, height + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

  return license_plate_img

def load_ir_model(model_xml, device, plugin_dir, cpu_extension):
  model_bin = os.path.splitext(model_xml)[0] + ".bin"

  # initialize plugin
  log.info("Initializing plugin for %s device...", device)
  plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
  if cpu_extension and 'CPU' in device:
    plugin.add_cpu_extension(cpu_extension)

  # read IR
  log.info("Reading IR...")
  net = IENetwork.from_ir(model=model_xml, weights=model_bin)

  if "CPU" in plugin.device:
    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if not_supported_layers:
      log.error("Following layers are not supported by the plugin for specified device %s:\n %s",
                plugin.device, ', '.join(not_supported_layers))
      log.error("Please try to specify cpu extensions library path in sample's command line parameters using "
                "--cpu_extension command line argument")
      sys.exit(1)

  # input / output check
  assert len(net.inputs.keys()) == 1, "LPRNet must have only single input"
  assert len(net.outputs) == 1, "LPRNet must have only single output topologies"
  input_blob = next(iter(net.inputs))
  out_blob = next(iter(net.outputs))
  log.info("Loading IR to the plugin...")
  exec_net = plugin.load(network=net, num_requests=2)
  shape = net.inputs[input_blob].shape
  del net

  return exec_net, plugin, input_blob, out_blob, shape


# pylint: disable=too-many-locals
def main():
  log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
  args = build_argparser().parse_args()
  cfg = load_module(args.path_to_config)
  exec_net, plugin, input_blob, out_blob, shape = load_ir_model(args.model, args.device,
                                                                args.plugin_dir, args.cpu_extension)
  n_batch, channels, height, width = shape


  cur_request_id = 0
  while 1:
    frame = cv2.imread(args.input_image)
    img_to_display = frame.copy()
    in_frame = cv2.resize(frame, (width, height))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n_batch, channels, height, width))

    exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
    if exec_net.requests[cur_request_id].wait(-1) == 0:

      # Parse detection results of the current request
      lp_code = exec_net.requests[cur_request_id].outputs[out_blob]
      lp_number = decode_ie_output(lp_code, cfg.r_vocab)
      img_to_display = display_license_plate(lp_number, img_to_display)
      cv2.imshow('License Plate', img_to_display)
      key = cv2.waitKey(0)
      if key == 27:
        break

  del exec_net
  del plugin


if __name__ == '__main__':
  sys.exit(main() or 0)
