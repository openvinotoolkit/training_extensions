from __future__ import print_function
from argparse import ArgumentParser
import json
import logging as log
import os
import sys

import cv2
from pycocotools.coco import COCO

from openvino.inference_engine import IENetwork, IEPlugin
from ssd_detector.readers.object_detector_json import ObjectDetectorJson


def build_argparser():
  parser = ArgumentParser()
  parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
  parser.add_argument("-it", "--input_type",
                      help="Input type for the model, could be video file or annotation file in JSON format or 'cam'"
                           "for capturing video stream from camera",
                      default="json", choices=["video", "json", "cam"], required=True, )
  parser.add_argument("-i", "--input", help="Path to video file or annotation file in json format", required=True,
                      type=str)
  parser.add_argument("-l", "--cpu_extension",
                      help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                           "impl.", type=str, default=None)
  parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
  parser.add_argument("-d", "--device",
                      help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                           "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                      type=str)
  parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                      default=0.5, type=float)
  parser.add_argument("--dump_predictions_to_json", help="Dump predictions to json file", default=True)
  parser.add_argument("--output_json_path", help="Path to output json file with predictions",
                      default="ie_preidctions.json")
  parser.add_argument("--show", dest='show', action='store_true', help="Show predictions")
  parser.add_argument("--dump_output_video", help="Save output video with predictions", default=True)
  parser.add_argument("--path_to_output_video", help="Path to output video with predictions", default="output.avi")
  parser.add_argument("--delay", help="Delay before show next image", default=10, type=int)
  return parser


class Input:
  def __init__(self, input_type, input):
    self.input_type = input_type
    self.item_counter = 0

    assert input_type in ('json', 'video', 'cam')

    if input_type == 'json':
      coco_annotation = COCO(input)
      annotation_directory = os.path.join(os.getcwd(), os.path.dirname(input))
      classes = ObjectDetectorJson.get_classes_from_coco_annotation(input)
      self.json_data = ObjectDetectorJson.convert_coco_to_toolbox_format(coco_annotation, classes,
                                                                         annotation_directory)
    if input_type == 'video':
      self.cap = cv2.VideoCapture(input)

    if input_type == 'cam':
      self.cap = cv2.VideoCapture(0)

  def get_next_item(self):
    assert self.input_type in ('json', 'video', 'cam')

    if self.input_type == 'json':
      image_size = self.json_data[self.item_counter]['image_size']
      img = cv2.imread(self.json_data[self.item_counter]['image'])
      img = cv2.resize(img, tuple(image_size))
      annot = self.json_data[self.item_counter]['image_id']
      self.item_counter += 1
      return img, annot

    if self.input_type == 'cam' or 'video':
      _, img = self.cap.read()
      self.item_counter += 1
      return img, self.item_counter

    return None, None

  def is_finished(self):
    assert self.input_type in ('json', 'video', 'cam')

    if self.input_type == 'json':
      return self.item_counter >= len(self.json_data)

    if self.input_type == 'cam' or 'video':
      return not self.cap.isOpened()

    return None, None


# pylint: disable=too-many-locals,too-many-statements
def main():
  log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
  args = build_argparser().parse_args()
  model_xml = args.model
  model_bin = os.path.splitext(model_xml)[0] + ".bin"
  # Plugin initialization for specified device and load extensions library if specified
  log.info("Initializing plugin for %s device...", args.device)
  plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
  if args.cpu_extension and 'CPU' in args.device:
    plugin.add_cpu_extension(args.cpu_extension)

  # Read IR
  log.info("Reading IR...")
  net = IENetwork(model=model_xml, weights=model_bin)

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
  assert len(net.outputs) == 1, "Sample supports only single output topologies"
  input_blob = next(iter(net.inputs))
  out_blob = next(iter(net.outputs))
  log.info("Loading IR to the plugin...")
  exec_net = plugin.load(network=net, num_requests=2)

  # Read and pre-process input image
  batch, channels, height, width = net.inputs[input_blob].shape
  del net

  predictions = []
  data = Input(args.input_type, args.input)
  cur_request_id = 0

  fps = 25
  out_width = 640
  out_height = 480
  if args.dump_output_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.path_to_output_video, fourcc, fps, (int(out_width), int(out_height)))

  while not data.is_finished():
    frame, img_id = data.get_next_item()
    initial_h, initial_w, _ = frame.shape
    in_frame = cv2.resize(frame, (width, height))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((batch, channels, height, width))

    exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
    if exec_net.requests[cur_request_id].wait(-1) == 0:

      # Parse detection results of the current request
      res = exec_net.requests[cur_request_id].outputs[out_blob]
      coco_detections = []
      for obj in res[0][0]:
        # Draw only objects when probability more than specified threshold
        if obj[2] > args.prob_threshold:
          top_left_x = float(obj[3] * initial_w)
          top_left_y = float(obj[4] * initial_h)
          bottom_right_x = float(obj[5] * initial_w)
          bottom_right_y = float(obj[6] * initial_h)

          obj_width = round(bottom_right_x - top_left_x, 1)
          obj_height = round(bottom_right_y - top_left_y, 1)
          class_id = int(obj[1])

          coco_det = {}
          coco_det['image_id'] = img_id
          coco_det['category_id'] = class_id
          coco_det['bbox'] = [round(top_left_x, 1), round(top_left_y, 1), obj_width, obj_height]
          coco_det['score'] = round(float(obj[2]), 1)
          coco_detections.append(coco_det)

          # Draw box and label\class_id
          cv2.rectangle(frame, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)),
                        (255, 0, 0), 2)
          cv2.putText(frame, str(class_id) + ' ' + str(round(obj[2] * 100, 1)) + ' %',
                      (int(top_left_x), int(top_left_y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      predictions.extend(coco_detections)

    if args.dump_output_video:
      img_resized = cv2.resize(frame, (out_width, out_height))
      out.write(img_resized)

    if args.show:
      cv2.imshow("Detection Results", frame)
      key = cv2.waitKey(args.delay)
      if key == 27:
        break

  if args.dump_predictions_to_json:
    with open(args.output_json_path, 'w') as output_file:
      json.dump(predictions, output_file, sort_keys=True, indent=4)

  cv2.destroyAllWindows()
  del exec_net
  del plugin


if __name__ == '__main__':
  sys.exit(main() or 0)
