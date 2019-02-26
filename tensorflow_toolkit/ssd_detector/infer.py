from __future__ import print_function
import argparse
import json
import os
import pickle

import cv2
import tensorflow as tf
from utils.helpers import load_module
from ssd_detector.readers.object_detector_json import ObjectDetectorJson
from ssd_detector.trainer import create_session, detection_model, InputInferData, InputValData


def parse_args():
  parser = argparse.ArgumentParser(
    description='Perform inference of a detection model on a video file or an annotation file in JSON format')

  parser.add_argument('path_to_config', help='Path to a config.py')

  input_type_group = parser.add_mutually_exclusive_group(required=True)
  input_type_group.add_argument('--json', dest='input_type', action='store_const', const='json',
                                help='Get images from annotation file in JSON format')
  input_type_group.add_argument('--video', dest='input_type', action='store_const', const='video',
                                help='Get images from video file')
  input_type_group.set_defaults(input_type='json')

  parser.add_argument('--input', help='Path to the input file', required=True)
  parser.add_argument('--conf_threshold', type=float, help='Confidence threshold for detector', default=0.1)

  dump_to_json_group = parser.add_mutually_exclusive_group()
  dump_to_json_group.add_argument('--dump-to-json', dest='dump_to_json', action='store_true')
  dump_to_json_group.add_argument('--no-dump-to-json', dest='dump_to_json', action='store_false')
  dump_to_json_group.set_defaults(dump_to_json=True)

  show_group = parser.add_mutually_exclusive_group()
  show_group.add_argument('--show', dest='show', action='store_true')
  show_group.add_argument('--no-show', dest='show', action='store_false')
  show_group.set_defaults(show=False)

  dump_output_video_group = parser.add_mutually_exclusive_group()
  dump_output_video_group.add_argument('--dump-to-video', dest='dump_output_video', action='store_true')
  dump_output_video_group.add_argument('--no-dump-to-video', dest='dump_output_video', action='store_false')
  dump_output_video_group.set_defaults(dump_output_video=False)

  parser.add_argument('--path_to_output_video', help='Path to output video with predictions', default='output.avi')
  return parser.parse_args()


# pylint: disable=too-many-locals
def process_image(predictions, img_size, img_id, conf_threshold, classes):
  img_width, img_height = img_size

  coco_detections = []
  for prediction in predictions:
    det_label = int(prediction[0])
    det_conf = float(prediction[1])

    if conf_threshold and det_conf < conf_threshold:
      continue

    top_left_x = float(prediction[2] * img_width)
    top_left_y = float(prediction[3] * img_height)
    bottom_right_x = float(prediction[4] * img_width)
    bottom_right_y = float(prediction[5] * img_height)
    if det_label >= len(classes):
      print('Wrong label: {0}'.format(det_label))
      exit(1)

    obj_width = round(bottom_right_x - top_left_x, 1)
    obj_height = round(bottom_right_y - top_left_y, 1)

    coco_det = dict()
    coco_det['image_id'] = img_id
    coco_det['category_id'] = det_label
    coco_det['bbox'] = [round(top_left_x, 1), round(top_left_y, 1), obj_width, obj_height]
    coco_det['score'] = det_conf
    coco_detections.append(coco_det)

  return coco_detections


def draw_detections(img, coco_detections):
  for det in coco_detections:
    x, y, width, height = det['bbox'] # pylint: disable=invalid-name
    top_left = (int(x), int(y))
    bottom_right = (int(x + width), int(y + height))
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
    cv2.putText(img, '{0}: {1}'.format(det['category_id'], det['score']), top_left,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  return img


# pylint: disable=too-many-arguments
def predict_on_video(predictions, path_to_video, classes, show=False, conf_threshold=None, dump_output_video=False,
                     path_to_output_video='output.avi'):
  output = []

  cap = cv2.VideoCapture(path_to_video)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  if dump_output_video:
    out = cv2.VideoWriter(path_to_output_video, fourcc, fps, (int(width), int(height)))
  for i, pred in enumerate(predictions):
    if cap.isOpened():
      _, frame = cap.read()
      det = process_image(pred[:, 1:], (width, height), i, conf_threshold, classes)
      output.extend(det)
      img = draw_detections(frame, det)
      if show:
        cv2.imshow('detections', img)
        key = cv2.waitKey(10)
        if key == 27:
          break
      if dump_output_video:
        out.write(img)
    else:
      break

  cap.release()
  if dump_output_video:
    out.release()

  return output


def predict_on_json(predictions, annotation_path, classes, show=False, conf_threshold=None, dump_output_video=False,
                    path_to_output_video='output.avi', width=640, height=480, fps=30):
  annotation_generator, _ = ObjectDetectorJson.json_iterator(annotation_path, classes)
  annotation_data = [pickle.loads(x) for x in annotation_generator()]

  output = []
  if dump_output_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path_to_output_video, fourcc, fps, (int(width), int(height)))
  for i, pred in enumerate(predictions):
    annotation = annotation_data[i]
    image_size = annotation['image_size']
    img_path = annotation['image']
    img_id = annotation['image_id']
    det = process_image(pred[:, 1:], image_size, img_id, conf_threshold, classes)
    output.extend(det)
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, tuple(image_size))
    img = draw_detections(frame, det)
    if show:
      cv2.imshow('detections', img)
      key = cv2.waitKey(10)
      if key == 27:
        break
    if dump_output_video:
      img_resized = cv2.resize(img, (width, height))
      out.write(img_resized)
  if dump_output_video:
    out.release()

  return output


def infer(config, source, path, conf_threshold=None, dump_to_json=False,
          show=False, dump_output_video=False, path_to_output_video='output.avi'):
  session_config = create_session(config, 'infer')

  out_dir = os.path.join(config.MODEL_DIR, config.infer.out_subdir)
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  run_config = tf.estimator.RunConfig(session_config=session_config)

  config.detector_params['log_dir'] = config.MODEL_DIR

  predictor = tf.estimator.Estimator(model_fn=detection_model,
                                     params=config.detector_params, model_dir=config.MODEL_DIR, config=run_config)

  checkpoint_path = tf.train.latest_checkpoint(config.MODEL_DIR)

  basename = os.path.basename(path)
  filename = os.path.splitext(basename)[0]
  name = '{0}.json'.format(filename)
  output_json_path = os.path.join(out_dir, name)

  if source == 'video':
    input_data = InputInferData(path, config.input_shape, config.infer.batch_size)
  elif source == 'json':
    input_data = InputValData(batch_size=config.infer.batch_size, input_shape=config.input_shape, json_path=path,
                              classes=config.classes,
                              num_parallel_calls=config.infer.execution.transformer_parallel_calls,
                              prefetch_size=config.infer.execution.transformer_prefetch_size)

  predictions = predictor.predict(input_fn=input_data.input_fn, checkpoint_path=checkpoint_path)
  if source == 'video':
    predictions = predict_on_video(predictions, path, config.classes, show, conf_threshold, dump_output_video,
                                   path_to_output_video)
  elif source == 'json':
    predictions = predict_on_json(predictions, path, config.classes, show, conf_threshold, dump_output_video,
                                  path_to_output_video)

  if dump_to_json:
    with open(output_json_path, 'w') as output_file:
      json.dump(predictions, output_file, sort_keys=True, indent=4)


def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)

  infer(cfg, args.input_type, args.input, args.conf_threshold, args.dump_to_json, args.show,
        args.dump_output_video, args.path_to_output_video)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
