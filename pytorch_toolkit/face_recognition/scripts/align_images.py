"""
 Copyright (c) 2018 Intel Corporation
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
import os.path as osp
import json

import cv2 as cv
import torch
from tqdm import tqdm
from torchvision.transforms import transforms

from model import landnet
from utils import utils
from utils import augmentation
from utils.face_align import FivePointsAligner

class LandnetPT:
    """Wrapper for landmarks regression model"""
    def __init__(self, model):
        self.net = model
        self.transformer = transforms.Compose(
            [augmentation.ResizeNumpy((48, 48)), augmentation.NumpyToTensor(switch_rb=True)])

    def get_landmarks(self, batch):
        converted_batch = []
        for item in batch:
            converted_batch.append(self.transformer(item))
        pt_blob = torch.stack(converted_batch).cuda()
        landmarks = self.net(pt_blob)
        return landmarks.data.cpu().numpy()


class FaceDetector:
    """Wrapper class for face detector"""
    def __init__(self, proto, model, conf=.6, expand_ratio=(1.1, 1.05), size=(300, 300)):
        self.net = cv.dnn.readNetFromCaffe(proto, model)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        last_layer_id = self.net.getLayerId(self.net.getLayerNames()[-1])
        last_layer = self.net.getLayer(last_layer_id)
        assert last_layer.type == 'DetectionOutput'

        self.confidence = conf
        self.expand_ratio = expand_ratio
        self.det_res = size

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        frame_height = frame_shape[0]
        frame_width = frame_shape[1]
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_width)
                top = int(max(detection[4], 0) * frame_height)
                right = int(max(detection[5], 0) * frame_width)
                bottom = int(max(detection[6], 0) * frame_height)
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                # classId = int(detection[1]) - 1  # Skip background label
                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections

    def get_detections(self, frame):
        """Returns all detections on frame"""
        blob = cv.dnn.blobFromImage(frame, 1., (self.det_res[0], self.det_res[1]), crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        detections = self.__decode_detections(out, frame.shape)
        return detections


def draw_detections(frame, detections, landmarks):
    """Draw detections and landmarks on a frame"""
    for _, rect in enumerate(detections):
        left, top, right, bottom = rect
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        for point in landmarks.reshape(-1, 2):
            point = (int(left + point[0] * (right - left)), int(top + point[1] * (bottom - top)))
            cv.circle(frame, point, 5, (255, 0, 0), -1)

    return frame


def run_dumping(images_list, face_det, landmarks_regressor, vis_flag):
    """Dumps detections and landmarks from images"""
    detected_num = 0
    data = []
    for path in tqdm(images_list, 'Dumping data'):
        image = cv.imread(path, cv.IMREAD_COLOR)
        if image is None:
            continue

        detections = face_det.get_detections(image)
        landmarks = None
        if detections:
            left, top, right, bottom = detections[0][0]
            roi = image[top:bottom, left:right]
            landmarks = landmarks_regressor.get_landmarks([roi]).reshape(-1)
            data.append({'path': path, 'bbox': detections[0][0], 'landmarks': landmarks})
            detected_num += 1
            if vis_flag:
                FivePointsAligner.align(roi, landmarks,
                                        d_size=(200,200), normalize=False, show=True)
        else:
            data.append({'path': path, 'bbox': None, 'landmarks': None})

    print('Detection ratio: ', float(detected_num) / len(data))

    return data


def create_images_list(images_root, imgs_list):
    input_filenames = []
    input_dir = os.path.abspath(images_root)

    if imgs_list is None:
        stop = False
        for path, _, files in os.walk(input_dir):
            if stop:
                break
            for name in files:
                if name.lower().endswith('.jpg') or name.lower().endswith('.png') \
                        or name.lower().endswith('.jpeg') or name.lower().endswith('.gif') \
                        or not '.' in name:
                    filename = os.path.join(path, name)
                    input_filenames.append(filename)
    else:
        with open(imgs_list) as f:
            data = json.load(f)
            for path in data['path']:
                filename = osp.join(images_root, path)
                input_filenames.append(filename)

    return input_filenames


def save_data(data, filename, root_dir):
    print('Saving data...')
    with open(filename, 'w') as f:
        for instance in data:
            line = osp.relpath(instance['path'], start=root_dir) + ' | '
            if instance['bbox'] is not None:
                for x in instance['landmarks']:
                    line += str(x) + ' '
                line += ' | '
                left, top, right, bottom = instance['bbox']
                line += str(left) + ' ' + str(top) + ' ' + str(right - left) + ' ' + str(bottom - top)

            f.write(line.strip() + '\n')

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--images_root', type=str, default=None, required=True)
    parser.add_argument('--images_list', type=str, default=None, required=False)
    parser.add_argument('--fd_proto', type=str, default='../demo/face_detector/deploy_fd.prototxt', help='')
    parser.add_argument('--fd_model', type=str, default='../demo/face_detector/sq_300x300_iter_120000.caffemodel',
                        help='')
    parser.add_argument('--fr_thresh', type=float, default=0.1)
    parser.add_argument('--det_res', type=int, nargs=2, default=[300, 300], help='Detection net input resolution.')
    parser.add_argument('--landnet_model', type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    face_detector = FaceDetector(args.fd_proto, args.fd_model, conf=args.fr_thresh, size=args.det_res)

    with torch.cuda.device(args.device):
        landmarks_regressor = utils.load_model_state(landnet.LandmarksNet(), args.landnet_model, args.device)
        data = run_dumping(create_images_list(args.images_root, args.images_list), face_detector,
                           LandnetPT(landmarks_regressor), args.visualize)
        save_data(data, osp.join(args.images_root, 'list.txt'), args.images_root)

if __name__ == '__main__':
    main()
