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

import glog as log
import cv2 as cv
import numpy as np
from scipy.spatial.distance import cosine

from utils import face_align
from utils.ie_tools import load_ie_model


class FaceDetector:
    """Wrapper class for face detector"""
    def __init__(self, model_path, conf=.6, device='CPU', ext_path=''):
        self.net = load_ie_model(model_path, device, None, ext_path)
        self.confidence = conf
        self.expand_ratio = (1.1, 1.05)

    def get_detections(self, frame):
        """Returns all detections on frame"""
        _, _, h, w = self.net.get_input_shape().shape
        out = self.net.forward(cv.resize(frame, (w, h)))
        detections = self.__decode_detections(out, frame.shape)
        return detections

    def __decode_detections(self, out, frame_shape):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > self.confidence:
                left = int(max(detection[3], 0) * frame_shape[1])
                top = int(max(detection[4], 0) * frame_shape[0])
                right = int(max(detection[5], 0) * frame_shape[1])
                bottom = int(max(detection[6], 0) * frame_shape[0])
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections


class VectorCNN:
    """Wrapper class for a nework returning a vector"""
    def __init__(self, model_path, device='CPU'):
        self.net = load_ie_model(model_path, device, None)

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        _, _, h, w = self.net.get_input_shape().shape
        outputs = [self.net.forward(cv.resize(frame, (w, h))) for frame in batch]
        return outputs


def get_embeddings(frame, detections, face_reid, landmarks_predictor):
    """Get embeddings for all detected faces on the frame"""
    rois = []
    embeddings = []
    for rect, _ in detections:
        left, top, right, bottom = rect
        rois.append(frame[top:bottom, left:right])

    if rois:
        landmarks = landmarks_predictor.forward(rois)
        assert len(landmarks) == len(rois)

        for i, _ in enumerate(rois):
            roi_keypoints = landmarks[i].reshape(-1)
            rois[i] = face_align.FivePointsAligner.align(rois[i], roi_keypoints,
                                                         d_size=(rois[i].shape[1], rois[i].shape[0]),
                                                         normalized=True, show=False)
        embeddings = face_reid.forward(rois)
        assert len(rois) == len(embeddings)

    return embeddings


def find_nearest(x, gallery, thr):
    """Finds the nearest to a given embedding in the gallery"""
    if gallery:
        diffs = np.array([cosine(x, y) for y in gallery.values()])
        min_pos = diffs.argmin()
        min_dist = diffs[min_pos]
        if min_dist < thr:
            return min_pos, list(gallery.keys())[min_pos]
    return None, None


def match_embeddings(embeds, gallery, thr):
    """Matches input embeddings with ones in the gallery"""
    indexes = []
    for emb in embeds:
        _, name = find_nearest(emb, gallery, thr)
        if name is not None:
            indexes.append(name)
        else:
            indexes.append('Unknown')

    return indexes, gallery


def draw_detections(frame, detections, indexes):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        label = str(indexes[i])
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame


def load_gallery(path_to_gallery, face_det, landmarks_detector, face_recognizer):
    """Computes embeddings for gallery"""
    gallery = {}
    files = os.listdir(path_to_gallery)
    files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
    for file in files:
        img = cv.imread(osp.join(path_to_gallery, file))
        detections = face_det.get_detections(img)

        if not detections:
            detections = [[0, 0, img.shape[0], img.shape[1]], 0]
            log.warn('Warning: failed to detect face on the image ' + file)

        embed = get_embeddings(img, detections, face_recognizer, landmarks_detector)
        gallery[file.replace('.png', '').replace('.jpg', '')] = embed[0]
    return gallery


def run(params, capture, face_det, face_recognizer, landmarks_detector):
    """Starts the face recognition demo"""
    win_name = 'Deep Face Recognition'
    gallery = load_gallery(params.path_to_gallery, face_det, landmarks_detector, face_recognizer)

    while cv.waitKey(1) != 27:
        has_frame, frame = capture.read()
        if not has_frame:
            return

        detections = face_det.get_detections(frame)
        embeds = get_embeddings(frame, detections, face_recognizer, landmarks_detector)
        ids, gallery = match_embeddings(embeds, gallery, params.fr_thresh)
        frame = draw_detections(frame, detections, ids)
        cv.imshow(win_name, frame)

def main():
    """Prepares data for the face recognition demo"""
    parser = argparse.ArgumentParser(description='Face recognition live demo script')
    parser.add_argument('--video', type=str, default=None, help='Input video')
    parser.add_argument('--cam_id', type=int, default=-1, help='Input cam')

    parser.add_argument('--fd_model', type=str, required=True)
    parser.add_argument('--fd_thresh', type=float, default=0.6, help='Threshold for FD')

    parser.add_argument('--fr_model', type=str, required=True)
    parser.add_argument('--fr_thresh', type=float, default=0.6, help='Threshold for FR')

    parser.add_argument('--path_to_gallery', type=str, required=True, help='Path to gallery with subjects')

    parser.add_argument('--ld_model', type=str, default='', help='Path to a snapshots with landmarks detection model')

    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels '
                             'impl.', type=str, default=None)

    args = parser.parse_args()

    if args.cam_id >= 0:
        log.info('Reading from cam {}'.format(args.cam_id))
        cap = cv.VideoCapture(args.cam_id)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    else:
        assert args.video
        log.info('Reading from {}'.format(args.video))
        cap = cv.VideoCapture(args.video)
    assert cap.isOpened()

    face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device, args.cpu_extension)
    face_recognizer = VectorCNN(args.fr_model, args.device)
    landmarks_detector = VectorCNN(args.ld_model, args.device)
    run(args, cap, face_detector, face_recognizer, landmarks_detector)

if __name__ == '__main__':
    main()
