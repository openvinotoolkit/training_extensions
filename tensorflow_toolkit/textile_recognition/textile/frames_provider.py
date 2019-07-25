# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import cv2
from textile.textile_cv_detector.detect_by_simple_dense_optical_flow import TextileDetector, get_rect_tl, get_rect_br


class FramesProvider:
    def __init__(self, images_folder):
        self.impaths = []
        self.probe_classes = []
        for probe_class in os.listdir(images_folder):
            for path in os.listdir(os.path.join(images_folder, probe_class)):
                full_path = os.path.join(images_folder, probe_class, path)
                self.impaths.append(full_path)
                self.probe_classes.append(probe_class)

    def go_to_next_video(self):
        pass

    def frames_gen(self):
        for impath, probe_class in zip(self.impaths, self.probe_classes):
            image = cv2.imread(impath)
            yield image, probe_class, image


class CvatFramesProvider:
    def __init__(self, cvat_xmls_folder, videos_root_folder):
        from textile.cvat_annotation import CvatAnnotation

        self.should_go_to_next_video = False
        self.annotations = []
        for xml_path in os.listdir(cvat_xmls_folder):
            self.annotations.append(CvatAnnotation(os.path.join(cvat_xmls_folder, xml_path)))
        self.videos_root_folder = videos_root_folder

    def go_to_next_video(self):
        self.should_go_to_next_video = True

    def frames_gen(self):
        for annotation in self.annotations:
            video_path = os.path.join(self.videos_root_folder, annotation.task_name)
            cap = cv2.VideoCapture(video_path)

            frame_idx = 0
            while True:
                if self.should_go_to_next_video:
                    self.should_go_to_next_video = False
                    break

                _, frame = cap.read()

                if frame is None:
                    break

                probe_class = None

                view_frame = frame.copy()

                if annotation.annotation[frame_idx]:
                    obj = annotation.annotation[frame_idx][0]
                    rect = (obj['xtl'], obj['ytl']), (obj['xbr'], obj['ybr'])
                    probe_class = obj['id']

                    frame = frame[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

                    cv2.rectangle(view_frame, rect[0], rect[1], (255, 0, 255), 20)

                frame_idx += 1

                yield frame, probe_class, view_frame


class VideoFramesProvider:
    def __init__(self, filelist_path, videos_root_folder):
        self.videos_list = []
        self.videos_root_folder = videos_root_folder
        self.frame_step = 5
        self.should_go_to_next_video = False

        data_dir = os.path.dirname(filelist_path)
        with open(filelist_path) as f_list:
            for line in f_list:
                line = line.strip()
                if line.startswith("#"):
                    continue
                chunks = line.split()
                assert len(chunks) == 2
                video_path, probe_class = chunks
                self.videos_list.append({"rel_path": os.path.join(data_dir, video_path), "probe_class": probe_class})

    def go_to_next_video(self):
        self.should_go_to_next_video = True

    def frames_gen(self):
        for video_descr in self.videos_list:
            rel_path = video_descr["rel_path"]
            probe_class = video_descr["probe_class"]
            video_path = rel_path # os.path.join(self.videos_root_folder, rel_path)

            if not os.path.exists(video_path):
                raise Exception('File not found: {}'.format(video_path))

            textile_detector = TextileDetector(self.frame_step)

            cap = cv2.VideoCapture(video_path)

            while True:
                if self.should_go_to_next_video:
                    self.should_go_to_next_video = False
                    break
                _, frame = cap.read()

                if frame is None:
                    break

                view_frame = frame.copy()

                bbox = textile_detector.handle_frame(frame)
                if bbox is not None:
                    tl_x, tl_y = get_rect_tl(bbox)
                    br_x, br_y = get_rect_br(bbox)

                    frame = frame[tl_y:br_y, tl_x:br_x]
                    cv2.rectangle(view_frame, (tl_x, tl_y), (br_x, br_y), (255, 0, 255), 20)
                else:
                    frame = None

                yield frame, probe_class, view_frame
