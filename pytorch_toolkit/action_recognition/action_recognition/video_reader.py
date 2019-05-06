import os
import numpy as np

import cv2
from PIL import Image


class VideoReader:
    """This class reads specified frame numbers from video"""

    def read(self, video_path, frame_indices):
        raise NotImplementedError

    def __call__(self, video_path, frame_indices):
        return self.read(video_path, frame_indices)


class ImageDirReader(VideoReader):
    """Reads clip from directory with frames with a given file name pattern

    For example:
        video1/frame_00001.jpg
        video1/frame_00002.jpg
        ....

    """

    def __init__(self, read_image_fn, frame_name_pattern):
        self.read_image_fn = read_image_fn
        self.frame_name_pattern = frame_name_pattern

    def read(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_path, self.frame_name_pattern % i)
            if os.path.exists(image_path):
                video.append(self.read_image_fn(image_path))
            else:
                return video

        return video


class VideoFileReader(VideoReader):
    """Reads clip from video file."""

    def read(self, video_path, frame_indices):
        video = []
        cap = cv2.VideoCapture(video_path)

        unique_indices = sorted(set(frame_indices))
        current_frame_idx = 0
        frame_map = {}
        for next_frame_idx in unique_indices:
            while current_frame_idx != next_frame_idx:
                status, frame = cap.read()
                current_frame_idx += 1
                if not status:
                    cap.release()
                    return video
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_map[next_frame_idx] = frame
        cap.release()
        video = [frame_map[i] for i in frame_indices]
        return video


def pil_read_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            return np.asarray(img)


def accimage_read_image(path):
    try:
        import accimage
        return np.asarray(accimage.Image(path))
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_read_image(path)


def opencv_read_image(path):
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def make_video_reader(video_format='frames', image_reader='opencv'):
    if image_reader == 'opencv':
        image_read_fn = opencv_read_image
    elif image_reader == 'pil':
        image_read_fn = pil_read_image
    elif image_reader == 'accimage':
        image_read_fn = accimage_read_image
    else:
        raise ValueError("Unknown image reading function")

    if video_format and video_format.lower() == 'video':
        return VideoFileReader()
    return ImageDirReader(image_read_fn, 'image_%05d.jpg')


def read_flow(path, frame_indices):
    video = []
    for i in frame_indices[:-1]:
        flow_x_path = os.path.join(path, 'flow_x_{:05d}.jpg'.format(i))
        flow_y_path = os.path.join(path, 'flow_y_{:05d}.jpg'.format(i))
        if os.path.exists(flow_x_path) and os.path.exists(flow_y_path):
            with open(flow_x_path, 'rb') as f:
                with Image.open(f) as img:
                    flow_x = img.convert('L')
            with open(flow_y_path, 'rb') as f:
                with Image.open(f) as img:
                    flow_y = img.convert('L')
            video.append(flow_x)
            video.append(flow_y)
        else:
            raise Exception(flow_x_path + " does not exist")
    return video
