from __future__ import division, print_function

import json
import os
import subprocess
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
from tqdm import tqdm

FFMPEG_CMD = "ffmpeg -i \"{input!s}\" -q:v {q} {filters} -an -threads 1 -y -loglevel panic \"{output!s}\""
SCALE_OPT = "-vf \"scale=trunc(((iw*{scale})/'min(iw,ih)')/2)*2:trunc(((ih*{scale})/'min(iw,ih)')/2)*2\""


def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # video file exists but contains invalid video stream
        return None, None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return n_frames, fps


def convert_video(video, output, scale, quality):
    if scale:
        filters_opt = SCALE_OPT.format(scale=scale)
    else:
        filters_opt = ""

    cmd_format = FFMPEG_CMD.format(input=video, output=output, filters=filters_opt, q=quality)
    subprocess.call(cmd_format, shell=True)


def callback(data, orig_path, destination_path, scale=None, quality=1, video_format='frames', convert=True):
    video_name, annotation = data

    if annotation['subset'] == 'testing':
        label = 'test'
    else:
        label = annotation['annotations']['label']

    class_path = os.path.join(destination_path, label)
    Path(class_path).mkdir(exist_ok=True)

    for ext in ('.mp4', '.avi'):
        orig_video_path = os.path.join(orig_path, label, video_name) + ext
        if os.path.exists(orig_video_path):
            break
    else:
        return None

    base_video_name = os.path.join(destination_path, label, video_name)
    if video_format == 'video':
        destination_video_path = base_video_name + ".mp4"
    else:  # video_format == 'frames'
        Path(base_video_name).mkdir(exist_ok=True)
        destination_video_path = os.path.join(base_video_name, "image_%05d.jpg")

    if convert:
        convert_video(orig_video_path, destination_video_path, scale=scale, quality=quality)
    else:
        destination_video_path = orig_video_path

    n_frames, fps = get_video_properties(destination_video_path)
    if not n_frames:
        return None
    annotation['fps'] = fps
    annotation['n_frames'] = n_frames
    return video_name, annotation


def convert_videos(annotation_path, raw_path, destination_path, video_size, video_format='frames',
                   video_quality=1, subset=None, convert=True, n_jobs=8):
    with open(annotation_path) as f:
        annotation = json.load(f)

    if subset is None:
        data = annotation['database']
    else:
        data = {k: v for k, v in annotation['database'].items() if v.get('subset') == subset}

    updated_videos = []
    # Pre-process video files
    with Pool(processes=n_jobs) as p:
        cb = partial(callback, orig_path=raw_path, destination_path=destination_path, scale=video_size,
                     video_format=video_format, quality=video_quality, convert=convert)
        for updated in tqdm(p.imap_unordered(cb, data.items()), total=len(data)):
            if updated:
                updated_videos.append(updated)

    # update annotation file (add n_frames and fps)
    annotation['database'].update(updated_videos)
    with open(os.path.join(destination_path, os.path.basename(annotation_path)), 'w') as f:
        json.dump(annotation, f)


if __name__ == "__main__":
    parser = ArgumentParser("Pre-process videos and update annotation file")
    parser.add_argument("-a", "--annotation_file",
                        help="Annotation file in dataset format")
    parser.add_argument("-r", "--raw_dir", help="Directory where raw videos are stored")
    parser.add_argument("-d", "--destination_dir", help="Directory where converted videos should be saved")
    parser.add_argument("--video-size", type=int, help="Size of the smallest size of converted video")
    parser.add_argument("--video-format", default='video', choices=['frames', 'video'],
                        help='In what format preprocessed videos should be stored')
    parser.add_argument("-q", "--video-quality", type=int,
                        help="Quality of output videos. Lower values will likely correspond to a better accuracy, "
                             "but will require more disk space. Optimal value is 1 for videos and 4 for frames")
    parser.add_argument("--subset", type=str, help="Dataset subset")
    parser.add_argument("--no-convert", action="store_true", help="Do not convert videos, only update annotation file")
    parser.add_argument("-j", "--threads", default=8, type=int, help="Number of video conversion threads")
    args = parser.parse_args()

    if not args.video_quality:
        args.video_quality = 1 if args.video_format == 'video' else 4

    if args.no_convert:
        args.destination_dir = args.raw_dir

    if not os.path.exists(args.destination_dir):
        os.mkdir(args.destination_dir)

    convert_videos(args.annotation_file, args.raw_dir, args.destination_dir, video_size=args.video_size,
                   n_jobs=args.threads, subset=args.subset, video_format=args.video_format,
                   video_quality=args.video_quality, convert=(not args.no_convert))
