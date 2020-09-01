import copy
from collections import Counter

import numpy as np
import torch
from torch.utils import data

from action_recognition.utils import cached
from action_recognition.video_reader import make_video_reader, read_flow
from .annotation import load_json_annotation


@cached()
def load_annotation(annotation_path, flow_path, root_path, subset, video_format):
    return load_json_annotation(root_path, annotation_path, subset, flow_path, video_format)


def make_dataset(args, subset, spatial_transform, temporal_transform, target_transform):
    """Constructs VideoDataset instance for specified subset"""
    assert subset in ['training', 'validation', 'testing']

    if subset == 'testing':
        num_samples_per_video = args.n_test_clips
    elif subset == 'validation':
        num_samples_per_video = args.n_val_clips
    else:  # train
        num_samples_per_video = args.n_clips

    if subset == 'testing':
        if args.test_subset == 'val':
            subset = 'validation'
        elif args.test_subset == 'test':
            subset = 'testing'

    return_flow = False
    return_rgb = True
    if "flow" in args.model:
        if "two_stream" not in args.model:
            return_rgb = False
        return_flow = True

    return VideoDataset(
        args.video_path,
        args.annotation_path,
        subset,
        num_samples_per_video,
        spatial_transform,
        temporal_transform,
        target_transform,
        sample_duration=(args.sample_duration * args.temporal_stride),
        flow_path=args.flow_path,
        return_rgb=return_rgb,
        return_flow=return_flow,
        video_format=getattr(args, 'video_format', None),
        image_reader=getattr(args, 'image_reader', "opencv")
    )


def sample_clips(videos, num_samples_per_video, sample_duration):
    """Extracts clips with given length from each video.
    Args:
        videos: List of video samples
        num_samples_per_video: Number of clips sampled for each video. If num_samples_per_video = 1 ,
            then all frames are returned for each video. If num_samples_per_video <= 0, then all sequential
            clips are sampled from video. If num_samples_per_video > 1, then clips are sampled uniformly.
        sample_duration: Number of frames in sampled clips. Actual value may be smaller for short clips.

    Returns: List of clip samples

    """
    videos = sorted(videos, key=lambda v: v['video'].split('/')[-1])
    clips = []
    for sample in videos:
        segment_start, segment_end = sample['segment']
        n_frames = segment_end - segment_start + 1

        if num_samples_per_video == 1:
            # use all frames from video
            sample['frame_indices'] = list(range(segment_start, segment_end + 1))
            clips.append(sample)

        else:

            num_samples = num_samples_per_video
            if num_samples_per_video <= 0:
                # Use all clips of sample_duration or the whole video if it's short enough.
                step = 1
                num_samples = max(1, n_frames - sample_duration + 1)
            else:
                step = max(1, (n_frames - sample_duration) // (num_samples_per_video - 1))

            for clip_start in range(segment_start, segment_start + step * num_samples, step):
                sampled_clip = copy.deepcopy(sample)
                clip_end = min(segment_end + 1, clip_start + sample_duration)
                sampled_clip['frame_indices'] = list(range(clip_start, clip_end))
                if sampled_clip['frame_indices']:
                    clips.append(sampled_clip)

    return clips


class VideoDataset(data.Dataset):
    """Generic video dataset.

    Args:
        video_path (Path): Directory with video files. Will be used by annotation_loader to resolve real paths.
        annotation_path (Path): Path to annotation file.
        subset (str): Which subset of dataset to use (validation/training/testing)
        n_samples_for_each_video (int): How many clips should be sampled from every video per epoch.
        spatial_transform (callable): A function/transform that takes in a clip (list of frames) and returns
            transformed version
        temporal_transform (callable): A function/transform that takes a list of clip frames and returns transformed
            version
        target_transform (callable): A function/transform that takes in the annotation object and returns transformed
            version
        sample_duration (int): Number of frames in sampled clips
        flow_path (Path): Path to a optical flow directory
        return_rgb (bool): Whether RGB frame should be returned
        return_flow (bool): Whether Optical flow should be returned
        video_reader (callable): Callable that takes in a path to video and transformed frame indices and returns
            list of frames. If None, then object will be created according to the video_format.
        video_format (str): Type of video_loader to be instantiated. If "video", then created video_loader will
            attempt to read frames from .mp4 file. If "frames", then it will try to read from directory with images
        image_reader (str): Backend for reading image files (pil, opencv, accimage)
    """

    def __init__(
            self,
            video_path,
            annotation_path,
            subset,
            n_samples_for_each_video=1,
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None,
            sample_duration=16,
            flow_path=None,
            return_rgb=True,
            return_flow=False,
            video_reader=None,
            video_format='frames',
            image_reader='opencv'
    ):
        if not video_reader:
            self.video_loader = make_video_reader(video_format, image_reader)
        else:
            self.video_loader = video_reader

        self.data, self.class_names = load_annotation(annotation_path, flow_path, video_path, subset, video_format)

        if not self.data:
            raise ValueError("No videos found in {!s} directory. Please check correctness of provided paths"
                             .format(video_path))

        self.data = sample_clips(self.data, n_samples_for_each_video, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.return_rgb = return_rgb
        self.return_flow = return_flow

    def __str__(self):
        return 'VideoDataset(rgb={}, flow={}, classes={}, len={})'.format(
            self.return_rgb,
            self.return_flow,
            self.class_names,
            len(self)
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clips (dict): Dictionary where keys are input modalities and values are corresponding tensors
            targets (dict): Dictionary with annotation data (label, video_id, etc)
        """
        clip_index = index // len(self.spatial_transform)
        spatial_transform_index = index % len(self.spatial_transform)

        self.spatial_transform[spatial_transform_index].randomize_parameters()

        frame_indices = self.data[clip_index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clips = {
            **self._read_rgb(clip_index, frame_indices, spatial_transform_index),
            **self._read_flow(clip_index, frame_indices, spatial_transform_index)
        }

        target = self.data[clip_index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clips, target

    def _read_rgb(self, clip_index, frames, spatial_transform_index):
        if not self.return_rgb:
            return {}

        video_path = self.data[clip_index]['video']
        clip = self.video_loader(str(video_path), frames)

        clip = [self.spatial_transform[spatial_transform_index](frame) for frame in clip]

        return {'rgb_clip': torch.stack(clip, 0)}

    def _read_flow(self, clip_index, frames, spatial_transform_index):
        if not self.return_flow:
            return {}

        flow_path = self.data[clip_index]['flow']
        clip = read_flow(str(flow_path), frames)

        clip = [self.spatial_transform[spatial_transform_index](frame) for frame in clip]

        clip = torch.stack(clip, 0)
        N, _, H, W = clip.shape
        clip = clip.view((N // 2, 2, H, W))
        return {'flow_clip': clip}

    def __len__(self):
        return len(self.data) * len(self.spatial_transform)

    def get_sample_weights(self, class_weights):
        """Transforms per-class sampling probability to per-clip sampling probability.
        Used with torch.utils.data.WeightedRandomSampler in order to balance classes"""
        if class_weights is not None:
            class_weights = np.asarray(class_weights)
            class_weights /= np.sum(class_weights)
        else:
            num_labels = len(self.class_names)
            sample_count = Counter(data_elem['label'] for data_elem in self.data)
            class_weights = [(1 / sample_count[label]) / num_labels for label in range(num_labels)]

        return [class_weights[data_elem['label']] for data_elem in self.data]
