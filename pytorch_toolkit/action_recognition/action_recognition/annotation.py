import cv2

from .utils import load_json, load_value_file


def get_video_names_and_annotations(data, subset):
    """Selects clips of a given subset from the parsed json annotation"""
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value)

    return video_names, annotations


def get_video_props(video_path, video_format, annotation):
    """Tries to read video properties (total number of frames and FPS) from annotation
    file or read it from file otherwise"""

    n_frames = annotation.get('n_frames')
    fps = annotation.get('fps')
    if n_frames and fps:
        return n_frames, fps

    if video_format == 'frames':
        if not video_path.exists():
            return 0, 0
        n_frames = int(load_value_file(video_path / 'n_frames'))
        fps = 30
    else:
        cap = cv2.VideoCapture(video_path.as_posix())
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    return n_frames, fps


def load_json_annotation(root_path, annotation_path, subset, flow_path=None, video_format='frames'):
    """Load annotation in ActivityNet-like format"""
    data = load_json(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)

    idx_to_class = dict(enumerate(data['labels']))
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    videos = []
    for i, (video_name, annotation) in enumerate(zip(video_names, annotations)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        if video_format == 'video' and not video_name.lower().endswith('.mp4'):
            video_name += '.mp4'

        video_path = root_path / video_name

        n_frames, fps = get_video_props(video_path, video_format, annotation)

        if n_frames == 0:
            continue

        flow_full_path = flow_path
        if flow_path is not None:
            flow_full_path = (flow_path / video_name).as_posix()

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path.as_posix(),
            'flow': flow_full_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'fps': fps,
            'video_id': video_name.split('/')[1],
            'label': class_to_idx[annotation['annotations']['label']],
        }
        videos.append(sample)

    return videos, idx_to_class
