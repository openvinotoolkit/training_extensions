class Compose(object):
    """Compose multiple target transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):
    """Returns video label and name. Used for training and validation."""

    def __call__(self, target):
        return {
            'label': target['label'],
            'video': target['video']
        }


class VideoID(object):
    """Returns video name. Used for video prediction."""

    def __call__(self, target):
        return {
            'label': target['label'],
            'video_id': target['video_id']
        }
