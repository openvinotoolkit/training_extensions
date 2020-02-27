import random

from .spatial_transforms import _repr_params


class LoopPadding(object):
    """Extend short clip to a given size"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(size=self.size)


class TemporalStride(object):
    """Skips frames with a given step. Increases effective temporal receptive field."""

    def __init__(self, stride=1):
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[::self.stride]

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(stride=self.stride)


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(size=self.size)


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(size=self.size)


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[int(begin_index):int(end_index)]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

    def __repr__(self):
        return self.__class__.__name__ + _repr_params(size=self.size)
