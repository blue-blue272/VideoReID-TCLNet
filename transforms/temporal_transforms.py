from __future__ import absolute_import

import random
import math
import numpy as np


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = list(frame_indices)

        while len(out) < self.size:
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size=4):
        self.size = size

    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)
        size = self.size

        if len(frame_indices) >= (size - 1) * 8 + 1:
            out = frame_indices[0: (size - 1) * 8 + 1: 8]
        elif len(frame_indices) >= (size - 1) * 4 + 1:
            out = frame_indices[0: (size - 1) * 4 + 1: 4]
        elif len(frame_indices) >= (size - 1) * 2 + 1:
            out = frame_indices[0: (size - 1) * 2 + 1: 2]
        elif len(frame_indices) >= size:
            out = frame_indices[0:size:1]
        else:
            out = frame_indices[0:size]
            while len(out) < size:
                for index in out:
                    if len(out) >= size:
                        break
                    out.append(index)

        return out



class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size=4, stride=8):
        self.size = size
        self.stride = stride

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)

        if len(frame_indices) >= self.size * self.stride:
            rand_end = len(frame_indices) - (self.size - 1) * self.stride - 1
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + (self.size - 1) * self.stride + 1
            out = frame_indices[begin_index:end_index:self.stride]
        elif len(frame_indices) >= self.size:
            index = np.random.choice(len(frame_indices), size=self.size, replace=False)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        else:
            index = np.random.choice(len(frame_indices), size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]

        return out
