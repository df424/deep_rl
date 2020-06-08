
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import cv2
import math


class ImagePrepStage(ABC):
    @abstractmethod
    def prep(self, image: np.ndarray) -> np.ndarray:
        pass

class ImageValueScalePrepStage(ImagePrepStage):
    def __init__(self, scale_value: float):
        self._scale_value = scale_value

    def prep(self, image: np.ndarray) -> np.ndarray:
        return image / self._scale_value

class Rgb2GrayscalePrepStage(ImagePrepStage):
    def __init__(self):
        self._channel_vals = np.array([0.2126, 0.7152, 0.0722])

    def prep(self, image: np.ndarray) -> np.ndarray:
        return np.dot(image[...,:], self._channel_vals)

class ImageResizePrepStage(ImagePrepStage):
    def __init__(self, new_size: Tuple[int, int]):
        self._new_size = new_size

    def prep(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self._new_size)

class ImageCropStage(ImagePrepStage):
    def __init__(self, new_size: Tuple[int, int], orient: str='center'):
        self._new_size = new_size
        self._orient = orient

    def prep(self, image: np.ndarray) -> np.ndarray:
        if self._orient == 'center':
            h = image.shape[0]
            w = image.shape[1]

            # Find out how many pixels we need to chop off of each dimension. 
            dh = max(0, h-self._new_size[0])
            dw = max(0, w-self._new_size[1])

            top = math.floor(dh/2)
            bot = top + self._new_size[0]

            left = math.floor(dw/2)
            right = left + self._new_size[1]

            print(f'{h}, {w}, {dh}, {dw}, {top}, {bot}, {left}, {right}')

            return image[top:bot, left:right]
        else:
            raise ValueError(f'Unsupported orientation: {self._orient}')

class ImagePrepPipeline():
    def __init__(self):
        self._pipeline = []

    def add_stage(self, stage: ImagePrepStage):
        self._pipeline.append(stage)

    def prep(self, image: np.ndarray) -> np.ndarray:
        rv = image

        for stage in self._pipeline:
            rv = stage.prep(rv)

        return rv
