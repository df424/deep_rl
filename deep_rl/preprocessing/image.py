
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.misc import imresize
import cv2


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
        cv2.resize(image, self._new_size)

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
