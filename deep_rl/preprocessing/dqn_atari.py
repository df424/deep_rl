
import numpy as np

from deep_rl.preprocessing.observation_preprocessor import ObservationPreprocessor
from deep_rl.preprocessing.image import ImagePrepPipeline, ImageValueScalePrepStage 
from deep_rl.preprocessing.image import ImageResizePrepStage, Rgb2GrayscalePrepStage, ImageCropStage

class DQNAtariPreprocessor(ObservationPreprocessor):
    def __init__(self):
        self._image_pipe = ImagePrepPipeline()
        self._image_pipe.add_stage(ImageResizePrepStage(new_size=(110, 84)))
        self._image_pipe.add_stage(ImageCropStage(new_size=(84, 84), orient='center'))
        self._image_pipe.add_stage(Rgb2GrayscalePrepStage())
        self._image_pipe.add_stage(ImageValueScalePrepStage(256))

    def prep(self, observation: np.ndarray) -> np.ndarray:
        return self._image_pipe.prep(observation)