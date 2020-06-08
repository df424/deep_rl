
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

        self._first = True

    def prep(self, observation: np.ndarray) -> np.ndarray:
        # Preprocess the image using the image pipeline.
        preped_obs = self._image_pipe.prep(observation)

        # If this is the first time weve been asked to update an observation
        # We need to do something about it being a single observation since the dqn
        # is expecting four stacked images we can just duplicate the observation four times..
        if self._first:
            self._buffer = np.repeat(preped_obs[np.newaxis,:,:], 4, axis=0)
            self._first = False

        # Overwrite the oldest observation.
        self._buffer[-1] = preped_obs

        # Roll the buffer.
        self._buffer = np.roll(self._buffer, shift=1, axis=0)

        # Finally return the buffer.
        return self._buffer