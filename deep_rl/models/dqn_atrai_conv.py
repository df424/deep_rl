
import tensorflow as tf

class AtariDQNQNet(tf.keras.Model):

    def __init__(self, num_actions: int):
        super(AtariDQNQNet, self).__init__()
        self._num_actions = num_actions

        self._conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), activation='relu')
        self._conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu')
        self._flatten = tf.keras.layers.Flatten()
        self._linear1 = tf.keras.layers.Dense(256, activation='relu')
        self._linear2 = tf.keras.layers.Dense(self._num_actions)

    def call(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._flatten(x)
        x = self._linear1(x)
        x = self._linear2(x)
        return x