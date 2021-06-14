from typing import Tuple

import tensorflow as tf

from constants import N_LABELED_CLASSES


def discriminator(
    input_shape: Tuple[int],
) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dropout(input_shape=input_shape, rate=0.1),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(N_LABELED_CLASSES + 1),
        ]
    )
