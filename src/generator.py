from typing import Tuple

import tensorflow as tf

from constants import BERT_POOLED_OUTPUT_DIMS


def generator(
    input_shape: Tuple[int],
) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, input_shape=input_shape),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(BERT_POOLED_OUTPUT_DIMS),
        ]
    )
