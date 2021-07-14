from typing import Tuple

import tensorflow as tf


def make_discriminator(input_units: int, n_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dropout(input_shape=(input_units,), rate=0.1),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(n_classes + 1),
        ]
    )


def make_generator(input_shape: Tuple[int], output_units: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, input_shape=input_shape),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(output_units),
        ]
    )


def make_baseline_classifier(input_shape: Tuple[int], n_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dropout(input_shape=input_shape, rate=0.1),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(n_classes),
        ]
    )
