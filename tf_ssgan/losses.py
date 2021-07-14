import tensorflow as tf


class FeatureMatchingError(tf.keras.losses.Loss):
    def call(self, X_real: tf.Tensor, X_fake: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(
            tf.square(tf.reduce_mean(X_real, axis=0) - tf.reduce_mean(X_fake, axis=0))
        )
