from typing import Any, Dict, Optional, Tuple

import tensorflow as tf


def get_gaussian_latent_vector(
    batch_size: int, input_shape: Tuple[int, ...]
) -> tf.Tensor:
    return tf.random.normal(shape=(batch_size, *input_shape))


class FeatureMatchingError(tf.keras.losses.Loss):
    def call(self, X_real: tf.Tensor, X_fake: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(
            tf.square(tf.reduce_mean(X_real, axis=0) - tf.reduce_mean(X_fake, axis=0))
        )


class GAN(tf.keras.Model):
    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        latent_vector_size: int,
        name: Optional[str] = None,
    ) -> None:
        super(GAN, self).__init__(name=name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_vector_size = latent_vector_size

    def compile(
        self,
        g_optimizer: tf.keras.optimizers.Optimizer,
        d_optimizer: tf.keras.optimizers.Optimizer,
    ) -> None:
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        # g_losses and metrics
        self.g_loss_unsup = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, name="g_loss_unsup"
        )
        self.g_loss_feature_matching = FeatureMatchingError(
            name="g_loss_feature_matching"
        )
        self.g_loss_tracker = tf.keras.metrics.Mean("g_loss")

        # d losses and metrics
        self.d_loss_sup = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, name="d_loss_sup"
        )
        self.d_loss_unsup = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, name="d_loss_unsup"
        )
        self.d_loss_tracker = tf.keras.metrics.Mean("d_loss")

        # total loss
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        self.classifier_loss = tf.keras.metrics.CategoricalCrossentropy(
            from_logits=True, name="classifier_loss"
        )
        self.classifier_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name="classifier_accuracy"
        )

    @property
    def metrics(self):
        return [
            self.g_loss_tracker,
            self.d_loss_tracker,
            self.total_loss_tracker,
        ] + self.task_metrics

    def call(self, X: tf.Tensor) -> tf.Tensor:
        return self.discriminator(X)[:, :-1]

    def test_step(self, data: Tuple[Any, tf.Tensor]) -> Dict[str, int]:
        X, y = data
        y_logits = self(X, training=False)
        y_probs = tf.nn.softmax(y_logits)
        self.classifier_loss.update_state(y, y_probs)
        self.classifier_accuracy.update_state(y, y_probs)
        return {m.name: m.result() for m in self.metrics}

    @property
    def task_metrics(self):
        return [
            self.classifier_loss,
            self.classifier_accuracy,
        ]

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, float]:
        X, y = data

        batch_size = tf.shape(y)[0]
        latent_vector = tf.random.normal(shape=(batch_size, self.latent_vector_size))

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            # predictions
            fake_X = self.generator(latent_vector, training=True)
            fake_probs = self.discriminator(fake_X, training=True)
            prob_fake_is_fake = fake_probs[:, -1]
            real_probs = self.discriminator(X, training=True)
            y_pred = real_probs[:, :-1]

            # g losses
            g_loss_unsup = self.g_loss_unsup(
                tf.zeros_like(prob_fake_is_fake), prob_fake_is_fake
            )
            g_loss_feature_matching = self.g_loss_feature_matching(X, fake_X)
            g_loss = g_loss_unsup + g_loss_feature_matching
            self.g_loss_tracker.update_state(g_loss)

            # d losses
            d_loss_sup = self.d_loss_sup(y, y_pred)
            d_loss_unsup = self.d_loss_unsup(
                tf.ones_like(prob_fake_is_fake), prob_fake_is_fake
            )
            d_loss = d_loss_sup + d_loss_unsup
            self.d_loss_tracker.update_state(d_loss)

            self.total_loss_tracker.update_state(
                self.d_loss_tracker.result() + self.g_loss_tracker.result()
            )

            self.classifier_accuracy(y, y_pred)
            self.classifier_loss(y, y_pred)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )

        return {metric.name: metric.result() for metric in self.metrics}
