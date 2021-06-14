from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from bert import get_bert
from constants import BERT_POOLED_OUTPUT_DIMS, EPSILON, LATENT_VECTOR_DIM, RANDOM_SEED
from discriminator import discriminator
from generator import generator

D_LR = 0.0004
G_LR = 0.0004

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def get_random_noise(z_dim: int, batch_size: int) -> tf.Tensor:
    return tf.random.uniform([batch_size, z_dim], minval=0, maxval=1)


def loss_G_feature_matching(bert_features: tf.Tensor, generator_features: tf.Tensor):
    return tf.reduce_mean(
        tf.square(
            tf.reduce_mean(bert_features, axis=0)
            - tf.reduce_mean(generator_features, axis=0)
        )
    )


def loss_G_unsupervised(D_fake_probs: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(tf.math.log(1 - D_fake_probs[:, -1] + EPSILON))


def loss_D_unsuperverised(
    D_real_probs: tf.Tensor, D_fake_probs: tf.Tensor
) -> tf.Tensor:
    unsupervised_1 = -tf.reduce_mean(tf.math.log(1 - D_real_probs[:, -1] + EPSILON))
    unsupervised_2 = -tf.reduce_mean(tf.math.log(D_fake_probs[:, -1] + EPSILON))
    return unsupervised_1 + unsupervised_2


def loss_D_supervised(y: tf.Tensor, D_real_logits: tf.Tensor) -> tf.Tensor:
    filtered_logits = D_real_logits[:, :-1]
    real_log_probs = tf.nn.log_softmax(filtered_logits, axis=-1)
    per_example_loss = -tf.math.reduce_sum(y[:, :-1] * real_log_probs, axis=-1)
    # don't consider the unlabeled examples when computing supervised loss
    is_unlabeled = y[:, -1]
    is_labeled = ~tf.cast(is_unlabeled, dtype=tf.bool)

    tf.boolean_mask(per_example_loss, is_labeled)
    return tf.math.reduce_mean(per_example_loss)


# optimizers
g_optimizer = tf.keras.optimizers.Adam(G_LR)
d_optimizer = tf.keras.optimizers.Adam(D_LR)


# metrics
supervised_loss_tracker = tf.metrics.Mean(name="supervised_loss")
g_loss_tracker = tf.metrics.Mean(name="g_loss")
g_loss_u_tracker = tf.metrics.Mean(name="g_loss_u")
g_loss_feature_matching_tracker = tf.metrics.Mean(name="g_loss_feature_matching")
d_loss_tracker = tf.metrics.Mean(name="d_loss")
accuracy_tracker = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
categorical_crossentropy_tracker = tf.keras.metrics.CategoricalCrossentropy(
    name="categorical_crossentropy",
    from_logits=False,
)


class GanBert(tf.keras.Model):
    def __init__(self) -> None:
        super(GanBert, self).__init__()
        self.latent_vector_dim = LATENT_VECTOR_DIM
        self.B = get_bert()
        self.G = generator((self.latent_vector_dim,))
        self.D = discriminator((BERT_POOLED_OUTPUT_DIMS,))

    def compile(self, d_optimizer, g_optimizer) -> None:
        super(GanBert, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return [
            supervised_loss_tracker,
            g_loss_tracker,
            g_loss_u_tracker,
            g_loss_feature_matching_tracker,
            d_loss_tracker,
            accuracy_tracker,
            categorical_crossentropy_tracker,
        ]

    def call(self, inputs: Any) -> tf.Tensor:
        x = self.B(inputs)
        y_pred = self.D(x)
        return tf.nn.softmax(y_pred[:, :-1])

    def test_step(self, data: Tuple[Any, tf.Tensor]) -> Dict[str, int]:
        X, y = data
        y_probs = self(X, training=False)
        accuracy_tracker.update_state(y, y_probs)
        categorical_crossentropy_tracker.update_state(y, y_probs)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        X, y = data
        batch_size = tf.shape(y)[0]

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_vector_dim)
        )
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            real_bert_encoding = self.B(X, training=True)
            fake_bert_encoding = self.G(random_latent_vectors, training=True)

            combined_encodings = tf.concat(
                [real_bert_encoding, fake_bert_encoding], axis=0
            )

            y_logits = self.D(combined_encodings, training=True)
            y_probs = tf.nn.softmax(y_logits)
            D_real_logits, D_fake_logits = tf.split(
                y_logits, [batch_size, tf.shape(y_probs)[0] - batch_size], axis=0
            )
            D_real_probs, D_fake_probs = tf.split(
                y_probs, [batch_size, tf.shape(y_probs)[0] - batch_size], axis=0
            )

            g_loss_u = loss_G_unsupervised(D_fake_probs)
            g_loss_feature_matching = loss_G_feature_matching(
                real_bert_encoding, fake_bert_encoding
            )
            g_loss = g_loss_u + g_loss_feature_matching

            D_loss_s = loss_D_supervised(y, D_real_logits)
            D_loss_u = loss_D_unsuperverised(D_real_probs, D_fake_probs)
            d_loss = D_loss_s + D_loss_u

        supervised_loss_tracker.update_state(D_loss_s)
        g_loss_tracker.update_state(g_loss)
        g_loss_u_tracker.update_state(g_loss_u),
        g_loss_feature_matching_tracker(g_loss_feature_matching),
        d_loss_tracker.update_state(d_loss)
        accuracy_tracker.update_state(y[:, :-1], D_real_probs[:, :-1])
        categorical_crossentropy_tracker.update_state(y[:, :-1], D_real_probs[:, :-1])

        g_gradients = g_tape.gradient(g_loss, self.G.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.D.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.D.trainable_variables))

        return {m.name: m.result() for m in self.metrics}
