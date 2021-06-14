import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401

from constants import IMAGE_DIR, N_LABELED_CLASSES, SMALL_BERT


def get_bert() -> tf.keras.Model:
    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )
    encoder = hub.KerasLayer(SMALL_BERT, trainable=False, name="encoder")
    x = encoder(inputs)
    output = x["pooled_output"]
    return tf.keras.Model(inputs, output, name="prediction")


def get_bert_classifier(n_classes: int) -> tf.keras.Model:
    bert = get_bert()
    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )
    x = bert(inputs)
    output = tf.keras.layers.Dense(n_classes, activation="softmax", name="classifier")(
        x
    )
    return tf.keras.Model(inputs, output, name="prediction")


def export_model_summary() -> None:
    bert_classifier = get_bert_classifier(N_LABELED_CLASSES)
    bert_classifier.summary()

    bert_classifier_path = os.path.join(IMAGE_DIR, "bert_only.png")
    tf.keras.utils.plot_model(
        bert_classifier,
        bert_classifier_path,
        expand_nested=True,
        show_shapes=True,
    )
