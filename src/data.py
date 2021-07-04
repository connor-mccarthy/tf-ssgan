from typing import List

import colorama
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401
from colorama import Fore

from constants import RANDOM_SEED, SMALL_BERT
from training_config import (
    ADJUSTED_TOTAL_SAMPLES,
    AG_NEWS_NUM_LABELED_CLASSES,
    BATCH_SIZE,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
)

colorama.init(autoreset=True)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def convert_dataset_to_tensors(
    dataset: tf.data.Dataset, samples: int
) -> List[tf.Tensor]:
    X_y = list(zip(*list(tfds.as_numpy(dataset))))
    return [tf.stack(list(i)[:samples], axis=0) for i in X_y]


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


bert_preprocessing_model = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
)
bert = get_bert()


@tf.function
def apply_bert(X: tf.Tensor) -> tf.Tensor:
    return bert(bert_preprocessing_model(X))


(ds1, ds2), ds_info = tfds.load(
    "ag_news_subset",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
total_samples = (
    ds_info.splits["train"].num_examples + ds_info.splits["test"].num_examples
)
assert (
    total_samples >= ADJUSTED_TOTAL_SAMPLES
), f"Total size of train_ds, test_ds, and val_ds must be less than full size of dataset: {total_samples}."

ds = ds1.concatenate(ds2)
X, y = convert_dataset_to_tensors(ds, len(ds))
y = tf.one_hot(y, depth=AG_NEWS_NUM_LABELED_CLASSES)
X = X[:ADJUSTED_TOTAL_SAMPLES]
y = y[:ADJUSTED_TOTAL_SAMPLES]

X_train, X_val, X_test = tf.split(
    X,
    [TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES],
    axis=0,
)
y_train, y_val, y_test = tf.split(
    y,
    [TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES],
    axis=0,
)


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_ds = (
    train_ds.batch(BATCH_SIZE)
    .map(lambda X, y: (apply_bert(X), y))
    .cache()
    .shuffle(len(train_ds))
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    test_ds.batch(BATCH_SIZE)
    .map(lambda X, y: (apply_bert(X), y))
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    val_ds.batch(BATCH_SIZE)
    .map(lambda X, y: (apply_bert(X), y))
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

if __name__ == "__main__":
    for x, y in train_ds.take(5):
        print(Fore.GREEN + "x:", x)
        print(Fore.BLUE + "y:")
        print()
