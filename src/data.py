from typing import List, Tuple

import colorama
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_text  # noqa
from colorama import Fore

from constants import (
    BATCH_SIZE,
    N_LABELED_CLASSES,
    RANDOM_SEED,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
)

colorama.init(autoreset=True)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def configure_dataset_for_performance(
    dataset: tf.data.Dataset,
    is_training=False,
) -> tf.data.Dataset:
    if is_training:
        dataset = dataset.shuffle(len(dataset))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


(ds, _), ds_info = tfds.load(
    "ag_news_subset",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def convert_to_tensors(dataset: tf.data.Dataset, samples: int) -> List[tf.Tensor]:
    X_y = list(zip(*list(tfds.as_numpy(ds))))
    return [tf.stack(list(i)[:samples], axis=0) for i in X_y]


X, y = convert_to_tensors(ds, 3_000)
y = tf.one_hot(y, depth=N_LABELED_CLASSES)


def apply_unlabeling_mask(X: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    unlabeled_mask = tf.transpose(
        tf.reshape(tf.random.categorical(tf.math.log([[0.1, 0.9]]), X.shape[0]), [-1])
    )

    y_samples = []
    for i in range(y.shape[0]):
        is_unlabeled = unlabeled_mask[i].numpy()
        if is_unlabeled:
            y_samples.append(
                tf.constant(
                    [0 for _ in range(N_LABELED_CLASSES)] + [1], dtype=tf.float32
                )
            )
        else:
            y_samples.append(
                tf.concat([y[i], tf.constant([0], dtype=tf.float32)], axis=-1)
            )

    y = tf.stack(y_samples, axis=0)
    return X, y


bert_preprocessing_model = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
)

X_test = bert_preprocessing_model(X[:TEST_SAMPLES])
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y[:TEST_SAMPLES]))
ds_test = configure_dataset_for_performance(ds_test)
start_idx = TEST_SAMPLES

X_val = bert_preprocessing_model(X[start_idx : start_idx + VAL_SAMPLES])
ds_val = tf.data.Dataset.from_tensor_slices(
    (X_val, y[start_idx : start_idx + VAL_SAMPLES])
)
ds_val = configure_dataset_for_performance(ds_val)

start_idx = start_idx + VAL_SAMPLES

X_train, y_train = apply_unlabeling_mask(
    X[start_idx : start_idx + TRAIN_SAMPLES], y[start_idx : start_idx + TRAIN_SAMPLES]
)
X_train = bert_preprocessing_model(X_train)
ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_train = configure_dataset_for_performance(ds_train, is_training=True)

if __name__ == "__main__":
    for x, y in ds_val.take(5):
        print(Fore.GREEN + "x:", x)
        print(Fore.BLUE + "y:", y)
        print()
