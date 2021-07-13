import random

import numpy as np
import tensorflow as tf

from constants import RANDOM_SEED

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import os  # noqa: E402

import colorama  # noqa: E402
import pandas as pd  # noqa: E402
from colorama import Fore  # noqa: E402
from plot_keras_history import plot_history  # noqa: E402

from constants import RESULTS_DIR, SAVED_MODEL_DIR  # noqa: E402
from data import test_ds, train_ds, val_ds  # noqa: E402
from ganbert import GANBERT  # noqa: E402
from model_components import make_baseline_classifier  # noqa: E402
from model_components import make_discriminator, make_generator  # noqa: E402
from training_config import AG_NEWS_NUM_LABELED_CLASSES  # noqa: E402
from training_config import BERT_POOLED_OUTPUT_DIMS  # noqa: E402

colorama.init(autoreset=True)


def post_fit(model: tf.keras.Model, results, history, loss_name: str) -> None:
    print()
    print(
        Fore.GREEN + f"Test set results after {len(history.history[loss_name])} epochs:"
    )
    metric_names = [metric.name for metric in model.metrics]
    for metric_name, result in zip(metric_names, results):
        print(Fore.BLUE + f" * {metric_name}:", result)
        print()

    fig_path = os.path.join(RESULTS_DIR, f"{model.name}_training.png")
    plot_history(history, path=fig_path)
    print(f"Saved {model.name} training figures to {fig_path}.")

    scores_path = os.path.join(RESULTS_DIR, f"{model.name}_scores.csv")
    scores = dict(zip(metric_names, results))
    df = pd.DataFrame.from_dict(scores, orient="index")
    df.to_csv(scores_path)
    print(f"Saved {model.name} scores to {scores_path}.")

    weights_path = os.path.join(SAVED_MODEL_DIR, model.name)
    model.save_weights(weights_path)
    print(f"Saved {model.name} weights to {weights_path}.")
    print()


def fit_baseline_model(use_saved_weights=False) -> None:
    model = make_baseline_classifier(
        input_shape=(BERT_POOLED_OUTPUT_DIMS,), n_classes=AG_NEWS_NUM_LABELED_CLASSES
    )
    baseline_path = os.path.join(SAVED_MODEL_DIR, model.name)
    if use_saved_weights:
        model.load_weights(baseline_path)
    else:
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1000,
            callbacks=[
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                ),
            ],
        )
    results = model.evaluate(test_ds)
    post_fit(model, results, history, "loss")


def fit_ganbert(use_saved_weights=False) -> None:
    LATENT_VECTOR_SIZE = 100
    generator = make_generator(
        input_shape=(LATENT_VECTOR_SIZE,), output_units=BERT_POOLED_OUTPUT_DIMS
    )
    discriminator = make_discriminator(
        input_units=BERT_POOLED_OUTPUT_DIMS, n_classes=AG_NEWS_NUM_LABELED_CLASSES
    )
    ganbert = GANBERT(
        generator=generator,
        discriminator=discriminator,
        latent_vector_size=LATENT_VECTOR_SIZE,
        name="ganbert",
    )
    ganbert_path = os.path.join(SAVED_MODEL_DIR, ganbert.name)
    if use_saved_weights:
        ganbert.load_weights(ganbert_path)
    else:
        g_optim = tf.keras.optimizers.Adam(1e-4)
        d_optim = tf.keras.optimizers.Adam(1e-4)
        ganbert.compile(
            g_optimizer=g_optim,
            d_optimizer=d_optim,
        )

        history = ganbert.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1000,
            callbacks=[
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_classifier_loss",
                    patience=3,
                    restore_best_weights=True,
                ),
            ],
        )
    results = ganbert.evaluate(test_ds)
    post_fit(ganbert, results, history, "val_classifier_loss")


def main() -> None:
    fit_baseline_model()
    fit_ganbert()


if __name__ == "__main__":
    main()
