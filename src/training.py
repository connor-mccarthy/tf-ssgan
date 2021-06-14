import os
from datetime import datetime

import tensorflow as tf
from plot_keras_history import plot_history
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from constants import SAVED_MODELS_DIR
from data import ds_test, ds_train, ds_val
from ganbert import GanBert


def train() -> None:
    model = GanBert()  # type: ignore
    model.compile(
        g_optimizer=tf.keras.optimizers.Adam(0.0004),
        d_optimizer=tf.keras.optimizers.Adam(0.0004),
    )
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(
            monitor="val_supervised_loss", patience=3, restore_best_weights=True
        ),
        TerminateOnNaN(),
    ]
    fit_kwargs = dict(epochs=10, callbacks=callbacks, verbose=1)
    history = model.fit(ds_train, validation_data=ds_val, **fit_kwargs)
    plot_history(history.history)

    scores = model.evaluate(ds_test)
    metric_dict = dict(zip(model.metrics_names, scores))
    print("Test metrics:", metric_dict)

    saved_model_path = os.path.join(SAVED_MODELS_DIR, "trained-bert")
    model.save(saved_model_path)
    print("Saved model to", saved_model_path)


if __name__ == "__main__":
    train()
