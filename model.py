"""CNN + LSTM + Attention model utilities for ExplainInvest."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


def build_cnn_lstm_attention_model(
    time_steps: int,
    num_features: int,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build the requested hybrid architecture for 3-class trading signal prediction."""
    inp = tf.keras.layers.Input(shape=(time_steps, num_features))

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)

    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn_out)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_trading_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    validation_split: float = 0.2,
    class_weight: Optional[Dict[int, float]] = None,
) -> tf.keras.callbacks.History:
    """Train the model with one-hot labels and early stopping to reduce overfitting."""
    y_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        y_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    return history


def predict_signals(
    model: tf.keras.Model, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict class probabilities, hard labels, and confidence values for each sample."""
    probs = model.predict(X, verbose=0)
    labels = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    return labels, probs, confidence