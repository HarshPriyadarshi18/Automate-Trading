"""Three-layer outlier detection module for ExplainInvest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest


@dataclass
class OutlierDetectionResult:
    """Container for all three outlier layer outputs and final combined mask."""

    autoencoder_flag: np.ndarray
    isolation_flag: np.ndarray
    atr_flag: np.ndarray
    combined_flag: np.ndarray
    reconstruction_error: np.ndarray


class ThreeLayerOutlierDetector:
    """Detect anomalies using autoencoder, isolation forest, and ATR volatility checks."""

    def __init__(
        self,
        input_dim: int,
        contamination: float = 0.05,
        atr_window: int = 20,
        atr_multiplier: float = 2.0,
        ae_threshold_percentile: float = 95.0,
    ) -> None:
        """Initialize model components and thresholds for all outlier layers."""
        self.input_dim = input_dim
        self.contamination = contamination
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.ae_threshold_percentile = ae_threshold_percentile

        self.autoencoder = self._build_autoencoder(input_dim)
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=300,
        )
        self.reconstruction_threshold: float | None = None

    def _build_autoencoder(self, input_dim: int) -> tf.keras.Model:
        """Build a dense autoencoder with encoder and decoder sizes from the specification."""
        inp = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(256, activation="relu")(inp)
        latent = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dense(256, activation="relu")(latent)
        out = tf.keras.layers.Dense(input_dim, activation="linear")(x)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
        return model

    def fit(self, X_flat: np.ndarray, epochs: int = 40, batch_size: int = 64) -> None:
        """Fit autoencoder and isolation forest on nominal training samples."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ]

        self.autoencoder.fit(
            X_flat,
            X_flat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )

        reconstructed = self.autoencoder.predict(X_flat, verbose=0)
        rec_err = np.mean(np.square(X_flat - reconstructed), axis=1)
        self.reconstruction_threshold = float(np.percentile(rec_err, self.ae_threshold_percentile))

        self.isolation_forest.fit(X_flat)

    def detect(self, X_flat: np.ndarray, atr_values: np.ndarray) -> OutlierDetectionResult:
        """Run all three outlier checks and combine them using OR logic."""
        if self.reconstruction_threshold is None:
            raise RuntimeError("Detector must be fit before calling detect().")

        reconstructed = self.autoencoder.predict(X_flat, verbose=0)
        reconstruction_error = np.mean(np.square(X_flat - reconstructed), axis=1)
        ae_flag = reconstruction_error > self.reconstruction_threshold

        iso_pred = self.isolation_forest.predict(X_flat)
        iso_flag = iso_pred == -1

        atr_series = pd.Series(atr_values)
        atr_roll_mean = atr_series.rolling(window=self.atr_window, min_periods=1).mean().values
        atr_flag = atr_values > (self.atr_multiplier * atr_roll_mean)

        combined = ae_flag | iso_flag | atr_flag

        return OutlierDetectionResult(
            autoencoder_flag=ae_flag,
            isolation_flag=iso_flag,
            atr_flag=atr_flag,
            combined_flag=combined,
            reconstruction_error=reconstruction_error,
        )
