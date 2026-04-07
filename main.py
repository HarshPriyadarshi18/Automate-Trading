"""Main pipeline runner for ExplainInvest."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_collection import FEATURE_COLUMNS, collect_and_preprocess_data
from explainability import ExplainabilityEngine, format_top_features
from labelling import generate_pseudo_labels
from model import build_cnn_lstm_attention_model, predict_signals, train_trading_model
from outlier_detection import ThreeLayerOutlierDetector
from portfolio import allocate_portfolio, compute_asset_correlation


SIGNAL_NAME = {0: "BUY", 1: "SELL", 2: "HOLD"}

# Runtime-focused defaults to keep end-to-end execution practical on local machines.
DATA_START = "2021-01-01"
DATA_END = "2024-12-31"
TIME_STEPS = 32
OUTLIER_FIT_EPOCHS = 12
OUTLIER_FIT_BATCH = 128
KMEANS_CLUSTERS = 12
MODEL_EPOCHS = 10
SHAP_BACKGROUND_MAX = 20
SHAP_NSAMPLES = 30
CORR_LOOKBACK = 200


def set_reproducibility(seed: int = 42) -> None:
    """Set random seeds for reproducibility across NumPy and TensorFlow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def split_train_test_by_asset(dataset: dict, train_ratio: float = 0.8) -> dict:
    """Build train/test arrays while preserving chronological order within each asset."""
    X_train_list: List[np.ndarray] = []
    X_test_list: List[np.ndarray] = []
    current_train_list: List[np.ndarray] = []
    current_test_list: List[np.ndarray] = []
    next_train_list: List[np.ndarray] = []
    next_test_list: List[np.ndarray] = []
    atr_train_list: List[np.ndarray] = []
    atr_test_list: List[np.ndarray] = []

    for asset_name, asset_data in dataset["assets"].items():
        n = len(asset_data.X)
        split_idx = min(n - 1, max(1, int(n * train_ratio)))

        X_train_list.append(asset_data.X[:split_idx])
        X_test_list.append(asset_data.X[split_idx:])

        current_train_list.append(asset_data.current_close[:split_idx])
        current_test_list.append(asset_data.current_close[split_idx:])
        next_train_list.append(asset_data.next_close[:split_idx])
        next_test_list.append(asset_data.next_close[split_idx:])
        atr_train_list.append(asset_data.atr[:split_idx])
        atr_test_list.append(asset_data.atr[split_idx:])

    return {
        "X_train": np.concatenate(X_train_list, axis=0),
        "X_test": np.concatenate(X_test_list, axis=0),
        "current_train": np.concatenate(current_train_list, axis=0),
        "current_test": np.concatenate(current_test_list, axis=0),
        "next_train": np.concatenate(next_train_list, axis=0),
        "next_test": np.concatenate(next_test_list, axis=0),
        "atr_train": np.concatenate(atr_train_list, axis=0),
        "atr_test": np.concatenate(atr_test_list, axis=0),
    }


def prepare_asset_latest_inputs(dataset: dict) -> Dict[str, dict]:
    """Extract latest sequence and latest feature row per asset for live-style inference."""
    result = {}
    for asset_name, asset_data in dataset["assets"].items():
        latest_sequence = asset_data.X[-1:]
        latest_atr = float(asset_data.atr[-1])
        latest_features = {
            col: float(asset_data.raw_df[col].iloc[-1]) for col in FEATURE_COLUMNS
        }
        result[asset_name] = {
            "X_latest": latest_sequence,
            "latest_atr": latest_atr,
            "latest_features": latest_features,
        }
    return result


def run_pipeline_collect() -> dict:
    """Execute ExplainInvest and return structured results for downstream UIs/APIs."""
    set_reproducibility(42)

    dataset = collect_and_preprocess_data(start=DATA_START, end=DATA_END, time_steps=TIME_STEPS)
    split_data = split_train_test_by_asset(dataset, train_ratio=0.8)

    X_train = split_data["X_train"]

    detector = ThreeLayerOutlierDetector(
        input_dim=X_train.shape[1] * X_train.shape[2],
        contamination=0.05,
    )
    detector.fit(
        X_train.reshape(X_train.shape[0], -1),
        epochs=OUTLIER_FIT_EPOCHS,
        batch_size=OUTLIER_FIT_BATCH,
    )

    train_outliers = detector.detect(
        X_train.reshape(X_train.shape[0], -1),
        split_data["atr_train"],
    )

    y_train, _, _ = generate_pseudo_labels(
        X=X_train,
        current_close=split_data["current_train"],
        next_close=split_data["next_train"],
        clean_mask=~train_outliers.combined_flag,
        n_clusters=KMEANS_CLUSTERS,
        move_threshold=0.00005,
    )

    clean_train_mask = ~train_outliers.combined_flag
    X_train_clean = X_train[clean_train_mask]
    y_train_clean = y_train[clean_train_mask]

    model = build_cnn_lstm_attention_model(
        time_steps=X_train.shape[1],
        num_features=X_train.shape[2],
        learning_rate=1e-3,
    )

    unique_classes = np.unique(y_train_clean)
    cw_values = compute_class_weight("balanced", classes=unique_classes, y=y_train_clean)
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, cw_values)}
    for missing_class in [0, 1, 2]:
        if missing_class not in class_weight_dict:
            class_weight_dict[missing_class] = 1.0

    train_trading_model(
        model=model,
        X_train=X_train_clean,
        y_train=y_train_clean,
        epochs=MODEL_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
    )

    latest_inputs = prepare_asset_latest_inputs(dataset)

    asset_signals: Dict[str, int] = {}
    asset_confidence: Dict[str, float] = {}
    asset_volatility: Dict[str, float] = {}
    asset_outlier: Dict[str, bool] = {}
    per_asset_top_features: Dict[str, List[str]] = {}

    background_count = min(SHAP_BACKGROUND_MAX, len(X_train_clean))
    if background_count < 5:
        raise RuntimeError("Not enough clean training samples for SHAP background.")
    background_data = X_train_clean[
        np.random.choice(len(X_train_clean), size=background_count, replace=False)
    ]
    explainer = ExplainabilityEngine(model, background_data, FEATURE_COLUMNS)

    for asset_name, payload in latest_inputs.items():
        X_latest = payload["X_latest"]
        latest_atr = payload["latest_atr"]

        asset_full = dataset["assets"][asset_name]
        latest_flat = asset_full.X[-1:].reshape(1, -1)
        latest_atr_arr = np.asarray([asset_full.atr[-1]], dtype=np.float32)
        asset_outlier_result = detector.detect(latest_flat, latest_atr_arr)
        last_outlier_flag = bool(asset_outlier_result.combined_flag[0])

        pred_label, _, pred_conf = predict_signals(model, X_latest)
        signal = int(pred_label[0])
        confidence = float(pred_conf[0])

        asset_signals[asset_name] = signal
        asset_confidence[asset_name] = confidence
        asset_volatility[asset_name] = latest_atr
        asset_outlier[asset_name] = last_outlier_flag

        top_features = explainer.explain_single_trade(
            X_single=X_latest,
            predicted_class=signal,
            latest_feature_values=payload["latest_features"],
            top_k=5,
            nsamples=SHAP_NSAMPLES,
        )
        per_asset_top_features[asset_name] = format_top_features(
            top_features, SIGNAL_NAME[signal]
        )

    close_series_map = {
        asset_name: dataset["assets"][asset_name].raw_df["Close"]
        for asset_name in dataset["assets"].keys()
    }
    corr_matrix = compute_asset_correlation(close_series_map, lookback=CORR_LOOKBACK)

    weights = allocate_portfolio(
        asset_signals=asset_signals,
        asset_confidence=asset_confidence,
        asset_volatility=asset_volatility,
        asset_outlier_flag=asset_outlier,
        correlation_matrix=corr_matrix,
        corr_threshold=0.8,
    )

    rows = []
    for asset_name in dataset["assets"].keys():
        rows.append(
            {
                "asset": asset_name,
                "signal": SIGNAL_NAME[asset_signals[asset_name]],
                "signal_idx": int(asset_signals[asset_name]),
                "confidence": float(asset_confidence[asset_name]),
                "portfolio_weight": float(weights.get(asset_name, 0.0)),
                "outlier_flagged": bool(asset_outlier[asset_name]),
                "top_features": per_asset_top_features[asset_name],
            }
        )

    rows.sort(key=lambda x: x["portfolio_weight"], reverse=True)

    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "assets": rows,
        "class_weights": class_weight_dict,
        "meta": {
            "time_steps": int(dataset["time_steps"]),
            "asset_count": int(len(rows)),
            "train_samples": int(len(X_train_clean)),
        },
    }


def run_pipeline() -> None:
    """Execute ExplainInvest from data fetch to signal, allocation, and explainability output."""
    result = run_pipeline_collect()

    class_weights = result["class_weights"]
    print(
        f"Class weights → BUY={class_weights.get(0, 1.0):.2f}, "
        f"SELL={class_weights.get(1, 1.0):.2f}, HOLD={class_weights.get(2, 1.0):.2f}"
    )
    print("[7/7] Final ExplainInvest output")

    for row in result["assets"]:
        print("-" * 70)
        print(f"Asset:            {row['asset']}")
        print(f"Signal:           {row['signal']}")
        print(f"Confidence:       {row['confidence'] * 100.0:.2f}%")
        print(f"Portfolio Weight: {row['portfolio_weight'] * 100.0:.2f}%")
        print(f"Outlier Flagged:  {'Yes' if row['outlier_flagged'] else 'No'}")
        print("Top 5 SHAP Features:")
        for line in row["top_features"]:
            print(f"  {line}")


if __name__ == "__main__":
    run_pipeline()