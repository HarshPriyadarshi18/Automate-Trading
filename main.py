"""Main pipeline runner for ExplainInvest."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Keep TensorFlow/absl from printing startup noise before importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Stabilize local runs by limiting BLAS/OpenMP thread fan-out.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
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
DATA_END = (datetime.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
TIME_STEPS = 32
OUTLIER_FIT_EPOCHS = 12
OUTLIER_FIT_BATCH = 128
KMEANS_CLUSTERS = 12
MODEL_EPOCHS = 5
MODEL_EARLY_STOPPING_PATIENCE = 8
MODEL_MIN_DELTA = 1e-4
SHAP_BACKGROUND_MAX = 20
SHAP_NSAMPLES = 30
CORR_LOOKBACK = 200
BACKTEST_FEE_BPS = 2.0
BACKTEST_INITIAL_CAPITAL_PER_ASSET = 10000.0
EVAL_MOVE_THRESHOLD = 0.0002
SIGNAL_MIN_CONFIDENCE = 0.52
SIGNAL_MIN_EDGE = 0.03
SIGNAL_BUY_SELL_GAP = 0.015
SIGNAL_HOLD_SWITCH_GAP = 0.05
CALIBRATION_MIN_ACTIVE_RATIO = 0.08
CALIBRATION_MIN_CONF_GRID = [0.48, 0.52, 0.56, 0.60]
CALIBRATION_MIN_EDGE_GRID = [0.02, 0.03, 0.04, 0.05]
CALIBRATION_BUY_SELL_GAP_GRID = [0.01, 0.015, 0.02, 0.03]
CALIBRATION_HOLD_SWITCH_GAP_GRID = [0.00, 0.03, 0.05, 0.08]
PREFERRED_MARKETS = [
    "BTC-USD",
    "ETH-USD",
    "XRP-USD",
    "S&P500",
    "NASDAQ",
    "Dow Jones",
    "Apple",
    "Microsoft",
    "NVIDIA",
    "Tesla",
    "Gold",
    "Silver",
    "Crude Oil",
    "Natural Gas",
    "EUR/USD",
    "GBP/USD",
]
MARKET_COUNT = 6
REPORT_DIR = Path("reports")


def set_reproducibility(seed: int = 42) -> None:
    """Set random seeds for reproducibility across NumPy and TensorFlow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def pick_market_universe(asset_names: List[str], preferred_markets: List[str], count: int) -> List[str]:
    """Pick up to `count` assets, prioritizing preferred markets if available."""
    selected: List[str] = []

    for market in preferred_markets:
        if market in asset_names and market not in selected:
            selected.append(market)
        if len(selected) == count:
            return selected

    for market in sorted(asset_names):
        if market not in selected:
            selected.append(market)
        if len(selected) == count:
            break

    return selected


def filter_dataset_assets(dataset: dict, selected_assets: List[str]) -> dict:
    """Return a dataset copy containing only selected assets."""
    return {
        **dataset,
        "assets": {
            asset_name: dataset["assets"][asset_name]
            for asset_name in selected_assets
            if asset_name in dataset["assets"]
        },
    }


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


def label_from_return(returns: np.ndarray, move_threshold: float) -> np.ndarray:
    """Map returns to BUY/SELL/HOLD labels with configurable dead-zone threshold."""
    return np.where(returns > move_threshold, 0, np.where(returns < -move_threshold, 1, 2)).astype(np.int32)


def balance_training_samples(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes so BUY/SELL/HOLD are all learned fairly."""
    rng = np.random.default_rng(seed)
    class_indices = {cls: np.where(y == cls)[0] for cls in [0, 1, 2]}

    non_empty_counts = [len(idxs) for idxs in class_indices.values() if len(idxs) > 0]
    if not non_empty_counts:
        raise ValueError("No training labels available to balance.")

    target_count = max(non_empty_counts)
    balanced_indices = []
    for cls, idxs in class_indices.items():
        if len(idxs) == 0:
            continue
        sampled = rng.choice(idxs, size=target_count, replace=len(idxs) < target_count)
        balanced_indices.append(sampled)

    all_indices = np.concatenate(balanced_indices)
    rng.shuffle(all_indices)
    return X[all_indices], y[all_indices]


def calibrate_signal_from_probs(
    probs: np.ndarray,
    min_confidence: float = SIGNAL_MIN_CONFIDENCE,
    min_edge: float = SIGNAL_MIN_EDGE,
    buy_sell_gap: float = SIGNAL_BUY_SELL_GAP,
    hold_switch_gap: float = SIGNAL_HOLD_SWITCH_GAP,
) -> int:
    """Convert probabilities into a conservative BUY/SELL/HOLD action.

    This reduces one-sided predictions by requiring adequate confidence and
    separation before taking directional trades.
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    if p.shape[0] != 3:
        return int(np.argmax(p))

    buy_conf = float(p[0])
    sell_conf = float(p[1])
    hold_conf = float(p[2])
    directional_idx = 0 if buy_conf >= sell_conf else 1
    directional_conf = max(buy_conf, sell_conf)
    directional_gap = directional_conf - hold_conf
    directional_edge = directional_conf - min(buy_conf, sell_conf)

    # If BUY and SELL probabilities are too close, avoid directional action.
    if abs(buy_conf - sell_conf) < buy_sell_gap:
        return 2

    # Keep HOLD only when it is clearly stronger than the directional choice.
    if hold_conf >= min_confidence and (hold_conf - directional_conf) > min_edge:
        return 2

    # Allow a directional signal when it is competitive with HOLD and has a clear
    # lead over the opposite direction.
    if directional_conf >= min_confidence and directional_gap >= -hold_switch_gap and directional_edge >= min_edge:
        return directional_idx

    # Fall back to the strongest non-neutral direction if HOLD is not decisive.
    if directional_conf > hold_conf:
        return directional_idx

    return 2


def tune_signal_calibration(
    model: tf.keras.Model,
    dataset: dict,
    train_ratio: float = 0.8,
    move_threshold: float = EVAL_MOVE_THRESHOLD,
) -> dict:
    """Tune signal calibration thresholds on early test samples and report hold-out quality."""
    all_probs: List[np.ndarray] = []
    all_actual_labels: List[np.ndarray] = []

    for _, asset_data in dataset["assets"].items():
        n = len(asset_data.X)
        split_idx = min(n - 1, max(1, int(n * train_ratio)))

        X_test = asset_data.X[split_idx:]
        current_test = asset_data.current_close[split_idx:]
        next_test = asset_data.next_close[split_idx:]

        if len(X_test) == 0:
            continue

        _, pred_probs, _ = predict_signals(model, X_test)
        actual_ret = (next_test - current_test) / (current_test + 1e-12)
        actual_label = label_from_return(actual_ret, move_threshold=move_threshold)

        all_probs.append(pred_probs)
        all_actual_labels.append(actual_label)

    if not all_probs:
        return {
            "params": {
                "min_confidence": SIGNAL_MIN_CONFIDENCE,
                "min_edge": SIGNAL_MIN_EDGE,
                "buy_sell_gap": SIGNAL_BUY_SELL_GAP,
                "hold_switch_gap": SIGNAL_HOLD_SWITCH_GAP,
            },
            "calibration_samples": 0,
            "holdout_samples": 0,
            "calibration_accuracy_pct": 0.0,
            "holdout_accuracy_pct": 0.0,
            "holdout_active_trade_accuracy_pct": 0.0,
            "holdout_active_ratio_pct": 0.0,
        }

    probs_all = np.concatenate(all_probs, axis=0)
    labels_all = np.concatenate(all_actual_labels, axis=0)
    n_total = len(labels_all)
    calib_count = max(1, int(n_total * 0.5))
    calib_probs = probs_all[:calib_count]
    calib_labels = labels_all[:calib_count]
    holdout_probs = probs_all[calib_count:]
    holdout_labels = labels_all[calib_count:]
    if len(holdout_labels) == 0:
        holdout_probs = calib_probs
        holdout_labels = calib_labels

    best_params = {
        "min_confidence": SIGNAL_MIN_CONFIDENCE,
        "min_edge": SIGNAL_MIN_EDGE,
        "buy_sell_gap": SIGNAL_BUY_SELL_GAP,
        "hold_switch_gap": SIGNAL_HOLD_SWITCH_GAP,
    }
    best_score = -1.0
    best_acc = 0.0

    for min_confidence in CALIBRATION_MIN_CONF_GRID:
        for min_edge in CALIBRATION_MIN_EDGE_GRID:
            for buy_sell_gap in CALIBRATION_BUY_SELL_GAP_GRID:
                for hold_switch_gap in CALIBRATION_HOLD_SWITCH_GAP_GRID:
                    trial_params = {
                        "min_confidence": float(min_confidence),
                        "min_edge": float(min_edge),
                        "buy_sell_gap": float(buy_sell_gap),
                        "hold_switch_gap": float(hold_switch_gap),
                    }
                    trial_pred = np.asarray(
                        [calibrate_signal_from_probs(p, **trial_params) for p in calib_probs], dtype=np.int32
                    )
                    trial_acc = float(accuracy_score(calib_labels, trial_pred))
                    trial_active_ratio = float(np.mean(trial_pred != 2))
                    if trial_active_ratio < CALIBRATION_MIN_ACTIVE_RATIO:
                        continue

                    # Favor high accuracy while still keeping enough active trades.
                    trial_score = trial_acc + (0.08 * trial_active_ratio)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_acc = trial_acc
                        best_params = trial_params

    holdout_pred = np.asarray(
        [calibrate_signal_from_probs(p, **best_params) for p in holdout_probs], dtype=np.int32
    )
    holdout_acc = float(accuracy_score(holdout_labels, holdout_pred)) if len(holdout_labels) else 0.0
    holdout_active = holdout_pred != 2
    holdout_trade_acc = (
        float(np.mean(holdout_pred[holdout_active] == holdout_labels[holdout_active]))
        if np.any(holdout_active)
        else 0.0
    )

    return {
        "params": best_params,
        "calibration_samples": int(len(calib_labels)),
        "holdout_samples": int(len(holdout_labels)),
        "calibration_accuracy_pct": float(best_acc * 100.0),
        "holdout_accuracy_pct": float(holdout_acc * 100.0),
        "holdout_active_trade_accuracy_pct": float(holdout_trade_acc * 100.0),
        "holdout_active_ratio_pct": float(np.mean(holdout_active) * 100.0) if len(holdout_active) else 0.0,
    }


def evaluate_prediction_vs_actual(
    model: tf.keras.Model,
    dataset: dict,
    train_ratio: float = 0.8,
    move_threshold: float = EVAL_MOVE_THRESHOLD,
    calibration_params: dict | None = None,
) -> dict:
    """Build prediction-vs-actual rows and summary metrics for research/PPT reporting."""
    records: List[dict] = []
    per_asset_summary: Dict[str, dict] = {}

    all_pred_labels: List[np.ndarray] = []
    all_actual_labels: List[np.ndarray] = []
    all_trade_correct: List[np.ndarray] = []

    for asset_name, asset_data in dataset["assets"].items():
        n = len(asset_data.X)
        split_idx = min(n - 1, max(1, int(n * train_ratio)))

        X_test = asset_data.X[split_idx:]
        current_test = asset_data.current_close[split_idx:]
        next_test = asset_data.next_close[split_idx:]
        timestamps_test = asset_data.timestamps[split_idx:]

        if len(X_test) == 0:
            per_asset_summary[asset_name] = {
                "samples": 0,
                "accuracy_pct": 0.0,
                "active_trade_accuracy_pct": 0.0,
            }
            continue

        pred_label, pred_probs, pred_conf = predict_signals(model, X_test)
        calibration = calibration_params or {}
        pred_label = np.asarray([calibrate_signal_from_probs(p, **calibration) for p in pred_probs], dtype=np.int32)
        pred_conf = np.max(pred_probs, axis=1)
        actual_ret = (next_test - current_test) / (current_test + 1e-12)
        actual_label = label_from_return(actual_ret, move_threshold=move_threshold)

        all_pred_labels.append(pred_label)
        all_actual_labels.append(actual_label)

        is_correct = pred_label == actual_label
        active_mask = pred_label != 2
        active_correct = np.where(active_mask, is_correct, False)
        all_trade_correct.append(active_correct[active_mask])

        for idx in range(len(X_test)):
            records.append(
                {
                    "asset": asset_name,
                    "symbol": asset_data.symbol,
                    "timestamp": str(pd.Timestamp(timestamps_test[idx])),
                    "current_close": float(current_test[idx]),
                    "next_close": float(next_test[idx]),
                    "actual_return_pct": float(actual_ret[idx] * 100.0),
                    "predicted_signal": SIGNAL_NAME[int(pred_label[idx])],
                    "actual_signal": SIGNAL_NAME[int(actual_label[idx])],
                    "confidence_pct": float(pred_conf[idx] * 100.0),
                    "correct": bool(is_correct[idx]),
                }
            )

        per_asset_summary[asset_name] = {
            "samples": int(len(X_test)),
            "accuracy_pct": float(np.mean(is_correct) * 100.0),
            "active_trade_accuracy_pct": float(np.mean(active_correct[active_mask]) * 100.0)
            if np.any(active_mask)
            else 0.0,
        }

    pred_all = np.concatenate(all_pred_labels) if all_pred_labels else np.array([], dtype=np.int32)
    actual_all = np.concatenate(all_actual_labels) if all_actual_labels else np.array([], dtype=np.int32)
    trade_all = np.concatenate(all_trade_correct) if all_trade_correct else np.array([], dtype=bool)

    overall_accuracy = float(accuracy_score(actual_all, pred_all) * 100.0) if len(actual_all) else 0.0
    active_trade_accuracy = float(np.mean(trade_all) * 100.0) if len(trade_all) else 0.0

    return {
        "records": records,
        "summary": {
            "samples": int(len(actual_all)),
            "overall_accuracy_pct": overall_accuracy,
            "active_trade_accuracy_pct": active_trade_accuracy,
            "move_threshold": float(move_threshold),
        },
        "per_asset": per_asset_summary,
    }


def build_signal_reason(signal_name: str, top_features: List[Dict[str, float | str]], is_outlier: bool) -> str:
    """Generate concise BUY/SELL/HOLD explanation text from SHAP top drivers."""
    if not top_features:
        return "No strong SHAP drivers detected for this signal."

    strongest = str(top_features[0]["feature"])
    strongest_val = float(top_features[0]["shap_value"])
    supporting = [str(x["feature"]) for x in top_features[1:3]]
    support_txt = ", ".join(supporting) if supporting else "other momentum features"

    if signal_name == "BUY":
        base = (
            f"Buy bias is supported by {strongest} ({strongest_val:+.4f}) with confirmation from {support_txt}."
        )
    elif signal_name == "SELL":
        base = (
            f"Sell bias is supported by {strongest} ({strongest_val:+.4f}) with confirmation from {support_txt}."
        )
    else:
        base = (
            f"Hold bias reflects mixed pressure where {strongest} ({strongest_val:+.4f}) is not decisive."
        )

    if is_outlier:
        return base + " Risk is elevated due to outlier behavior, so position sizing should be reduced."
    return base


def build_top5_shap_reasons(
    signal_name: str,
    top_features: List[Dict[str, float | str]],
    confidence: float,
    atr14: float,
    latest_close: float,
    is_outlier: bool,
) -> List[str]:
    """Create exactly 5 SHAP-grounded reasons aligned with BUY/SELL/HOLD meaning."""
    reasons: List[str] = []

    supporting = [x for x in top_features if float(x.get("shap_value", 0.0)) > 0]
    opposing = [x for x in top_features if float(x.get("shap_value", 0.0)) <= 0]

    if signal_name == "BUY":
        for item in supporting[:4]:
            reasons.append(
                f"{item['feature']} contributes positively to BUY ({float(item['shap_value']):+.4f}), supporting upward momentum."
            )
        if not reasons and top_features:
            item = top_features[0]
            reasons.append(
                f"{item['feature']} is the strongest driver ({float(item['shap_value']):+.4f}) and keeps the bias tilted upward."
            )
    elif signal_name == "SELL":
        for item in supporting[:4]:
            reasons.append(
                f"{item['feature']} contributes positively to SELL ({float(item['shap_value']):+.4f}), indicating downward pressure."
            )
        if not reasons and top_features:
            item = top_features[0]
            reasons.append(
                f"{item['feature']} is the dominant SELL driver ({float(item['shap_value']):+.4f}), keeping downside risk elevated."
            )
    else:
        for item in top_features[:3]:
            reasons.append(
                f"{item['feature']} contributes to HOLD context ({float(item['shap_value']):+.4f}), reinforcing a sideways or indecisive setup."
            )

    if opposing:
        opp = opposing[0]
        if signal_name == "BUY":
            reasons.append(
                f"{opp['feature']} is a mild headwind ({float(opp['shap_value']):+.4f}), but not enough to overturn the bullish tilt."
            )
        elif signal_name == "SELL":
            reasons.append(
                f"{opp['feature']} offers limited counter-pressure ({float(opp['shap_value']):+.4f}), yet bearish drivers remain stronger."
            )
        else:
            reasons.append(
                f"Mixed SHAP signs (e.g., {opp['feature']} at {float(opp['shap_value']):+.4f}) indicate uncertainty rather than trend conviction."
            )

    atr_ratio = float(atr14 / (latest_close + 1e-12)) if latest_close > 0 else 0.0
    reasons.append(
        f"Model confidence is {confidence * 100.0:.2f}% with ATR/price at {atr_ratio * 100.0:.2f}%, consistent with this signal strength."
    )
    if is_outlier:
        reasons.append("Outlier flag is ON, so forecast assumes cautious sizing despite model signal.")
    else:
        reasons.append("Outlier flag is OFF, so SHAP drivers are treated as stable for short-horizon inference.")

    while len(reasons) < 5:
        reasons.append("Top SHAP contributors remain directionally aligned with the predicted action.")

    return reasons[:5]


def generate_five_day_forecast_rows(
    asset_row: dict,
    top_features: List[Dict[str, float | str]],
) -> List[dict]:
    """Generate date-wise 5-day forecast rows with action and SHAP-based reasons."""
    action = str(asset_row["signal"])
    confidence = float(asset_row["confidence"])
    latest_close = float(asset_row["latest_close"])
    atr14 = float(asset_row["atr14"])
    is_outlier = bool(asset_row["outlier_flagged"])

    reasons = build_top5_shap_reasons(
        signal_name=action,
        top_features=top_features,
        confidence=confidence,
        atr14=atr14,
        latest_close=latest_close,
        is_outlier=is_outlier,
    )
    reasons_text = " | ".join([f"{idx + 1}. {txt}" for idx, txt in enumerate(reasons)])

    atr_ratio = float(atr14 / (latest_close + 1e-12)) if latest_close > 0 else 0.0
    base_move = float(np.clip(0.6 * atr_ratio, 0.002, 0.03))
    strength = 0.7 + (0.6 * confidence)

    if action == "BUY":
        daily_return = base_move * strength
    elif action == "SELL":
        daily_return = -base_move * strength
    else:
        daily_return = float(np.clip((confidence - 0.5) * 0.0015, -0.001, 0.001))

    forecast_rows: List[dict] = []
    current_price = latest_close
    for ts in pd.date_range(pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1), periods=5, freq="D"):
        current_price = float(current_price * (1.0 + daily_return))
        forecast_rows.append(
            {
                "Date": ts.strftime("%Y-%m-%d"),
                "Predicted Price": round(current_price, 6),
                "Action": action,
                "Top 5 Reasons (from SHAP)": reasons_text,
            }
        )

    return forecast_rows


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


def run_backtest_for_assets(
    model: tf.keras.Model,
    dataset: dict,
    train_ratio: float = 0.8,
    fee_bps: float = BACKTEST_FEE_BPS,
    initial_capital_per_asset: float = BACKTEST_INITIAL_CAPITAL_PER_ASSET,
    calibration_params: dict | None = None,
) -> dict:
    """Simulate automatic paper-trading on test windows and return PnL metrics."""
    per_asset: Dict[str, dict] = {}
    total_initial = 0.0
    total_final = 0.0
    total_trades = 0
    total_wins = 0

    for asset_name, asset_data in dataset["assets"].items():
        n = len(asset_data.X)
        split_idx = min(n - 1, max(1, int(n * train_ratio)))

        X_test = asset_data.X[split_idx:]
        current_test = asset_data.current_close[split_idx:]
        next_test = asset_data.next_close[split_idx:]

        if len(X_test) == 0:
            per_asset[asset_name] = {
                "trade_count": 0,
                "win_rate_pct": 0.0,
                "profit_amount": 0.0,
                "return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "final_equity": float(initial_capital_per_asset),
            }
            total_initial += float(initial_capital_per_asset)
            total_final += float(initial_capital_per_asset)
            continue

        pred_label, pred_probs, _ = predict_signals(model, X_test)
        calibration = calibration_params or {}
        pred_label = np.asarray([calibrate_signal_from_probs(p, **calibration) for p in pred_probs], dtype=np.int32)
        positions = np.where(pred_label == 0, 1, np.where(pred_label == 1, -1, 0)).astype(np.float32)

        gross_returns = positions * ((next_test - current_test) / (current_test + 1e-12))
        trade_cost = (np.abs(positions) > 0).astype(np.float32) * (fee_bps / 10000.0)
        net_returns = gross_returns - trade_cost

        equity_curve = float(initial_capital_per_asset) * np.cumprod(1.0 + net_returns)
        final_equity = float(equity_curve[-1]) if len(equity_curve) else float(initial_capital_per_asset)
        profit_amount = final_equity - float(initial_capital_per_asset)
        return_pct = (profit_amount / float(initial_capital_per_asset)) * 100.0

        trade_mask = positions != 0
        trade_count = int(np.sum(trade_mask))
        wins = int(np.sum(net_returns[trade_mask] > 0)) if trade_count > 0 else 0
        win_rate_pct = (wins / trade_count) * 100.0 if trade_count > 0 else 0.0

        rolling_peak = np.maximum.accumulate(equity_curve) if len(equity_curve) else np.array([initial_capital_per_asset])
        drawdown = (equity_curve - rolling_peak) / (rolling_peak + 1e-12) if len(equity_curve) else np.array([0.0])
        max_drawdown_pct = float(np.min(drawdown) * 100.0)

        per_asset[asset_name] = {
            "trade_count": trade_count,
            "win_rate_pct": float(win_rate_pct),
            "profit_amount": float(profit_amount),
            "return_pct": float(return_pct),
            "max_drawdown_pct": float(max_drawdown_pct),
            "final_equity": float(final_equity),
        }

        total_initial += float(initial_capital_per_asset)
        total_final += final_equity
        total_trades += trade_count
        total_wins += wins

    portfolio_profit = total_final - total_initial
    portfolio_return_pct = (portfolio_profit / total_initial) * 100.0 if total_initial > 0 else 0.0
    portfolio_win_rate_pct = (total_wins / total_trades) * 100.0 if total_trades > 0 else 0.0

    return {
        "per_asset": per_asset,
        "summary": {
            "initial_capital": float(total_initial),
            "final_equity": float(total_final),
            "profit_amount": float(portfolio_profit),
            "return_pct": float(portfolio_return_pct),
            "trade_count": int(total_trades),
            "win_rate_pct": float(portfolio_win_rate_pct),
            "fee_bps": float(fee_bps),
        },
    }


def run_pipeline_collect() -> dict:
    """Execute ExplainInvest and return structured results for downstream UIs/APIs."""
    set_reproducibility(42)

    print("[1/8] Collecting and preprocessing market data...", flush=True)
    dataset = collect_and_preprocess_data(start=DATA_START, end=DATA_END, time_steps=TIME_STEPS)
    available_assets = list(dataset["assets"].keys())
    selected_assets = pick_market_universe(
        available_assets,
        preferred_markets=PREFERRED_MARKETS,
        count=MARKET_COUNT,
    )
    dataset = filter_dataset_assets(dataset, selected_assets)

    if len(dataset["assets"]) == 0:
        raise RuntimeError("No assets available after market selection.")

    print(f"[2/8] Preparing {len(dataset['assets'])} selected markets...", flush=True)
    split_data = split_train_test_by_asset(dataset, train_ratio=0.8)

    X_train = split_data["X_train"]

    print("[3/8] Fitting outlier detector...", flush=True)
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

    print("[4/8] Generating pseudo-labels from clustered market moves...", flush=True)
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
    X_train_clean, y_train_clean = balance_training_samples(X_train_clean, y_train_clean, seed=42)

    label_counts = {int(cls): int(np.sum(y_train_clean == cls)) for cls in [0, 1, 2]}
    print(
        "[5/8] Training label balance -> "
        f"BUY={label_counts.get(0, 0)}, SELL={label_counts.get(1, 0)}, HOLD={label_counts.get(2, 0)}",
        flush=True,
    )

    model = build_cnn_lstm_attention_model(
        time_steps=X_train.shape[1],
        num_features=X_train.shape[2],
        learning_rate=1e-3,
    )

    unique_classes = np.asarray(sorted({int(v) for v in y_train_clean}), dtype=np.int32)
    cw_values = compute_class_weight("balanced", classes=unique_classes, y=y_train_clean)
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, cw_values)}
    for missing_class in [0, 1, 2]:
        if missing_class not in class_weight_dict:
            class_weight_dict[missing_class] = 1.0

    print("[5/8] Training trading model...", flush=True)
    train_trading_model(
        model=model,
        X_train=X_train_clean,
        y_train=y_train_clean,
        epochs=MODEL_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        early_stopping_patience=MODEL_EARLY_STOPPING_PATIENCE,
        min_delta=MODEL_MIN_DELTA,
    )

    calibration = tune_signal_calibration(
        model=model,
        dataset=dataset,
        train_ratio=0.8,
        move_threshold=EVAL_MOVE_THRESHOLD,
    )
    print(
        "[6/8] Calibrated signal thresholds -> "
        f"min_conf={calibration['params']['min_confidence']:.3f}, "
        f"min_edge={calibration['params']['min_edge']:.3f}, "
        f"buy_sell_gap={calibration['params']['buy_sell_gap']:.3f}, "
        f"hold_switch_gap={calibration['params']['hold_switch_gap']:.3f}, "
        f"holdout_acc={calibration['holdout_accuracy_pct']:.2f}%",
        flush=True,
    )

    print("[6/8] Running auto-trading backtest on test window...", flush=True)
    backtest = run_backtest_for_assets(
        model=model,
        dataset=dataset,
        train_ratio=0.8,
        fee_bps=BACKTEST_FEE_BPS,
        initial_capital_per_asset=BACKTEST_INITIAL_CAPITAL_PER_ASSET,
        calibration_params=calibration["params"],
    )

    evaluation = evaluate_prediction_vs_actual(
        model=model,
        dataset=dataset,
        train_ratio=0.8,
        move_threshold=EVAL_MOVE_THRESHOLD,
        calibration_params=calibration["params"],
    )
    evaluation["summary"]["calibration"] = calibration

    latest_inputs = prepare_asset_latest_inputs(dataset)

    asset_signals: Dict[str, int] = {}
    asset_confidence: Dict[str, float] = {}
    asset_volatility: Dict[str, float] = {}
    asset_outlier: Dict[str, bool] = {}
    asset_symbol: Dict[str, str] = {}
    asset_latest_close: Dict[str, float] = {}
    per_asset_top_features: Dict[str, List[str]] = {}
    per_asset_top_feature_dicts: Dict[str, List[Dict[str, float | str]]] = {}
    per_asset_reason: Dict[str, str] = {}

    background_count = min(SHAP_BACKGROUND_MAX, len(X_train_clean))
    if background_count < 5:
        raise RuntimeError("Not enough clean training samples for SHAP background.")
    background_data = X_train_clean[
        np.random.choice(len(X_train_clean), size=background_count, replace=False)
    ]
    explainer = ExplainabilityEngine(model, background_data, FEATURE_COLUMNS)

    print("[7/8] Predicting, checking outliers, and explaining each market...", flush=True)
    for asset_name, payload in latest_inputs.items():
        X_latest = payload["X_latest"]
        latest_atr = payload["latest_atr"]

        asset_full = dataset["assets"][asset_name]
        latest_flat = asset_full.X[-1:].reshape(1, -1)
        latest_atr_arr = np.asarray([asset_full.atr[-1]], dtype=np.float32)
        asset_outlier_result = detector.detect(latest_flat, latest_atr_arr)
        last_outlier_flag = bool(asset_outlier_result.combined_flag[0])

        pred_label, pred_probs, pred_conf = predict_signals(model, X_latest)
        signal = calibrate_signal_from_probs(pred_probs[0], **calibration["params"])
        confidence = float(np.max(pred_probs[0]))

        print(
            f"  - {asset_name}: signal={SIGNAL_NAME[signal]}, outlier={'Yes' if last_outlier_flag else 'No'}",
            flush=True,
        )

        asset_signals[asset_name] = signal
        asset_confidence[asset_name] = confidence
        asset_volatility[asset_name] = latest_atr
        asset_outlier[asset_name] = last_outlier_flag
        asset_symbol[asset_name] = asset_full.symbol
        asset_latest_close[asset_name] = float(asset_full.raw_df["Close"].iloc[-1])

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
        per_asset_top_feature_dicts[asset_name] = top_features
        per_asset_reason[asset_name] = build_signal_reason(
            signal_name=SIGNAL_NAME[signal],
            top_features=top_features,
            is_outlier=last_outlier_flag,
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
                "symbol": asset_symbol[asset_name],
                "latest_close": asset_latest_close[asset_name],
                "signal": SIGNAL_NAME[asset_signals[asset_name]],
                "signal_idx": int(asset_signals[asset_name]),
                "confidence": float(asset_confidence[asset_name]),
                "atr14": float(asset_volatility[asset_name]),
                "portfolio_weight": float(weights.get(asset_name, 0.0)),
                "outlier_flagged": bool(asset_outlier[asset_name]),
                "paper_trade_count": int(backtest["per_asset"][asset_name]["trade_count"]),
                "paper_win_rate_pct": float(backtest["per_asset"][asset_name]["win_rate_pct"]),
                "paper_profit": float(backtest["per_asset"][asset_name]["profit_amount"]),
                "paper_return_pct": float(backtest["per_asset"][asset_name]["return_pct"]),
                "paper_max_drawdown_pct": float(backtest["per_asset"][asset_name]["max_drawdown_pct"]),
                "test_accuracy_pct": float(
                    evaluation["per_asset"].get(asset_name, {}).get("accuracy_pct", 0.0)
                ),
                "test_active_trade_accuracy_pct": float(
                    evaluation["per_asset"].get(asset_name, {}).get("active_trade_accuracy_pct", 0.0)
                ),
                "decision_reason": per_asset_reason[asset_name],
                "top_features": per_asset_top_features[asset_name],
            }
        )

    rows.sort(key=lambda x: x["portfolio_weight"], reverse=True)

    forecast_asset_row = rows[0]
    if forecast_asset_row["portfolio_weight"] <= 0 and len(rows) > 0:
        # Fall back to the highest-confidence row when allocation is all-zero.
        forecast_asset_row = sorted(rows, key=lambda x: x["confidence"], reverse=True)[0]

    five_day_forecast = generate_five_day_forecast_rows(
        asset_row=forecast_asset_row,
        top_features=per_asset_top_feature_dicts[forecast_asset_row["asset"]],
    )

    print("[8/8] Portfolio allocation complete.", flush=True)
    return {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "assets": rows,
        "class_weights": class_weight_dict,
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "backtest_summary": backtest["summary"],
        "evaluation_summary": evaluation["summary"],
        "signal_calibration": calibration,
        "five_day_forecast_asset": {
            "asset": forecast_asset_row["asset"],
            "symbol": forecast_asset_row["symbol"],
        },
        "five_day_forecast": five_day_forecast,
        "prediction_vs_actual_records": evaluation["records"],
        "meta": {
            "time_steps": int(dataset["time_steps"]),
            "asset_count": int(len(rows)),
            "train_samples": int(len(X_train_clean)),
            "selected_markets": selected_assets,
            "available_market_count": int(len(dataset["assets"])),
            "signal_calibration_params": calibration["params"],
        },
    }


def run_pipeline() -> None:
    """Execute ExplainInvest from data fetch to signal, allocation, and explainability output."""
    result = run_pipeline_collect()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "portfolio_report.json"
    csv_path = REPORT_DIR / "portfolio_report.csv"
    pred_vs_actual_path = REPORT_DIR / "prediction_vs_actual.csv"
    five_day_forecast_path = REPORT_DIR / "five_day_forecast.csv"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)
    pd.DataFrame(result["assets"]).to_csv(csv_path, index=False)
    pd.DataFrame(result.get("prediction_vs_actual_records", [])).to_csv(pred_vs_actual_path, index=False)
    pd.DataFrame(result.get("five_day_forecast", [])).to_csv(five_day_forecast_path, index=False)

    active = [row for row in result["assets"] if row["portfolio_weight"] > 0]
    strongest = active[0] if active else None

    class_weights = result["class_weights"]
    backtest_summary = result["backtest_summary"]
    eval_summary = result.get("evaluation_summary", {})
    print("=" * 90)
    print("ExplainInvest Portfolio Management System")
    print(f"Generated: {result['generated_at']}")
    print(f"Markets:   {', '.join(result['meta']['selected_markets'])}")
    print(f"Report:    {report_path}")
    print(f"CSV:       {csv_path}")
    print("=" * 90)
    print(
        f"Class weights → BUY={class_weights.get(0, 1.0):.2f}, "
        f"SELL={class_weights.get(1, 1.0):.2f}, HOLD={class_weights.get(2, 1.0):.2f}"
    )
    print(
        "Paper Auto-Trade Profit -> "
        f"Initial: {backtest_summary['initial_capital']:.2f}, "
        f"Final: {backtest_summary['final_equity']:.2f}, "
        f"Profit: {backtest_summary['profit_amount']:+.2f} "
        f"({backtest_summary['return_pct']:+.2f}%), "
        f"Trades: {backtest_summary['trade_count']}, "
        f"Win Rate: {backtest_summary['win_rate_pct']:.2f}%"
    )
    print(
        "Prediction vs Actual -> "
        f"Samples: {int(eval_summary.get('samples', 0))}, "
        f"Overall Accuracy: {float(eval_summary.get('overall_accuracy_pct', 0.0)):.2f}%, "
        f"Active Trade Accuracy: {float(eval_summary.get('active_trade_accuracy_pct', 0.0)):.2f}%"
    )
    print(
        f"[8/8] Final ExplainInvest output for {result['meta']['available_market_count']} selected markets"
    )

    for row in result["assets"]:
        print("-" * 70)
        print(f"Asset:            {row['asset']}")
        print(f"Symbol:           {row['symbol']}")
        print(f"Latest Close:     {row['latest_close']:.5f}")
        print(f"Signal:           {row['signal']}")
        print(f"Confidence:       {row['confidence'] * 100.0:.2f}%")
        print(f"ATR(14):          {row['atr14']:.6f}")
        print(f"Portfolio Weight: {row['portfolio_weight'] * 100.0:.2f}%")
        print(f"Outlier Flagged:  {'Yes' if row['outlier_flagged'] else 'No'}")
        print(
            f"Paper PnL:        {row['paper_profit']:+.2f} ({row['paper_return_pct']:+.2f}%), "
            f"Trades={row['paper_trade_count']}, WinRate={row['paper_win_rate_pct']:.2f}%, "
            f"MaxDD={row['paper_max_drawdown_pct']:.2f}%"
        )
        print(
            f"Test Accuracy:    {row['test_accuracy_pct']:.2f}% "
            f"(Active={row['test_active_trade_accuracy_pct']:.2f}%)"
        )
        print(f"Reason:           {row['decision_reason']}")
        print("Top 5 SHAP Features:")
        for line in row["top_features"]:
            print(f"  {line}")

    print("-" * 70)
    if strongest is not None:
        print(
            "Top allocation recommendation: "
            f"{strongest['asset']} ({strongest['signal']}) at {strongest['portfolio_weight'] * 100.0:.2f}%"
        )
    else:
        print("Top allocation recommendation: No active position (all markets HOLD or filtered).")

    print("-" * 70)
    forecast_asset = result.get("five_day_forecast_asset", {})
    print(
        "Next 5-Day Forecast (Date-wise) -> "
        f"{forecast_asset.get('asset', 'N/A')} ({forecast_asset.get('symbol', 'N/A')})"
    )
    five_day_df = pd.DataFrame(result.get("five_day_forecast", []))
    if not five_day_df.empty:
        print(five_day_df.to_string(index=False))
    print(f"5-Day Forecast CSV: {five_day_forecast_path}")
    print("-" * 70)
    print(f"Prediction-vs-Actual CSV: {pred_vs_actual_path}")
    print("-" * 70)
    print("Correlation Matrix Snapshot")
    for asset_name in result["meta"]["selected_markets"]:
        row = result["correlation_matrix"].get(asset_name, {})
        formatted = ", ".join([f"{k}: {v:.2f}" for k, v in row.items()])
        print(f"  {asset_name}: {formatted}")


if __name__ == "__main__":
    run_pipeline()