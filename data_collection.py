"""Data collection and preprocessing utilities for ExplainInvest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


ASSET_SYMBOLS: Dict[str, List[str]] = {
    # Forex
    "EUR/USD": ["EURUSD=X", "EURUSD%3DX"],
    "GBP/USD": ["GBPUSD=X", "GBPUSD%3DX"],
    "USD/JPY": ["USDJPY=X", "JPY=X"],
    "AUD/USD": ["AUDUSD=X"],
    "USD/CAD": ["USDCAD=X"],
    "USD/CHF": ["USDCHF=X"],
    "NZD/USD": ["NZDUSD=X"],
    # Stocks
    "S&P500": ["^GSPC"],
    "NASDAQ": ["^IXIC"],
    "Dow Jones": ["^DJI"],
    "Apple": ["AAPL"],
    "Microsoft": ["MSFT"],
    "Tesla": ["TSLA"],
    "NVIDIA": ["NVDA"],
    # Commodities
    "Gold": ["GC=F"],
    "Silver": ["SI=F"],
    "Crude Oil": ["CL=F"],
    "Copper": ["HG=F"],
    "Natural Gas": ["NG=F"],
    # Crypto
    "BTC-USD": ["BTC-USD"],
    "ETH-USD": ["ETH-USD"],
    "ADA-USD": ["ADA-USD"],
    "SOL-USD": ["SOL-USD"],
    "XRP-USD": ["XRP-USD"],
}


FEATURE_COLUMNS: List[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Avg_Price",
    "Volume",
    "Volume_MA20",
    "RSI14",
    "MACD",
    "MACD_Signal",
    "MA20",
    "MA50",
    "MA100",
    "MA200",
    "ATR14",
    "Bollinger_Upper",
    "Bollinger_Lower",
    "Sentiment",
]


@dataclass
class AssetDataset:
    """Container for per-asset preprocessed data and sequence metadata."""

    name: str
    symbol: str
    raw_df: pd.DataFrame
    scaled_df: pd.DataFrame
    X: np.ndarray
    current_close: np.ndarray
    next_close: np.ndarray
    atr: np.ndarray
    timestamps: np.ndarray


def _download_interval_chunks(
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    interval: str,
    chunk_days: int,
) -> pd.DataFrame:
    """Download one interval in chunks and return a deduplicated time series frame."""
    frames: List[pd.DataFrame] = []
    cursor = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        try:
            df_chunk = yf.download(
                symbol,
                start=cursor.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df_chunk is not None and not df_chunk.empty:
                # Normalize all timestamps to tz-naive UTC so concatenation and sorting
                # never mixes tz-aware and tz-naive indices.
                df_chunk.index = pd.to_datetime(df_chunk.index, utc=True).tz_convert(None)
                frames.append(df_chunk)
        except Exception:
            # Silently skip chunks that fail due to yfinance issues; continue to next chunk.
            pass
        cursor = chunk_end

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _download_symbol_with_fallbacks(symbol_candidates: List[str], start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Try multiple ticker symbols and return the first non-empty dataset."""
    for symbol in symbol_candidates:
        df = _download_market_data(symbol=symbol, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
        if not df.empty:
            return df, symbol
    return pd.DataFrame(), symbol_candidates[0] if symbol_candidates else ""


def _download_market_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download market data with retry and safe fallback for Yahoo intraday history limits.

    Yahoo Finance limits 1-hour data to roughly the last 730 days. This function
    fetches older history with daily bars and the recent window with hourly bars.
    Includes retry logic to handle transient API failures.
    """
    import time
    
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    now_dt = pd.Timestamp.utcnow().tz_localize(None)
    hourly_cutoff = now_dt - timedelta(days=729)

    for attempt in range(3):
        try:
            frames: List[pd.DataFrame] = []

            # Older segment: use daily candles where hourly is unavailable.
            if start_dt < min(end_dt, hourly_cutoff):
                old_end = min(end_dt, hourly_cutoff)
                old_df = _download_interval_chunks(
                    symbol=symbol,
                    start_dt=start_dt,
                    end_dt=old_end,
                    interval="1d",
                    chunk_days=365 * 4,
                )
                if not old_df.empty:
                    frames.append(old_df)

            # Recent segment: keep hourly candles where Yahoo supports it.
            hourly_start = max(start_dt, hourly_cutoff)
            if hourly_start < end_dt:
                recent_df = _download_interval_chunks(
                    symbol=symbol,
                    start_dt=hourly_start,
                    end_dt=end_dt,
                    interval="1h",
                    chunk_days=60,
                )
                if not recent_df.empty:
                    frames.append(recent_df)

            if not frames:
                return pd.DataFrame()

            merged = pd.concat(frames)
            merged.index = pd.to_datetime(merged.index, utc=True).tz_convert(None)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            return merged
        
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return pd.DataFrame()


def _download_market_data_for_candidates(symbol_candidates: List[str], start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Download market data by probing multiple ticker candidates."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    for symbol in symbol_candidates:
        df = _download_market_data(symbol=symbol, start=start, end=end)
        if not df.empty:
            return df, symbol

    return pd.DataFrame(), symbol_candidates[0] if symbol_candidates else ""


def _prepare_raw_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the downloaded OHLCV frame and enforce required numeric columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    expected = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    out = df[expected].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out["Volume"] = out["Volume"].fillna(0.0)
    return out


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create required technical indicators and placeholders for model features."""
    feat = df.copy()

    # Call ta.add_all_ta_features as requested, then keep a curated feature set.
    feat = ta.add_all_ta_features(
        feat,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )

    feat["Avg_Price"] = (feat["Open"] + feat["High"] + feat["Low"] + feat["Close"]) / 4.0
    feat["Volume_MA20"] = feat["Volume"].rolling(20, min_periods=1).mean()
    feat["RSI14"] = ta.momentum.rsi(feat["Close"], window=14, fillna=True)
    feat["MACD"] = ta.trend.macd(feat["Close"], fillna=True)
    feat["MACD_Signal"] = ta.trend.macd_signal(feat["Close"], fillna=True)
    feat["MA20"] = feat["Close"].rolling(20, min_periods=1).mean()
    feat["MA50"] = feat["Close"].rolling(50, min_periods=1).mean()
    feat["MA100"] = feat["Close"].rolling(100, min_periods=1).mean()
    feat["MA200"] = feat["Close"].rolling(200, min_periods=1).mean()
    feat["ATR14"] = ta.volatility.average_true_range(
        feat["High"], feat["Low"], feat["Close"], window=14, fillna=True
    )

    bb = ta.volatility.BollingerBands(close=feat["Close"], window=20, window_dev=2, fillna=True)
    feat["Bollinger_Upper"] = bb.bollinger_hband()
    feat["Bollinger_Lower"] = bb.bollinger_lband()

    # Placeholder sentiment until external sentiment model is integrated.
    feat["Sentiment"] = 0.0

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat


def _build_sequences(
    scaled_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert feature matrix into supervised sliding windows with aligned metadata."""
    X_list: List[np.ndarray] = []
    current_close_list: List[float] = []
    next_close_list: List[float] = []
    atr_list: List[float] = []
    timestamp_list: List[datetime] = []

    features_np = scaled_df[FEATURE_COLUMNS].values

    for i in range(time_steps, len(scaled_df) - 1):
        X_list.append(features_np[i - time_steps : i])
        current_close_list.append(float(raw_df["Close"].iloc[i]))
        next_close_list.append(float(raw_df["Close"].iloc[i + 1]))
        atr_list.append(float(raw_df["ATR14"].iloc[i]))
        timestamp_list.append(raw_df.index[i])

    if not X_list:
        empty = np.empty((0, time_steps, len(FEATURE_COLUMNS)), dtype=np.float32)
        return empty, np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(current_close_list, dtype=np.float32),
        np.asarray(next_close_list, dtype=np.float32),
        np.asarray(atr_list, dtype=np.float32),
        np.asarray(timestamp_list),
    )


def collect_and_preprocess_data(
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    time_steps: int = 48,
) -> dict:
    """Download all assets, engineer features, normalize, and create model sequences."""
    per_asset_feature_frames: Dict[str, pd.DataFrame] = {}
    per_asset_raw_frames: Dict[str, pd.DataFrame] = {}
    per_asset_symbol_used: Dict[str, str] = {}

    for asset_name, symbol_candidates in ASSET_SYMBOLS.items():
        try:
            raw_download, used_symbol = _download_market_data_for_candidates(symbol_candidates, start=start, end=end)
            if raw_download.empty:
                continue

            raw_ohlcv = _prepare_raw_ohlcv(raw_download)
            feat_df = _engineer_features(raw_ohlcv)

            if len(feat_df) > time_steps + 1:
                per_asset_raw_frames[asset_name] = feat_df.copy()
                per_asset_feature_frames[asset_name] = feat_df[FEATURE_COLUMNS].copy()
                per_asset_symbol_used[asset_name] = used_symbol
        except Exception as e:
            print(f"  WARNING: Failed to process {asset_name}: {e}")
            continue

    if not per_asset_feature_frames:
        raise RuntimeError("No data downloaded. Verify internet access and ticker symbols.")

    # Fit a global scaler across all assets to keep a shared 0-1 feature range.
    stacked = pd.concat(per_asset_feature_frames.values(), axis=0)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(stacked.values)

    asset_datasets: Dict[str, AssetDataset] = {}
    X_all: List[np.ndarray] = []
    close_all: List[np.ndarray] = []
    next_close_all: List[np.ndarray] = []
    atr_all: List[np.ndarray] = []
    asset_index: List[str] = []
    timestamp_all: List[np.ndarray] = []

    for asset_name, feat_df in per_asset_raw_frames.items():
        scaled_values = scaler.transform(feat_df[FEATURE_COLUMNS].values)
        scaled_df = feat_df.copy()
        scaled_df[FEATURE_COLUMNS] = scaled_values

        X, current_close, next_close, atr_values, timestamps = _build_sequences(
            scaled_df=scaled_df,
            raw_df=feat_df,
            time_steps=time_steps,
        )

        if len(X) == 0:
            continue

        dataset = AssetDataset(
            name=asset_name,
            symbol=per_asset_symbol_used.get(asset_name, ""),
            raw_df=feat_df,
            scaled_df=scaled_df,
            X=X,
            current_close=current_close,
            next_close=next_close,
            atr=atr_values,
            timestamps=timestamps,
        )
        asset_datasets[asset_name] = dataset

        X_all.append(X)
        close_all.append(current_close)
        next_close_all.append(next_close)
        atr_all.append(atr_values)
        timestamp_all.append(timestamps)
        asset_index.extend([asset_name] * len(X))

    return {
        "assets": asset_datasets,
        "feature_columns": FEATURE_COLUMNS,
        "scaler": scaler,
        "time_steps": time_steps,
        "X_all": np.concatenate(X_all, axis=0),
        "current_close_all": np.concatenate(close_all, axis=0),
        "next_close_all": np.concatenate(next_close_all, axis=0),
        "atr_all": np.concatenate(atr_all, axis=0),
        "timestamps_all": np.concatenate(timestamp_all, axis=0),
        "asset_index": np.asarray(asset_index),
    }
