"""Portfolio allocation logic for ExplainInvest."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_asset_correlation(
    asset_to_close: Dict[str, pd.Series],
    lookback: int = 500,
) -> pd.DataFrame:
    """Compute return-based correlation matrix across assets using recent history."""
    returns = {}
    for asset, close_series in asset_to_close.items():
        series = close_series.tail(lookback).pct_change().dropna()
        returns[asset] = series

    returns_df = pd.DataFrame(returns).dropna(how="all")
    if returns_df.empty:
        return pd.DataFrame(index=list(asset_to_close.keys()), columns=list(asset_to_close.keys())).fillna(0.0)

    corr = returns_df.corr().fillna(0.0)
    return corr


def allocate_portfolio(
    asset_signals: Dict[str, int],
    asset_confidence: Dict[str, float],
    asset_volatility: Dict[str, float],
    asset_outlier_flag: Dict[str, bool],
    correlation_matrix: pd.DataFrame,
    corr_threshold: float = 0.8,
) -> Dict[str, float]:
    """Allocate capital by confidence/volatility while enforcing hold/outlier/correlation rules."""
    raw_scores: Dict[str, float] = {}

    for asset, signal in asset_signals.items():
        if signal == 2 or asset_outlier_flag.get(asset, False):
            raw_scores[asset] = 0.0
            continue

        conf = float(asset_confidence.get(asset, 0.0))
        vol = float(asset_volatility.get(asset, 0.0))
        raw_scores[asset] = conf / (vol + 1e-6)

    total_score = sum(raw_scores.values())
    if total_score <= 0.0:
        return {asset: 0.0 for asset in asset_signals}

    weights = {asset: score / total_score for asset, score in raw_scores.items()}

    # Reduce both assets if they are too correlated and both currently carry risk.
    assets = list(weights.keys())
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a_i = assets[i]
            a_j = assets[j]
            if a_i not in correlation_matrix.index or a_j not in correlation_matrix.columns:
                continue

            corr_val = pd.to_numeric(correlation_matrix.loc[a_i, a_j], errors="coerce")
            if pd.isna(corr_val):
                continue

            corr_float = float(corr_val)
            if corr_float > corr_threshold and weights[a_i] > 0 and weights[a_j] > 0:
                weights[a_i] *= 0.5
                weights[a_j] *= 0.5

    return weights
