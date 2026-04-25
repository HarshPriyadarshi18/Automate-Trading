"""Streamlit dashboard for ExplainInvest portfolio management."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any
from pathlib import Path
import uuid

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import yfinance as yf

from main import run_pipeline_collect


REPORT_PATH = Path("reports/portfolio_report.json")
TRADE_LOG_PATH = Path("reports/live_trade_log.csv")

TRADE_LOG_COLUMNS = [
    "trade_id",
    "asset",
    "symbol",
    "side",
    "entry_ts_utc",
    "entry_price",
    "quantity",
    "entry_confidence_pct",
    "entry_atr",
    "status",
    "exit_ts_utc",
    "exit_price",
    "pnl",
    "pnl_pct",
    "outcome",
    "close_reason",
]

SIGNAL_STYLE = {
    "BUY": {"bg": "#0f5132", "accent": "#20c997", "label": "BUY"},
    "SELL": {"bg": "#58151c", "accent": "#f06595", "label": "SELL"},
    "HOLD": {"bg": "#5f4b0a", "accent": "#f2c94c", "label": "HOLD"},
}


def load_report(path: Path) -> dict:
    """Load JSON portfolio report from disk."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_report(report: dict) -> None:
    """Persist JSON and CSV report so dashboard and CLI stay aligned."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    pd.DataFrame(report.get("assets", [])).to_csv(REPORT_PATH.parent / "portfolio_report.csv", index=False)


def load_trade_log() -> pd.DataFrame:
    """Load paper auto-trade log from disk, creating it when missing."""
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TRADE_LOG_PATH.exists():
        pd.DataFrame(columns=TRADE_LOG_COLUMNS).to_csv(TRADE_LOG_PATH, index=False)
    df = pd.read_csv(TRADE_LOG_PATH)
    for col in TRADE_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in {"trade_id", "asset", "symbol", "side", "entry_ts_utc", "status", "exit_ts_utc", "outcome", "close_reason"} else 0.0
    return df[TRADE_LOG_COLUMNS].copy()


def save_trade_log(df: pd.DataFrame) -> None:
    """Persist paper auto-trade log to disk."""
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRADE_LOG_PATH, index=False)


def _safe_ts_utc(value: Any) -> pd.Timestamp:
    """Normalize any timestamp-like value into tz-aware UTC timestamp."""
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
        return ts
    return pd.Timestamp.now(tz="UTC")


def _to_float(value: object, default: float = 0.0) -> float:
    """Convert a pandas scalar or object into a plain float safely."""
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def run_auto_trade_cycle(
    snapshot_df: pd.DataFrame,
    live_snapshots: dict,
    hold_minutes: int,
    min_confidence_pct: float,
    capital_per_trade: float,
    close_on_timeout: bool,
    stop_atr_mult: float,
    target_atr_mult: float,
    min_reversal_hold_minutes: int,
) -> pd.DataFrame:
    """Open/close paper trades based on live signal and evaluate decisions with future price."""
    log_df = load_trade_log()

    for _, row in snapshot_df.iterrows():
        asset = str(row["asset"])
        symbol = str(row["symbol"])
        snap = live_snapshots.get(asset, {})
        signal = str(snap.get("auto_signal", "HOLD"))
        confidence_pct = float(snap.get("auto_confidence_pct", 0.0))
        risk = str(row["risk"])
        atr = float(row.get("atr14", 0.0))

        quote = snap.get("quote", {})
        live_price = float(snap.get("live_price", row.get("latest_close", 0.0)))
        now_ts = _safe_ts_utc(quote.get("timestamp") if quote.get("timestamp") is not None else pd.Timestamp.now(tz="UTC"))

        if live_price <= 0.0:
            continue

        open_mask = (log_df["asset"] == asset) & (log_df["status"] == "OPEN")
        has_open = bool(open_mask.any())

        if has_open:
            open_idx = log_df[open_mask].index[0]
            entry_ts = _safe_ts_utc(log_df.loc[open_idx, "entry_ts_utc"])
            age_minutes = (now_ts - entry_ts).total_seconds() / 60.0

            side = str(log_df.loc[open_idx, "side"])
            should_close = False
            close_reason = ""
            entry_atr = _to_float(log_df.loc[open_idx, "entry_atr"], max(atr, 1e-6)) if "entry_atr" in log_df.columns else max(atr, 1e-6)
            if entry_atr <= 0:
                entry_atr = max(atr, 1e-6)

            entry_price = _to_float(log_df.loc[open_idx, "entry_price"])
            qty = _to_float(log_df.loc[open_idx, "quantity"])

            if side == "BUY":
                stop_price = entry_price - (float(stop_atr_mult) * entry_atr)
                target_price = entry_price + (float(target_atr_mult) * entry_atr)
                if live_price <= stop_price:
                    should_close = True
                    close_reason = "STOP_LOSS"
                elif live_price >= target_price:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
            else:
                stop_price = entry_price + (float(stop_atr_mult) * entry_atr)
                target_price = entry_price - (float(target_atr_mult) * entry_atr)
                if live_price >= stop_price:
                    should_close = True
                    close_reason = "STOP_LOSS"
                elif live_price <= target_price:
                    should_close = True
                    close_reason = "TAKE_PROFIT"

            if close_on_timeout and age_minutes >= float(hold_minutes) and not should_close:
                should_close = True
                close_reason = "HOLD_WINDOW_DONE"
            elif (
                (side == "BUY" and signal == "SELL") or (side == "SELL" and signal == "BUY")
            ) and age_minutes >= float(min_reversal_hold_minutes) and not should_close:
                should_close = True
                close_reason = "SIGNAL_REVERSAL"

            if should_close:
                if side == "BUY":
                    pnl = (live_price - entry_price) * qty
                    pnl_pct = ((live_price / (entry_price + 1e-12)) - 1.0) * 100.0
                    outcome = "RIGHT" if pnl > 0 else "WRONG"
                else:
                    pnl = (entry_price - live_price) * qty
                    pnl_pct = ((entry_price / (live_price + 1e-12)) - 1.0) * 100.0
                    outcome = "RIGHT" if pnl > 0 else "WRONG"

                log_df.loc[open_idx, "status"] = "CLOSED"
                log_df.loc[open_idx, "exit_ts_utc"] = now_ts.isoformat()
                log_df.loc[open_idx, "exit_price"] = float(live_price)
                log_df.loc[open_idx, "pnl"] = float(pnl)
                log_df.loc[open_idx, "pnl_pct"] = float(pnl_pct)
                log_df.loc[open_idx, "outcome"] = outcome
                log_df.loc[open_idx, "close_reason"] = close_reason

        open_mask = (log_df["asset"] == asset) & (log_df["status"] == "OPEN")
        has_open = bool(open_mask.any())

        if not has_open and signal in {"BUY", "SELL"} and confidence_pct >= float(min_confidence_pct) and risk != "High":
            qty = float(capital_per_trade) / float(live_price)
            new_row = {
                "trade_id": str(uuid.uuid4()),
                "asset": asset,
                "symbol": symbol,
                "side": signal,
                "entry_ts_utc": now_ts.isoformat(),
                "entry_price": float(live_price),
                "quantity": float(qty),
                "entry_confidence_pct": float(confidence_pct),
                "entry_atr": float(max(atr, 1e-6)),
                "status": "OPEN",
                "exit_ts_utc": "",
                "exit_price": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "outcome": "",
                "close_reason": "",
            }
            log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)

    save_trade_log(log_df)
    return log_df


def rebuild_report_from_model() -> tuple[bool, str]:
    """Run full pipeline and persist a fresh report for dashboard use."""
    try:
        report = run_pipeline_collect()
        save_report(report)
        return True, "Model rerun complete. Report refreshed."
    except Exception as exc:
        return False, f"Model rerun failed: {exc}"


@st.cache_data(ttl=4)
def load_live_price_series(symbol: str, period: str = "3d", interval: str = "5m", nonce: int = 0) -> pd.DataFrame:
    """Fetch short-term movement for dashboard charts with fallback intervals."""
    fetch_plan = [
        (period, interval),
        ("5d", "15m"),
        ("7d", "30m"),
        ("1mo", "1h"),
        ("3mo", "1d"),
    ]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for fetch_period, fetch_interval in fetch_plan:
        try:
            raw = yf.download(
                symbol,
                period=fetch_period,
                interval=fetch_interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            raw = pd.DataFrame()

        if raw is None or raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        for col in required_cols:
            if col not in raw.columns:
                raw[col] = 0.0

        series = raw[required_cols].copy()
        series.index = pd.to_datetime(series.index)
        series = series.dropna(subset=["Close"])
        if not series.empty:
            series.attrs["interval_used"] = fetch_interval
            series.attrs["period_used"] = fetch_period
            return series

    return pd.DataFrame(columns=["Close"])


@st.cache_data(ttl=5)
def load_latest_quote(symbol: str, nonce: int = 0) -> dict:
    """Fetch latest quote with fallback intervals when 1m feed is unavailable."""
    attempts = [("1d", "1m"), ("5d", "5m"), ("1mo", "1h"), ("3mo", "1d")]
    for period, interval in attempts:
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                prepost=True,
            )
            if raw is None or raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if "Close" not in raw.columns:
                continue

            close_series = raw["Close"].dropna()
            if close_series.empty:
                continue

            last_ts = pd.to_datetime(close_series.index[-1])
            last_close = float(close_series.iloc[-1])
            return {
                "price": last_close,
                "timestamp": last_ts,
                "market_state": f"Live/Recent ({interval})",
            }
        except Exception:
            continue

    return {"price": None, "timestamp": None, "market_state": "Unknown"}


def load_latest_quote_uncached(symbol: str) -> dict:
    """Force-refresh quote without cached function, used when lag is too high."""
    attempts = [("1d", "1m"), ("5d", "5m"), ("1mo", "1h")]
    for period, interval in attempts:
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                prepost=True,
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if "Close" not in raw.columns:
                continue
            close_series = raw["Close"].dropna()
            if close_series.empty:
                continue
            return {
                "price": float(close_series.iloc[-1]),
                "timestamp": pd.to_datetime(close_series.index[-1]),
                "market_state": f"Live/Recent ({interval}, forced)",
            }
        except Exception:
            continue
    return {"price": None, "timestamp": None, "market_state": "Unknown"}


def build_live_chart_frame(live_df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Build chart-ready frame with trend and signal levels."""
    chart = live_df[["Close"]].copy()
    chart["EMA20"] = chart["Close"].ewm(span=20, adjust=False).mean()
    chart["EMA50"] = chart["Close"].ewm(span=50, adjust=False).mean()
    chart["Entry"] = float(plan["entry"])
    chart["Stop"] = float(plan["stop"])
    chart["Target"] = float(plan["target"])
    return chart


def build_normalized_movement_frame(live_df: pd.DataFrame) -> pd.DataFrame:
    """Build a normalized percent-change frame so each asset movement is comparable."""
    chart = live_df[["Close"]].copy()
    first_close = float(chart["Close"].iloc[0]) if not chart.empty else 0.0
    if first_close <= 0:
        chart["MovePct"] = 0.0
    else:
        chart["MovePct"] = ((chart["Close"] / first_close) - 1.0) * 100.0
    chart["MovePctEMA"] = chart["MovePct"].ewm(span=10, adjust=False).mean()
    return chart


def inject_theme() -> None:
    """Inject a dark trading-terminal-like theme."""
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 20%, #0c1829 0%, #070f1c 55%, #040913 100%);
            color: #dce6f9;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #081224 0%, #060c18 100%);
            border-right: 1px solid rgba(104, 132, 185, 0.25);
        }
        .panel {
            border: 1px solid rgba(104, 132, 185, 0.25);
            border-radius: 14px;
            padding: 14px 16px;
            background: linear-gradient(160deg, rgba(17,29,50,0.92), rgba(9,17,31,0.92));
        }
        .decision {
            border-radius: 18px;
            padding: 22px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        .decision h1 {
            margin: 0;
            letter-spacing: 0.1em;
        }
        .signal-chip {
            display: inline-block;
            font-size: 12px;
            letter-spacing: 0.08em;
            border-radius: 999px;
            padding: 4px 10px;
            color: #07101d;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .small-text {
            opacity: 0.8;
            font-size: 13px;
        }
        .metric-big {
            font-size: 30px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def trading_plan(row: pd.Series) -> dict:
    """Build directional entry/stop/target levels using ATR."""
    px = float(row["latest_close"])
    atr = float(row["atr14"])
    signal = str(row["signal"])

    if signal == "BUY":
        return {
            "entry": px,
            "stop": px - 1.3 * atr,
            "target": px + 2.1 * atr,
            "note": "Trend continuation setup",
        }
    if signal == "SELL":
        return {
            "entry": px,
            "stop": px + 1.3 * atr,
            "target": px - 2.1 * atr,
            "note": "Downtrend momentum setup",
        }
    return {
        "entry": px,
        "stop": px - 0.7 * atr,
        "target": px + 0.7 * atr,
        "note": "No fresh edge; wait for confirmation",
    }


def format_signal_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    """Return BUY, SELL, HOLD counts safely."""
    return (
        int((df["live_signal"] == "BUY").sum()),
        int((df["live_signal"] == "SELL").sum()),
        int((df["live_signal"] == "HOLD").sum()),
    )


def infer_live_signal(live_df: pd.DataFrame, base_signal: str, base_confidence_pct: float) -> tuple[str, float, str]:
    """Infer intraday live signal from trend and momentum."""
    if live_df.empty or len(live_df) < 20:
        return str(base_signal), float(base_confidence_pct), "Sideways"

    close = live_df["Close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    start_px = float(close.iloc[0]) if float(close.iloc[0]) != 0.0 else 1e-6
    move_pct = ((float(close.iloc[-1]) / start_px) - 1.0) * 100.0

    score = 0
    score += 1 if float(ema20.iloc[-1]) > float(ema50.iloc[-1]) else -1
    if move_pct > 0.15:
        score += 1
    elif move_pct < -0.15:
        score -= 1

    if move_pct > 0.35:
        trend = "Strong Uptrend"
    elif move_pct > 0.08:
        trend = "Uptrend"
    elif move_pct < -0.35:
        trend = "Strong Downtrend"
    elif move_pct < -0.08:
        trend = "Downtrend"
    else:
        trend = "Sideways"

    if score >= 2:
        return "BUY", min(99.0, max(55.0, 58.0 + abs(move_pct) * 6.0)), trend
    if score <= -2:
        return "SELL", min(99.0, max(55.0, 58.0 + abs(move_pct) * 6.0)), trend

    hold_conf = max(50.0, min(95.0, float(base_confidence_pct)))
    return "HOLD", hold_conf, trend


def infer_market_trend_signal(live_df: pd.DataFrame) -> tuple[str, float, str]:
    """Infer a dedicated auto-trading signal from live trend only."""
    if live_df.empty or len(live_df) < 30:
        return "HOLD", 0.0, "Insufficient live data"

    close = live_df["Close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    recent = close.tail(12)
    start_px = float(recent.iloc[0]) if float(recent.iloc[0]) != 0.0 else 1e-6
    short_move_pct = ((float(recent.iloc[-1]) / start_px) - 1.0) * 100.0

    trend_score = 0
    trend_score += 1 if float(ema20.iloc[-1]) > float(ema50.iloc[-1]) else -1
    trend_score += 1 if short_move_pct > 0.10 else -1 if short_move_pct < -0.10 else 0

    gap_pct = ((float(ema20.iloc[-1]) - float(ema50.iloc[-1])) / (float(ema50.iloc[-1]) + 1e-12)) * 100.0
    confidence = min(99.0, max(55.0, 58.0 + abs(gap_pct) * 20.0 + abs(short_move_pct) * 4.0))

    if short_move_pct > 0.35:
        trend = "Strong Uptrend"
    elif short_move_pct > 0.08:
        trend = "Uptrend"
    elif short_move_pct < -0.35:
        trend = "Strong Downtrend"
    elif short_move_pct < -0.08:
        trend = "Downtrend"
    else:
        trend = "Sideways"

    if trend_score >= 2:
        return "BUY", confidence, trend
    if trend_score <= -2:
        return "SELL", confidence, trend
    return "HOLD", 50.0, trend


def enable_auto_refresh(refresh_seconds: int) -> None:
    """Auto-reload Streamlit page at a fixed interval."""
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {int(refresh_seconds) * 1000});
        </script>
        """,
        height=0,
        width=0,
    )


def main() -> None:
    st.set_page_config(page_title="AI Signal Dashboard", page_icon="📊", layout="wide")
    inject_theme()

    if "last_model_refresh" not in st.session_state:
        st.session_state["last_model_refresh"] = None

    st.markdown("## AI Signal Dashboard")
    st.caption("Live-style Buy / Sell / Hold intelligence with confidence, risk filters, and action levels")

    st.sidebar.markdown("### Live Update")
    live_refresh = st.sidebar.button("Refresh Live Prices", use_container_width=True)
    chart_window = st.sidebar.selectbox(
        "Live graph window",
        options=["2 Days", "3 Days"],
        index=1,
    )
    chart_interval = st.sidebar.selectbox(
        "Live graph interval",
        options=["1m", "5m", "15m", "30m"],
        index=1,
    )
    rerun_on_refresh = st.sidebar.checkbox("Rebuild Model On Refresh", value=False)
    use_live_signal = st.sidebar.checkbox("Use Live Signal Override", value=False)
    auto_refresh = st.sidebar.checkbox("Auto Refresh Live", value=False)
    refresh_seconds = st.sidebar.slider("Auto refresh (sec)", min_value=10, max_value=300, value=30, step=5)
    max_lag_minutes = st.sidebar.slider("Max acceptable lag (min)", min_value=1, max_value=120, value=20, step=1)
    st.sidebar.markdown("### Auto Paper Trading")
    auto_paper_trade = st.sidebar.checkbox("Enable Auto Paper Trade", value=True)
    reset_trade_log = st.sidebar.button("Reset Auto Trade Log", use_container_width=True)
    hold_minutes = st.sidebar.slider("Max hold duration (min)", min_value=5, max_value=360, value=120, step=5)
    close_on_timeout = st.sidebar.checkbox("Force close by hold time", value=False)
    min_reversal_hold = st.sidebar.slider("Min hold before reversal close (min)", min_value=1, max_value=120, value=10, step=1)
    stop_atr_mult = st.sidebar.slider("Stop loss ATR x", min_value=0.5, max_value=5.0, value=1.3, step=0.1)
    target_atr_mult = st.sidebar.slider("Take profit ATR x", min_value=0.5, max_value=6.0, value=2.1, step=0.1)
    min_trade_conf = st.sidebar.slider("Min confidence for entry %", min_value=40, max_value=95, value=60, step=1)
    capital_per_trade = st.sidebar.number_input("Capital per trade", min_value=100.0, max_value=50000.0, value=1000.0, step=100.0)
    cache_nonce = int(time.time() // 5)

    if live_refresh:
        if rerun_on_refresh:
            with st.spinner("Running model pipeline for fresh decisions..."):
                ok, msg = rebuild_report_from_model()
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
        load_live_price_series.clear()
        load_latest_quote.clear()
        st.sidebar.success("Live price cache refreshed.")

    if auto_refresh:
        enable_auto_refresh(refresh_seconds)
        st.sidebar.caption(f"Auto refresh active: every {refresh_seconds}s")

    if reset_trade_log:
        save_trade_log(pd.DataFrame(columns=TRADE_LOG_COLUMNS))
        st.sidebar.success("Auto trade log reset. New entries will follow current trend strategy.")

    if not REPORT_PATH.exists():
        st.error("No report found. Run: python main.py")
        return

    report = load_report(REPORT_PATH)
    assets = report.get("assets", [])
    selected_markets = report.get("meta", {}).get("selected_markets", [])
    eval_summary = report.get("evaluation_summary", {})
    calibration_summary = eval_summary.get("calibration", {}) if isinstance(eval_summary, dict) else {}
    research_accuracy = float(calibration_summary.get("holdout_accuracy_pct", eval_summary.get("overall_accuracy_pct", 0.0)))
    if not assets:
        st.warning("Report exists, but contains no asset rows.")
        return

    df = pd.DataFrame(assets)
    if selected_markets:
        df = df[df["asset"].isin(selected_markets)].copy()
    df = df.head(6).copy()

    if len(df) < 6:
        st.warning("Report data is stale or incomplete. Run python main.py to refresh the 6-market report.")
        return

    df["risk"] = df["outlier_flagged"].map({True: "High", False: "Normal"})
    df["confidence_pct"] = (df["confidence"] * 100.0).round(2)
    df["weight_pct"] = (df["portfolio_weight"] * 100.0).round(2)

    if not auto_refresh:
        st.sidebar.caption("Auto refresh disabled. Use Refresh Live Prices for clean updates.")
    if not use_live_signal:
        st.sidebar.caption("Model mode active: dashboard signals follow python main.py output.")
    if not rerun_on_refresh:
        st.sidebar.caption("Fast mode active: Refresh updates prices without retraining model.")

    filtered = df.copy()
    live_assets_to_fetch = set(filtered["asset"].astype(str).tolist())

    period_value = "2d" if chart_window == "2 Days" else "3d"

    live_snapshots = {}
    for _, row in filtered.iterrows():
        asset_name = str(row["asset"])
        if asset_name in live_assets_to_fetch:
            live_df = load_live_price_series(
                str(row["symbol"]), period=period_value, interval=chart_interval, nonce=cache_nonce
            )
            quote = load_latest_quote(str(row["symbol"]), nonce=cache_nonce)
        else:
            live_df = pd.DataFrame(columns=["Close"])
            quote = {"price": None, "timestamp": None, "market_state": "Skipped (fast mode)"}

        live_price = float(row["latest_close"])
        if quote.get("price") is not None:
            live_price = float(quote["price"])
        elif not live_df.empty:
            live_price = float(live_df["Close"].iloc[-1])

        if use_live_signal:
            live_signal, live_confidence_pct, live_trend = infer_live_signal(
                live_df=live_df,
                base_signal=str(row["signal"]),
                base_confidence_pct=float(row["confidence_pct"]),
            )
            if live_df.empty:
                live_trend = "No intraday feed"
        else:
            live_signal = str(row["signal"])
            live_confidence_pct = float(row["confidence_pct"])
            live_trend = "Model-based"

        auto_signal, auto_confidence_pct, auto_trend = infer_market_trend_signal(live_df)

        live_snapshots[asset_name] = {
            "live_df": live_df,
            "quote": quote,
            "live_price": live_price,
            "live_signal": live_signal,
            "live_confidence_pct": live_confidence_pct,
            "live_trend": live_trend,
            "auto_signal": auto_signal,
            "auto_confidence_pct": auto_confidence_pct,
            "auto_trend": auto_trend,
        }

    filtered["live_signal"] = filtered["asset"].map(lambda x: live_snapshots[str(x)]["live_signal"])
    filtered["live_confidence_pct"] = filtered["asset"].map(
        lambda x: float(live_snapshots[str(x)]["live_confidence_pct"])
    )
    filtered["live_price"] = filtered["asset"].map(lambda x: float(live_snapshots[str(x)]["live_price"]))
    filtered["live_trend"] = filtered["asset"].map(lambda x: live_snapshots[str(x)]["live_trend"])

    trade_log_df = load_trade_log()
    if auto_paper_trade:
        trade_log_df = run_auto_trade_cycle(
            snapshot_df=filtered,
            live_snapshots=live_snapshots,
            hold_minutes=int(hold_minutes),
            min_confidence_pct=float(min_trade_conf),
            capital_per_trade=float(capital_per_trade),
            close_on_timeout=bool(close_on_timeout),
            stop_atr_mult=float(stop_atr_mult),
            target_atr_mult=float(target_atr_mult),
            min_reversal_hold_minutes=int(min_reversal_hold),
        )

    closed_trades = trade_log_df[trade_log_df["status"] == "CLOSED"].copy()
    open_trades = trade_log_df[trade_log_df["status"] == "OPEN"].copy()
    realized_pnl = float(pd.to_numeric(closed_trades["pnl"], errors="coerce").fillna(0.0).sum()) if not closed_trades.empty else 0.0
    closed_count = int(len(closed_trades))
    right_count = int(closed_trades["outcome"].eq("RIGHT").sum()) if not closed_trades.empty else 0
    wrong_count = int(closed_trades["outcome"].eq("WRONG").sum()) if not closed_trades.empty else 0
    decision_accuracy = (right_count / closed_count * 100.0) if closed_count > 0 else 0.0
    right_accuracy_pct = (right_count / closed_count * 100.0) if closed_count > 0 else 0.0
    wrong_accuracy_pct = (wrong_count / closed_count * 100.0) if closed_count > 0 else 0.0

    focus = filtered.iloc[0]
    signal_info = SIGNAL_STYLE.get(str(focus["live_signal"]), SIGNAL_STYLE["HOLD"])
    focus_for_plan = focus.copy()
    focus_for_plan["signal"] = focus["live_signal"]
    focus_for_plan["latest_close"] = focus["live_price"]
    plan = trading_plan(focus_for_plan)

    buy_count, sell_count, hold_count = format_signal_counts(filtered)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Markets", int(len(filtered)))
    m2.metric("BUY", buy_count)
    m3.metric("SELL", sell_count)
    m4.metric("HOLD", hold_count)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Auto Open Positions", int(len(open_trades)))
    a2.metric("Auto Closed Trades", closed_count)
    a3.metric("Auto Realized PnL", f"{realized_pnl:+.2f}")
    a4.metric("Decision Accuracy", f"{decision_accuracy:.2f}%")

    e1, e2, e3 = st.columns(3)
    e1.metric("Model Samples", int(eval_summary.get("samples", 0)))
    e2.metric("Prediction Accuracy", f"{float(eval_summary.get('overall_accuracy_pct', 0.0)):.2f}%")
    e3.metric("Active Trade Accuracy", f"{float(eval_summary.get('active_trade_accuracy_pct', 0.0)):.2f}%")

    left, right = st.columns([2.3, 1.2])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="signal-chip" style="background:{signal_info['accent']};">{focus['symbol']}</div>
            <div class="small-text">{focus['asset']} | Last updated: {report.get('generated_at', '-')}</div>
            <div class="metric-big">{focus['live_price']:.5f}</div>
            <div class="small-text">Live Confidence {focus['live_confidence_pct']:.2f}% | Portfolio Weight {focus['weight_pct']:.2f}%</div>
            <div class="small-text">Trend: {focus['live_trend']}</div>
            """,
            unsafe_allow_html=True,
        )

        chart_df = filtered[["asset", "confidence_pct", "weight_pct"]].set_index("asset")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Confidence by Market")
            st.bar_chart(chart_df[["confidence_pct"]], use_container_width=True)
        with c2:
            st.markdown("Allocation by Market")
            st.bar_chart(chart_df[["weight_pct"]], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            f"""
            <div class="decision" style="background:{signal_info['bg']};">
                <div class="small-text">TRADING SIGNAL</div>
                <h1 style="color:{signal_info['accent']};">{signal_info['label']}</h1>
                <div class="small-text">{plan['note']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(
            min(max(float(focus["live_confidence_pct"]) / 100.0, 0.0), 1.0),
            text=f"Live confidence: {focus['live_confidence_pct']:.2f}%",
        )

        c_entry, c_stop, c_target = st.columns(3)
        c_entry.metric("Entry", f"{plan['entry']:.5f}")
        c_stop.metric("Stop", f"{plan['stop']:.5f}")
        c_target.metric("Target", f"{plan['target']:.5f}")
        st.metric("Anomaly", "YES" if bool(focus["outlier_flagged"]) else "NO")

    st.markdown("### Trade Board")
    board = filtered[
        [
            "asset",
            "symbol",
            "live_price",
            "live_signal",
            "live_confidence_pct",
            "live_trend",
            "atr14",
            "weight_pct",
            "test_accuracy_pct",
            "risk",
            "decision_reason",
        ]
    ].rename(
        columns={
            "asset": "Market",
            "symbol": "Ticker",
            "live_price": "Live Price",
            "live_signal": "Live Action",
            "live_confidence_pct": "Live Confidence %",
            "live_trend": "Trend",
            "atr14": "ATR14",
            "weight_pct": "Allocation %",
            "test_accuracy_pct": "Test Accuracy %",
            "risk": "Risk",
            "decision_reason": "Reason",
        }
    )
    st.dataframe(board.sort_values("Allocation %", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("### Auto Trade Journal")
    if closed_trades.empty and open_trades.empty:
        st.info("No auto paper trades yet. Keep Auto Paper Trade enabled and refresh live data.")
    else:
        if not open_trades.empty:
            st.markdown("Open Positions")
            st.dataframe(
                open_trades[
                    [
                        "asset",
                        "symbol",
                        "side",
                        "entry_price",
                        "entry_confidence_pct",
                        "entry_ts_utc",
                    ]
                ].rename(
                    columns={
                        "asset": "Market",
                        "symbol": "Ticker",
                        "side": "Side",
                        "entry_price": "Entry",
                        "entry_confidence_pct": "Confidence %",
                        "entry_ts_utc": "Entry Time (UTC)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        if not closed_trades.empty:
            st.markdown("Closed Positions (Right/Wrong + Profit)")
            c1, c2, c3 = st.columns(3)
            c1.metric("RIGHT Decisions", right_count, f"{right_accuracy_pct:.2f}%")
            c2.metric("Decision Accuracy", f"{decision_accuracy:.2f}%")
            c3.metric("Research Accuracy", f"{research_accuracy:.2f}%")
            closed_view = closed_trades.copy()
            closed_view["pnl"] = pd.to_numeric(closed_view["pnl"], errors="coerce").fillna(0.0)
            closed_view["pnl_pct"] = pd.to_numeric(closed_view["pnl_pct"], errors="coerce").fillna(0.0)
            st.dataframe(
                closed_view.sort_values("exit_ts_utc", ascending=False)[
                    [
                        "asset",
                        "symbol",
                        "side",
                        "entry_price",
                        "exit_price",
                        "pnl",
                        "pnl_pct",
                        "outcome",
                        "close_reason",
                        "entry_ts_utc",
                        "exit_ts_utc",
                    ]
                ].rename(
                    columns={
                        "asset": "Market",
                        "symbol": "Ticker",
                        "side": "Side",
                        "entry_price": "Entry",
                        "exit_price": "Exit",
                        "pnl": "Profit",
                        "pnl_pct": "Profit %",
                        "outcome": "Decision",
                        "close_reason": "Closed By",
                        "entry_ts_utc": "Entry Time (UTC)",
                        "exit_ts_utc": "Exit Time (UTC)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("### Live Market Movement")
    movement_assets = filtered[filtered["asset"].astype(str).isin(live_assets_to_fetch)][
        ["asset", "symbol", "live_signal", "live_confidence_pct", "live_price", "atr14", "signal"]
    ].sort_values(
        "live_confidence_pct", ascending=False
    )

    for _, row in movement_assets.iterrows():
        row_for_plan = row.copy()
        row_for_plan["signal"] = row["live_signal"]
        row_for_plan["latest_close"] = row["live_price"]
        row_plan = trading_plan(row_for_plan)

        snapshot = live_snapshots.get(str(row["asset"]), {})
        live_df = snapshot.get("live_df", pd.DataFrame(columns=["Close"]))
        if live_df.empty:
            st.warning(f"No live intraday data available for {row['asset']} right now.")
            continue

        quote = snapshot.get("quote", {"price": None, "timestamp": None, "market_state": "Unknown"})
        normalized_df = build_normalized_movement_frame(live_df)

        start_px = float(live_df["Close"].iloc[0])
        end_px_chart = float(live_df["Close"].iloc[-1])
        end_px = float(quote["price"]) if quote.get("price") is not None else end_px_chart
        delta = end_px - start_px
        delta_pct = (delta / start_px * 100.0) if start_px else 0.0

        last_tick_raw = pd.to_datetime(quote["timestamp"]) if quote.get("timestamp") is not None else pd.to_datetime(live_df.index[-1])
        if last_tick_raw.tzinfo is None:
            last_tick_raw = last_tick_raw.tz_localize("UTC")
        tick_utc = last_tick_raw.tz_convert("UTC")
        now_utc = pd.Timestamp.now(tz="UTC")
        lag_min = max((now_utc - tick_utc).total_seconds() / 60.0, 0.0)

        if lag_min > float(max_lag_minutes):
            forced_quote = load_latest_quote_uncached(str(row["symbol"]))
            if forced_quote.get("price") is not None and forced_quote.get("timestamp") is not None:
                quote = forced_quote
                end_px = float(forced_quote["price"])
                last_tick_raw = pd.to_datetime(forced_quote["timestamp"])
                if last_tick_raw.tzinfo is None:
                    last_tick_raw = last_tick_raw.tz_localize("UTC")
                tick_utc = last_tick_raw.tz_convert("UTC")
                lag_min = max((now_utc - tick_utc).total_seconds() / 60.0, 0.0)

        local_tz = datetime.now().astimezone().tzinfo
        last_tick_local = tick_utc.tz_convert(local_tz)
        now_local = datetime.now().astimezone()

        title_col, signal_col, chart_mode_col = st.columns([2.0, 0.8, 1.6])
        with title_col:
            st.markdown(f"#### {row['asset']} | {row['symbol']}")
        with signal_col:
            st.markdown(
                f"<div class='signal-chip' style='background:{SIGNAL_STYLE[str(row['live_signal'])]['accent']};'>{row['live_signal']}</div>",
                unsafe_allow_html=True,
            )
        with chart_mode_col:
            chart_mode = st.radio(
                "Chart",
                options=["Price", "Price + Volume"],
                index=0,
                horizontal=True,
                key=f"chart_mode_{row['asset']}",
                label_visibility="collapsed",
            )

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Current", f"{end_px:.5f}", delta=f"{delta:+.5f}")
        g2.metric(f"Move % ({chart_window})", f"{delta_pct:+.2f}%")
        g3.metric("Live Signal", f"{row['live_signal']} ({row['live_confidence_pct']:.2f}%)")
        g4.metric("Data Lag", f"{lag_min:.1f} min")

        st.caption(
            " | ".join(
                [
                    "Feed: Yahoo Finance",
                    f"Live Price: {end_px:.5f}",
                    f"Window: {chart_window}",
                    f"Interval: {chart_interval}",
                    f"Last tick (local): {last_tick_local.strftime('%Y-%m-%d %H:%M:%S %Z')}",
                    f"Now (local): {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}",
                    f"State: {quote.get('market_state', 'Unknown')}",
                ]
            )
        )

        st.line_chart(
            normalized_df[["MovePct", "MovePctEMA"]],
            use_container_width=True,
        )

        chart_df = build_live_chart_frame(live_df, row_plan)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], name="Close", mode="lines", hovertemplate="<b>Time:</b> %{x}<br><b>Close:</b> %{y:.5f}<extra></extra>"))
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"], name="EMA20", mode="lines", hovertemplate="<b>Time:</b> %{x}<br><b>EMA20:</b> %{y:.5f}<extra></extra>"))
        fig_price.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"], name="EMA50", mode="lines", hovertemplate="<b>Time:</b> %{x}<br><b>EMA50:</b> %{y:.5f}<extra></extra>"))
        fig_price.add_hline(y=float(plan["entry"]), line_dash="dash", line_color="green", name="Entry")
        fig_price.add_hline(y=float(plan["stop"]), line_dash="dash", line_color="red", name="Stop")
        fig_price.add_hline(y=float(plan["target"]), line_dash="dash", line_color="blue", name="Target")
        fig_price.update_layout(
            title=f"Price Action: {row['asset']}",
            xaxis_title="Time",
            yaxis_title="Price",
            hovermode="x unified",
            height=400,
            template="plotly_dark",
        )
        st.plotly_chart(fig_price, use_container_width=True)

        if chart_mode == "Price + Volume":
            st.markdown("Volume Activity")
            if "Volume" in live_df.columns:
                vol_series = live_df["Volume"].astype(float).fillna(0.0)
                if vol_series.sum() > 0:
                    v1, v2 = st.columns(2)
                    v1.metric("Latest Volume", f"{vol_series.iloc[-1]:,.0f}")
                    v2.metric("Avg Volume", f"{vol_series.mean():,.0f}")
                    fig_vol = go.Figure(
                        data=[go.Bar(
                            x=live_df.index,
                            y=vol_series,
                            name="Volume",
                            marker_color="lightblue",
                            hovertemplate="<b>Time:</b> %{x}<br><b>Volume:</b> %{y:,.0f}<extra></extra>",
                        )]
                    )
                    fig_vol.update_layout(
                        title=f"Volume: {row['asset']}",
                        xaxis_title="Time",
                        yaxis_title="Volume",
                        hovermode="x unified",
                        height=300,
                        template="plotly_dark",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
                else:
                    st.info("Volume feed unavailable or zero for this market at current interval.")
            else:
                st.info("Volume column not provided by data source for this market.")

    st.markdown("### Explainability")
    for _, row in filtered.sort_values("confidence_pct", ascending=False).iterrows():
        with st.expander(
            f"{row['asset']} | Base Model: {row['signal']} ({row['confidence_pct']:.2f}%) | Live: {row['live_signal']} ({row['live_confidence_pct']:.2f}%)"
        ):
            st.write(f"Reason: {row.get('decision_reason', 'N/A')}")
            for line in row["top_features"]:
                st.write(f"- {line}")

    corr = report.get("correlation_matrix", {})
    if corr:
        st.markdown("### Correlation Matrix")
        st.dataframe(pd.DataFrame(corr).round(4), use_container_width=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Trade Board CSV",
            data=board.to_csv(index=False),
            file_name="trade_board.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download Full Report JSON",
            data=json.dumps(report, indent=2),
            file_name="portfolio_report.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
