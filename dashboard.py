"""Streamlit dashboard for ExplainInvest portfolio management."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import yfinance as yf

from main import run_pipeline_collect


REPORT_PATH = Path("reports/portfolio_report.json")

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
    live_render_mode = st.sidebar.selectbox(
        "Live render mode",
        options=["Focus only (fast)", "Top signals", "All filtered"],
        index=0,
    )
    top_live_assets = st.sidebar.slider("Top live assets", min_value=1, max_value=10, value=4, step=1)

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

    if not REPORT_PATH.exists():
        st.error("No report found. Run: python main.py")
        return

    report = load_report(REPORT_PATH)
    assets = report.get("assets", [])
    if not assets:
        st.warning("Report exists, but contains no asset rows.")
        return

    df = pd.DataFrame(assets)
    df["risk"] = df["outlier_flagged"].map({True: "High", False: "Normal"})
    df["confidence_pct"] = (df["confidence"] * 100.0).round(2)
    df["weight_pct"] = (df["portfolio_weight"] * 100.0).round(2)

    st.sidebar.markdown("### Trading Controls")
    symbols = df["asset"].tolist()
    selected_asset = st.sidebar.selectbox("Focus Market", options=symbols, index=0)
    signal_filter = st.sidebar.multiselect(
        "Signals",
        options=["BUY", "SELL", "HOLD"],
        default=["BUY", "SELL", "HOLD"],
    )
    risk_filter = st.sidebar.multiselect("Risk", options=["Normal", "High"], default=["Normal", "High"])
    min_conf = st.sidebar.slider("Minimum Confidence %", 0, 100, 0)

    if not auto_refresh:
        st.sidebar.caption("Auto refresh disabled. Use Refresh Live Prices for clean updates.")
    if not use_live_signal:
        st.sidebar.caption("Model mode active: dashboard signals follow python main.py output.")
    if not rerun_on_refresh:
        st.sidebar.caption("Fast mode active: Refresh updates prices without retraining model.")

    filtered = df[
        df["signal"].isin(signal_filter)
        & df["risk"].isin(risk_filter)
        & (df["confidence_pct"] >= min_conf)
    ].copy()

    if filtered.empty:
        st.warning("No markets match current filters. Relax controls in the sidebar.")
        return

    if selected_asset not in filtered["asset"].values:
        selected_asset = filtered.iloc[0]["asset"]

    if live_render_mode == "Focus only (fast)":
        live_assets_to_fetch = {str(selected_asset)}
    elif live_render_mode == "Top signals":
        live_assets_to_fetch = set(
            filtered.sort_values("confidence_pct", ascending=False)["asset"].head(int(top_live_assets)).astype(str).tolist()
        )
    else:
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

        live_snapshots[asset_name] = {
            "live_df": live_df,
            "quote": quote,
            "live_price": live_price,
            "live_signal": live_signal,
            "live_confidence_pct": live_confidence_pct,
            "live_trend": live_trend,
        }

    filtered["live_signal"] = filtered["asset"].map(lambda x: live_snapshots[str(x)]["live_signal"])
    filtered["live_confidence_pct"] = filtered["asset"].map(
        lambda x: float(live_snapshots[str(x)]["live_confidence_pct"])
    )
    filtered["live_price"] = filtered["asset"].map(lambda x: float(live_snapshots[str(x)]["live_price"]))
    filtered["live_trend"] = filtered["asset"].map(lambda x: live_snapshots[str(x)]["live_trend"])

    focus = filtered[filtered["asset"] == selected_asset].iloc[0]
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
            "risk",
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
            "risk": "Risk",
        }
    )
    st.dataframe(board.sort_values("Allocation %", ascending=False), use_container_width=True, hide_index=True)

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
