"""Flask dashboard for ExplainInvest portfolio management."""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

from flask import Flask, jsonify, redirect, render_template, request, url_for

from main import run_pipeline_collect


app = Flask(__name__)
_state_lock = Lock()
_state: Dict[str, Any] = {
    "result": None,
    "last_error": None,
    "is_running": False,
}


def _calculate_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build dashboard-level summary metrics from per-asset rows."""
    assets = result.get("assets", [])
    buy_count = sum(1 for row in assets if row.get("signal") == "BUY")
    sell_count = sum(1 for row in assets if row.get("signal") == "SELL")
    hold_count = sum(1 for row in assets if row.get("signal") == "HOLD")

    active_positions = [row for row in assets if row.get("portfolio_weight", 0.0) > 0.0]
    outlier_count = sum(1 for row in assets if row.get("outlier_flagged"))

    return {
        "asset_count": len(assets),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "active_positions": len(active_positions),
        "outlier_count": outlier_count,
        "total_weight_pct": sum(row.get("portfolio_weight", 0.0) for row in assets) * 100.0,
    }


@app.get("/")
def dashboard() -> str:
    """Render the portfolio dashboard with latest computed results."""
    with _state_lock:
        result = _state.get("result")
        error = _state.get("last_error")
        is_running = _state.get("is_running", False)

    summary: Optional[Dict[str, Any]] = None
    if result:
        summary = _calculate_summary(result)

    return render_template(
        "index.html",
        result=result,
        summary=summary,
        error=error,
        is_running=is_running,
    )


@app.post("/refresh")
def refresh() -> Any:
    """Run the full pipeline and update cached dashboard data."""
    with _state_lock:
        if _state["is_running"]:
            return redirect(url_for("dashboard"))
        _state["is_running"] = True
        _state["last_error"] = None

    try:
        result = run_pipeline_collect()
        with _state_lock:
            _state["result"] = result
    except Exception as exc:  # pragma: no cover
        with _state_lock:
            _state["last_error"] = str(exc)
    finally:
        with _state_lock:
            _state["is_running"] = False

    if request.headers.get("Accept") == "application/json":
        with _state_lock:
            if _state.get("result") is None:
                return jsonify({"ok": False, "error": _state.get("last_error")}), 500
            return jsonify({"ok": True, "data": _state["result"]})

    return redirect(url_for("dashboard"))


@app.get("/api/results")
def api_results() -> Any:
    """Return latest cached results in JSON format."""
    with _state_lock:
        result = _state.get("result")
        error = _state.get("last_error")

    if result is None:
        return jsonify({"ok": False, "error": error or "No results yet. Trigger /refresh first."}), 404
    return jsonify({"ok": True, "data": result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
