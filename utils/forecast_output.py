"""
utils/forecast_output.py
Forecast generation, prediction intervals, Erlang C staffing conversion,
what-if scenario analysis, and export utilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from config import CHART_COLORS, PLOTLY_TEMPLATE


# ── Future Forecast Generation ────────────────────────────────────────────────

def generate_forecast(model_result: dict, horizon: int,
                      train: pd.Series, freq: str) -> dict:
    """
    Generate a future forecast from a fitted model result.
    Returns dict with point forecast and prediction intervals where available.
    """
    key       = model_result.get("key")
    model_obj = model_result.get("model_obj")

    if model_obj is None:
        return {"success": False, "error": "No fitted model object found."}

    try:
        future_index = pd.date_range(
            start=train.index[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )

        # ── ARIMA / SARIMA ────────────────────────────────────────────────────
        if key == "arima":
            conf_int = model_obj.predict(n_periods=horizon, return_conf_int=True)
            point    = pd.Series(conf_int[0],       index=future_index)
            lower    = pd.Series(conf_int[1][:, 0], index=future_index)
            upper    = pd.Series(conf_int[1][:, 1], index=future_index)
            return {"success": True, "point": point, "lower": lower, "upper": upper}

        # ── ETS / Holt-Winters ────────────────────────────────────────────────
        elif key in ("ets", "holtwinters"):
            # HoltWintersResults uses forecast() + simulate() for intervals,
            # not get_forecast(). Use forecast() for the point estimate and
            # derive approximate intervals from in-sample residual std.
            point_vals = model_obj.forecast(horizon)
            point      = pd.Series(point_vals.values, index=future_index)

            # Approximate 80% prediction interval from residual std
            resid_std  = np.std(model_obj.resid.dropna())
            z_80       = 1.282  # 80% two-sided
            lower      = point - z_80 * resid_std
            upper      = point + z_80 * resid_std
            return {"success": True, "point": point, "lower": lower, "upper": upper}

        # ── Prophet ───────────────────────────────────────────────────────────
        elif key == "prophet":
            future = model_obj.make_future_dataframe(periods=horizon, freq=freq)
            fc     = model_obj.predict(future)
            fc_fut = fc.tail(horizon)
            point  = pd.Series(fc_fut["yhat"].values,       index=future_index)
            lower  = pd.Series(fc_fut["yhat_lower"].values, index=future_index)
            upper  = pd.Series(fc_fut["yhat_upper"].values, index=future_index)
            return {"success": True, "point": point, "lower": lower, "upper": upper}

        # ── ML models (XGBoost / LightGBM / RF / Ridge) ───────────────────────
        elif key in ("xgboost", "lightgbm", "randomforest", "ridge"):
            # ML models require features for future intervals
            # Build calendar features only (no lags available for true future)
            feature_cols = model_result.get("feature_cols", [])
            if not feature_cols:
                return {"success": False, "error": "Feature column list not stored in model result."}

            df_future = pd.DataFrame(index=future_index)
            cal_map = {
                "hour":           lambda idx: idx.hour,
                "day_of_week":    lambda idx: idx.dayofweek,
                "day_of_month":   lambda idx: idx.day,
                "week_of_month":  lambda idx: (idx.day - 1) // 7 + 1,
                "month":          lambda idx: idx.month,
                "is_payday_week": lambda idx: (((idx.day >= 14) & (idx.day <= 16)) | (idx.day >= 28)).astype(int),
            }

            for col in feature_cols:
                if col in cal_map:
                    df_future[col] = cal_map[col](df_future.index)
                else:
                    df_future[col] = 0  # lag/rolling features unknown for true future; fill with 0

            X_future = df_future[feature_cols]

            # Handle Ridge scaling
            if key == "ridge" and hasattr(model_obj, "_scaler"):
                X_future = model_obj._scaler.transform(X_future)

            point_vals = model_obj.predict(X_future)
            point      = pd.Series(point_vals, index=future_index)

            # Approximate intervals using residual std from training
            residual_std = model_result.get("residual_std", point.std() * 0.1)
            lower = point - 1.645 * residual_std
            upper = point + 1.645 * residual_std

            return {"success": True, "point": point, "lower": lower, "upper": upper}

        else:
            return {"success": False, "error": f"Unrecognised model key: {key}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Ensemble ──────────────────────────────────────────────────────────────────

def build_ensemble(forecasts: list, weights: list = None) -> pd.Series:
    """
    Combine multiple point forecasts into an ensemble.
    If weights is None, uses simple average.
    """
    valid = [f["point"] for f in forecasts if f.get("success") and "point" in f]
    if not valid:
        return pd.Series(dtype=float)

    aligned = pd.concat(valid, axis=1).dropna()

    if weights and len(weights) == aligned.shape[1]:
        w = np.array(weights)
        w = w / w.sum()
        return (aligned * w).sum(axis=1)

    return aligned.mean(axis=1)


# ── Erlang C ──────────────────────────────────────────────────────────────────

def erlang_c(agents: int, intensity: float) -> float:
    """
    Compute Erlang C probability — probability that a call has to wait.
    agents: number of agents
    intensity: traffic intensity = arrival_rate * aht (in same time unit)
    """
    if agents <= intensity:
        return 1.0

    from math import factorial, exp

    erlang_b_inv = 1.0
    for i in range(1, agents + 1):
        erlang_b_inv = 1.0 + erlang_b_inv * i / intensity

    erlang_b = 1.0 / erlang_b_inv
    ec = (agents * erlang_b) / (agents - intensity * (1 - erlang_b))
    return min(ec, 1.0)


def required_agents(
    volume_per_interval: float,
    aht_seconds: float,
    interval_seconds: float,
    target_sl: float,
    target_answer_seconds: float,
    shrinkage: float,
    max_agents: int = 500,
) -> dict:
    """
    Compute minimum agents required to meet the target service level.
    Uses Erlang C iteratively.

    Returns dict with agents_required, sl_achieved, occupancy, and raw_agents.
    """
    if volume_per_interval <= 0:
        return {"agents_required": 0, "raw_agents": 0, "sl_achieved": 1.0, "occupancy": 0.0}

    intensity = (volume_per_interval * aht_seconds) / interval_seconds

    best = {"agents": int(np.ceil(intensity)) + 1, "sl": 0.0}

    for n in range(max(1, int(np.ceil(intensity))), max_agents + 1):
        ec = erlang_c(n, intensity)
        sl = 1 - ec * np.exp(-(n - intensity) * (target_answer_seconds / aht_seconds))
        sl = max(0.0, min(sl, 1.0))
        if sl >= target_sl:
            best = {"agents": n, "sl": sl}
            break

    raw_agents = best["agents"]
    staffed    = int(np.ceil(raw_agents / (1 - shrinkage)))
    occupancy  = intensity / raw_agents if raw_agents > 0 else 0.0

    return {
        "raw_agents":       raw_agents,
        "agents_required":  staffed,
        "sl_achieved":      round(best["sl"], 4),
        "occupancy":        round(occupancy, 4),
    }


def build_staffing_plan(
    forecast_series: pd.Series,
    aht_seconds: float,
    interval_seconds: float,
    target_sl: float,
    target_answer_seconds: float,
    shrinkage: float,
) -> pd.DataFrame:
    """
    Apply Erlang C to every interval in the forecast to produce a full staffing plan.
    """
    rows = []
    for ts, vol in forecast_series.items():
        r = required_agents(
            volume_per_interval=max(0, vol),
            aht_seconds=aht_seconds,
            interval_seconds=interval_seconds,
            target_sl=target_sl,
            target_answer_seconds=target_answer_seconds,
            shrinkage=shrinkage,
        )
        rows.append({
            "Timestamp":        ts,
            "Forecast Volume":  round(max(0, vol), 1),
            "Raw Agents":       r["raw_agents"],
            "Agents Required":  r["agents_required"],
            "SL Achieved":      f"{r['sl_achieved']*100:.1f}%",
            "Occupancy":        f"{r['occupancy']*100:.1f}%",
        })

    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_forecast(historical: pd.Series, point: pd.Series,
                  lower: pd.Series = None, upper: pd.Series = None,
                  model_name: str = "Forecast") -> go.Figure:
    """
    Historical + forecast plot with optional prediction interval band.
    """
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical.index, y=historical,
        mode="lines",
        line=dict(color=CHART_COLORS["primary"], width=1.8),
        name="Historical",
    ))

    # Prediction interval
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(point.index) + list(point.index[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(200,151,78,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Prediction Interval",
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=point.index, y=point,
        mode="lines",
        line=dict(color=CHART_COLORS["accent"], width=2),
        name=f"{model_name} Forecast",
    ))

    # Divider between history and forecast — use a shape instead of add_vline
    # to avoid Plotly annotation arithmetic failing on datetime x-axes.
    if len(historical) > 0:
        cutoff = str(historical.index[-1])
        fig.add_shape(
            type="line",
            x0=cutoff, x1=cutoff,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dot", color=CHART_COLORS["muted"], width=1.5),
        )
        fig.add_annotation(
            x=cutoff, y=1,
            xref="x", yref="paper",
            text="Forecast start",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color=CHART_COLORS["muted"]),
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def plot_staffing_plan(staffing_df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis chart: forecast volume (bar) + required agents (line).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=staffing_df["Timestamp"],
        y=staffing_df["Forecast Volume"],
        name="Forecast Volume",
        marker_color=CHART_COLORS["secondary"],
        opacity=0.6,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=staffing_df["Timestamp"],
        y=staffing_df["Agents Required"],
        mode="lines+markers",
        line=dict(color=CHART_COLORS["accent"], width=2),
        marker=dict(size=4),
        name="Agents Required",
    ), secondary_y=True)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=50),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    fig.update_yaxes(title_text="Forecast Volume",  secondary_y=False)
    fig.update_yaxes(title_text="Agents Required",  secondary_y=True)

    return fig


# ── Export ────────────────────────────────────────────────────────────────────

def export_forecast_csv(forecast_df: pd.DataFrame) -> bytes:
    return forecast_df.to_csv(index=False).encode("utf-8")


def export_staffing_excel(forecast_df: pd.DataFrame, staffing_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        forecast_df.to_excel(writer,  sheet_name="Forecast",       index=False)
        staffing_df.to_excel(writer,  sheet_name="Staffing Plan",  index=False)
    return buffer.getvalue()