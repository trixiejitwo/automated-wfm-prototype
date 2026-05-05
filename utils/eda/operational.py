"""
utils/eda/operational.py
Operational metrics analysis: AHT, abandonment, SL%, occupancy, shrinkage.
Only activated when optional operational columns are present.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import CHART_COLORS, PLOTLY_TEMPLATE


def check_available_metrics(df: pd.DataFrame) -> dict:
    """
    Return which optional operational columns are present.
    """
    checks = {
        "aht":          any(c in df.columns for c in ["aht", "handle_time", "avg_handle_time"]),
        "abandoned":    any(c in df.columns for c in ["abandoned", "abandons", "abandon_count"]),
        "sl_pct":       any(c in df.columns for c in ["sl_pct", "service_level", "sl"]),
        "occupancy":    any(c in df.columns for c in ["occupancy", "occupancy_pct"]),
        "headcount":    any(c in df.columns for c in ["headcount", "agents", "ftes"]),
    }
    return checks


def get_col(df: pd.DataFrame, candidates: list):
    """Return the first matching column name from candidates, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_aht_trend(df: pd.DataFrame) -> go.Figure:
    """
    AHT over time with rolling 7-period average.
    Flags AHT creep which silently degrades service level.
    """
    col = get_col(df, ["aht", "handle_time", "avg_handle_time"])
    if col is None:
        return None

    series = df[col].dropna()
    roll   = series.rolling(7).mean()

    # Linear trend
    x_num = np.arange(len(series))
    slope, intercept, *_ = np.polyfit(x_num, series.values, 1), None
    slope = np.polyfit(x_num, series.values, 1)[0]
    trend_direction = "increasing" if slope > 0 else "decreasing"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series,
        mode="lines",
        line=dict(color=CHART_COLORS["neutral"], width=1),
        opacity=0.5,
        name="AHT (raw)",
    ))
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll,
        mode="lines",
        line=dict(color=CHART_COLORS["accent"], width=2),
        name="7-period Rolling Avg",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        xaxis_title="Date",
        yaxis_title="AHT (seconds)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=20, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
        title=dict(
            text=f"AHT Trend — {trend_direction.upper()}",
            font=dict(
                color=CHART_COLORS["warning"] if trend_direction == "increasing" else CHART_COLORS["success"],
                size=12,
            ),
        ),
    )
    return fig


def plot_aht_volume_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of AHT vs. Volume to detect the agent stress effect:
    high volume intervals where agents rush (AHT drops) or burn out (AHT rises).
    """
    aht_col = get_col(df, ["aht", "handle_time", "avg_handle_time"])
    if aht_col is None or "volume" not in df.columns:
        return None

    plot_df = df[["volume", aht_col]].dropna()

    fig = px.scatter(
        plot_df,
        x="volume",
        y=aht_col,
        trendline="ols",
        color_discrete_sequence=[CHART_COLORS["secondary"]],
        labels={"volume": "Volume", aht_col: "AHT (seconds)"},
    )
    fig.update_traces(marker=dict(size=4, opacity=0.5))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(t=20, b=20, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def plot_sl_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Hour x DOW heatmap of service level achievement.
    Reveals where SL is consistently missed — the most actionable WFM visual.
    """
    col = get_col(df, ["sl_pct", "service_level", "sl"])
    if col is None:
        return None

    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek

    dow_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.groupby(["hour", "dow"])[col].mean().unstack(fill_value=np.nan)
    pivot.columns = [dow_labels[c] for c in pivot.columns if c < len(dow_labels)]

    colorscale = [
        [0.0,  "#8B1A1A"],
        [0.5,  "#FEEBC8"],
        [0.75, "#C6F6D5"],
        [1.0,  "#22543D"],
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns,
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=colorscale,
        zmin=0, zmax=100,
        colorbar=dict(title="SL%"),
        hovertemplate="Day: %{x}<br>Hour: %{y}<br>SL: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        margin=dict(t=20, b=30, l=60, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def compute_occupancy_risk(df: pd.DataFrame, threshold: float = 0.85) -> dict:
    """
    Identify intervals where occupancy consistently exceeds the threshold.
    High occupancy (>85-90%) is a leading indicator of agent burnout and quality degradation.
    """
    col = get_col(df, ["occupancy", "occupancy_pct"])
    if col is None:
        return None

    series = df[col].dropna()
    # Normalize if stored as 0-100 instead of 0-1
    if series.max() > 1:
        series = series / 100

    high_occ = series[series >= threshold]
    return {
        "threshold": threshold,
        "pct_above_threshold": round(len(high_occ) / len(series) * 100, 2),
        "mean_occupancy": round(series.mean() * 100, 2),
        "max_occupancy":  round(series.max()  * 100, 2),
        "intervals_at_risk": len(high_occ),
        "risk_level": (
            "High" if len(high_occ) / len(series) > 0.20
            else "Medium" if len(high_occ) / len(series) > 0.05
            else "Low"
        ),
    }


def compute_shrinkage_backfill(df: pd.DataFrame) -> dict:
    """
    Back-calculate effective shrinkage if headcount and volume + AHT are available.
    Shrinkage = 1 - (required_hours / scheduled_hours)
    """
    hc_col  = get_col(df, ["headcount", "agents", "ftes"])
    aht_col = get_col(df, ["aht", "handle_time", "avg_handle_time"])

    if hc_col is None or aht_col is None or "volume" not in df.columns:
        return None

    plot_df = df[[hc_col, aht_col, "volume"]].dropna()
    if plot_df.empty:
        return None

    # Required productive hours = volume * AHT / 3600
    plot_df["required_hrs"] = plot_df["volume"] * plot_df[aht_col] / 3600
    # Assume interval duration from granularity
    interval_hrs = 0.5  # assume 30-min default
    plot_df["scheduled_hrs"] = plot_df[hc_col] * interval_hrs
    plot_df["shrinkage"] = (
        1 - (plot_df["required_hrs"] / plot_df["scheduled_hrs"])
    ).clip(0, 1)

    return {
        "mean_shrinkage": round(plot_df["shrinkage"].mean() * 100, 2),
        "median_shrinkage": round(plot_df["shrinkage"].median() * 100, 2),
        "p90_shrinkage": round(plot_df["shrinkage"].quantile(0.90) * 100, 2),
        "series": plot_df["shrinkage"],
    }
