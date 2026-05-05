"""
utils/eda/patterns.py
Volume pattern analysis: intraday curves, DOW profiles, heatmaps, and index tables.
These are the core WFM planning visuals.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import CHART_COLORS, PLOTLY_TEMPLATE

DOW_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── Intraday ─────────────────────────────────────────────────────────────────

def plot_intraday_profile(df: pd.DataFrame, granularity: str) -> go.Figure:
    """
    Plot the average intraday volume curve with ±1 std deviation band.
    Only meaningful for sub-daily granularity.
    """
    if granularity not in ("15T", "30T", "1H"):
        return None

    # Group by time-of-day
    df = df.copy()
    if granularity == "1H":
        df["interval"] = df.index.hour
        x_label = "Hour of Day"
    else:
        minutes = 15 if granularity == "15T" else 30
        df["interval"] = df.index.hour * 60 + (df.index.minute // minutes) * minutes
        x_label = "Time of Day (minutes from midnight)"

    grouped = df.groupby("interval")["volume"]
    mean_  = grouped.mean()
    std_   = grouped.std()
    upper  = mean_ + std_
    lower  = (mean_ - std_).clip(lower=0)

    fig = go.Figure()

    # Std band
    fig.add_trace(go.Scatter(
        x=list(mean_.index) + list(mean_.index[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor=f"rgba(46,109,164,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 Std (planning risk band)",
    ))

    # Mean curve
    fig.add_trace(go.Scatter(
        x=mean_.index, y=mean_,
        mode="lines+markers",
        line=dict(color=CHART_COLORS["primary"], width=2.5),
        marker=dict(size=4),
        name="Mean Volume",
    ))

    # P90 line — planning ceiling
    p90 = grouped.quantile(0.90)
    fig.add_trace(go.Scatter(
        x=p90.index, y=p90,
        mode="lines",
        line=dict(color=CHART_COLORS["warning"], width=1.5, dash="dash"),
        name="P90 (peak planning ceiling)",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360,
        xaxis_title=x_label,
        yaxis_title="Avg Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def plot_intraday_by_dow(df: pd.DataFrame, granularity: str) -> go.Figure:
    """
    Overlay average intraday curves for each day of week on one chart.
    Reveals which days have different shapes — critical for WFM scheduling.
    """
    if granularity not in ("15T", "30T", "1H"):
        return None

    df = df.copy()
    if granularity == "1H":
        df["interval"] = df.index.hour
    else:
        minutes = 15 if granularity == "15T" else 30
        df["interval"] = df.index.hour * 60 + (df.index.minute // minutes) * minutes

    df["dow"] = df.index.dayofweek
    colors = px.colors.qualitative.Set2

    fig = go.Figure()
    for dow_num, dow_name in enumerate(DOW_LABELS):
        subset = df[df["dow"] == dow_num].groupby("interval")["volume"].mean()
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset.index, y=subset,
            mode="lines",
            line=dict(color=colors[dow_num % len(colors)], width=2),
            name=dow_name,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360,
        xaxis_title="Interval",
        yaxis_title="Avg Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


# ── Day-of-Week ───────────────────────────────────────────────────────────────

def compute_dow_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DOW index: each day's average volume relative to the weekly mean.
    e.g., Monday = 1.18 means Monday is 18% above average.
    """
    weekly_mean = df["volume"].mean()
    dow_means = df.groupby(df.index.dayofweek)["volume"].mean()
    result = pd.DataFrame({
        "Day": DOW_LABELS[:len(dow_means)],
        "Avg Volume": dow_means.values.round(2),
        "DOW Index": (dow_means.values / weekly_mean).round(3),
        "vs. Mean": [(f"+{(v/weekly_mean - 1)*100:.1f}%" if v >= weekly_mean
                      else f"{(v/weekly_mean - 1)*100:.1f}%")
                     for v in dow_means.values],
    })
    return result


def plot_dow_bar(dow_df: pd.DataFrame) -> go.Figure:
    """Bar chart of average volume by day of week with index overlay."""
    colors = [
        CHART_COLORS["danger"] if idx > 1.15
        else CHART_COLORS["warning"] if idx > 1.05
        else CHART_COLORS["primary"] if idx >= 0.95
        else CHART_COLORS["neutral"]
        for idx in dow_df["DOW Index"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dow_df["Day"],
        y=dow_df["Avg Volume"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in dow_df["Avg Volume"]],
        textposition="outside",
        name="Avg Volume",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        xaxis_title="Day of Week",
        yaxis_title="Avg Volume",
        showlegend=False,
        margin=dict(t=20, b=20, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def plot_dow_stability(df: pd.DataFrame) -> go.Figure:
    """
    Box plots per DOW showing spread across all weeks.
    Wide boxes = unstable/risky. Tight boxes = predictable.
    """
    dow_groups = [
        df[df.index.dayofweek == d]["volume"].values
        for d in range(7)
    ]

    fig = go.Figure()
    colors = px.colors.qualitative.Pastel
    for i, (label, data) in enumerate(zip(DOW_LABELS, dow_groups)):
        if len(data) == 0:
            continue
        fig.add_trace(go.Box(
            y=data,
            name=label,
            marker_color=colors[i % len(colors)],
            boxmean=True,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=340,
        yaxis_title="Volume",
        showlegend=False,
        margin=dict(t=20, b=20, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


# ── Heatmaps ─────────────────────────────────────────────────────────────────

def plot_hour_dow_heatmap(df: pd.DataFrame, granularity: str) -> go.Figure:
    """
    The classic WFM heatmap: Hour x Day-of-Week, colored by mean volume.
    """
    if granularity not in ("15T", "30T", "1H"):
        return None

    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek

    pivot = df.groupby(["hour", "dow"])["volume"].mean().unstack(fill_value=0)
    pivot.columns = [DOW_LABELS[c] for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=[
            [0.0,  "#EDF2F7"],
            [0.25, "#BEE3F8"],
            [0.5,  "#2E6DA4"],
            [0.75, "#1B3A5C"],
            [1.0,  "#C8974E"],
        ],
        colorbar=dict(title="Avg Volume"),
        hovertemplate="Day: %{x}<br>Hour: %{y}<br>Avg Volume: %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        margin=dict(t=20, b=30, l=60, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def plot_month_dom_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Month x Day-of-Month heatmap.
    Reveals month-start/end spikes common in billing and payments contact centers.
    """
    df = df.copy()
    df["month"] = df.index.month
    df["dom"]   = df.index.day

    pivot = df.groupby(["month", "dom"])["volume"].mean().unstack(fill_value=np.nan)
    pivot.index = [MONTH_LABELS[m - 1] for m in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(d) for d in pivot.columns],
        y=pivot.index,
        colorscale="Blues",
        colorbar=dict(title="Avg Volume"),
        hovertemplate="Month: %{y}<br>Day: %{x}<br>Avg Volume: %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        xaxis_title="Day of Month",
        yaxis_title="Month",
        margin=dict(t=20, b=30, l=60, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def plot_yoy_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Year-over-year weekly volume comparison.
    Lines per year, week number on x-axis.
    """
    df = df.copy()
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week.astype(int)

    years = sorted(df["year"].unique())
    if len(years) < 2:
        return None

    weekly = df.groupby(["year", "week"])["volume"].sum().reset_index()
    colors = [CHART_COLORS["primary"], CHART_COLORS["accent"],
              CHART_COLORS["success"], CHART_COLORS["secondary"]]

    fig = go.Figure()
    for i, yr in enumerate(years):
        subset = weekly[weekly["year"] == yr]
        fig.add_trace(go.Scatter(
            x=subset["week"],
            y=subset["volume"],
            mode="lines",
            name=str(yr),
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=340,
        xaxis_title="ISO Week Number",
        yaxis_title="Total Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def compute_week_of_month_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average volume by week-of-month (1–5).
    Identifies first-week or last-week surges.
    """
    df = df.copy()
    df["wom"] = (df.index.day - 1) // 7 + 1
    result = df.groupby("wom")["volume"].mean().reset_index()
    result.columns = ["Week of Month", "Avg Volume"]
    result["Week of Month"] = result["Week of Month"].apply(lambda w: f"Week {w}")
    return result
