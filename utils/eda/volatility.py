"""
utils/eda/volatility.py
Volatility, risk, and anomaly analysis from a WFM planning perspective.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from config import CHART_COLORS, PLOTLY_TEMPLATE, OUTLIER_Z_THRESHOLD


# ── Rolling Volatility ────────────────────────────────────────────────────────

def compute_rolling_cv(series: pd.Series, window: int = 28) -> pd.Series:
    """
    Rolling Coefficient of Variation (std / mean * 100).
    A rising CV signals increasing forecast and staffing risk.
    """
    return (series.rolling(window).std() / series.rolling(window).mean() * 100)


def plot_rolling_cv(series: pd.Series, windows: list = [7, 14, 28]) -> go.Figure:
    """
    Plot rolling CV for multiple window sizes.
    """
    fig = go.Figure()
    colors = [CHART_COLORS["secondary"], CHART_COLORS["accent"], CHART_COLORS["danger"]]
    for i, w in enumerate(windows):
        cv = compute_rolling_cv(series, w)
        fig.add_trace(go.Scatter(
            x=cv.index, y=cv,
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=1.8),
            name=f"{w}-period Rolling CV (%)",
        ))

    # Threshold line at 30% — elevated volatility zone
    fig.add_hline(
        y=30, line_dash="dash", line_color=CHART_COLORS["warning"], line_width=1,
        annotation_text="30% CV threshold",
        annotation_font_size=10,
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        xaxis_title="Date",
        yaxis_title="Rolling CV (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


# ── Outlier Detection ─────────────────────────────────────────────────────────

def detect_outliers(series: pd.Series, method: str = "zscore", threshold: float = OUTLIER_Z_THRESHOLD) -> pd.Series:
    """
    Flag outliers. Returns a boolean Series (True = outlier).
    method: 'zscore' | 'iqr'
    """
    if method == "zscore":
        z = np.abs(stats.zscore(series.dropna()))
        outlier_mask = pd.Series(False, index=series.index)
        outlier_mask[series.dropna().index] = z > threshold
        return outlier_mask
    elif method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    return pd.Series(False, index=series.index)


def plot_outlier_series(series: pd.Series, outlier_mask: pd.Series) -> go.Figure:
    """
    Time series with outlier points highlighted in red.
    """
    fig = go.Figure()

    normal = series[~outlier_mask]
    anomalous = series[outlier_mask]

    fig.add_trace(go.Scatter(
        x=series.index, y=series,
        mode="lines",
        line=dict(color=CHART_COLORS["secondary"], width=1.5),
        name="Volume",
        opacity=0.8,
    ))

    fig.add_trace(go.Scatter(
        x=anomalous.index, y=anomalous,
        mode="markers",
        marker=dict(
            color=CHART_COLORS["danger"],
            size=8,
            symbol="circle-open",
            line=dict(width=2, color=CHART_COLORS["danger"]),
        ),
        name=f"Outliers ({len(anomalous)})",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


# ── Peak / Worst-Case Analysis ────────────────────────────────────────────────

def worst_case_profile(series: pd.Series, top_pct: float = 0.05) -> dict:
    """
    Characterize the worst-case volume days (top X% by volume).
    Returns threshold, count, and dates of worst-case periods.
    """
    threshold = series.quantile(1 - top_pct)
    worst = series[series >= threshold]
    freq = len(worst) / len(series) * 100
    return {
        "threshold": round(threshold, 2),
        "count": len(worst),
        "frequency_pct": round(freq, 2),
        "dates": worst.sort_values(ascending=False).head(10),
    }


def plot_p90_vs_mean_gap(df: pd.DataFrame, granularity: str) -> go.Figure:
    """
    Per-interval comparison of mean vs P90 volume.
    The gap is the buffer needed above forecast to hit staffing targets.
    Only for sub-daily granularity.
    """
    if granularity not in ("15T", "30T", "1H"):
        return None

    df = df.copy()
    if granularity == "1H":
        df["interval"] = df.index.hour
    else:
        minutes = 15 if granularity == "15T" else 30
        df["interval"] = df.index.hour * 60 + (df.index.minute // minutes) * minutes

    grouped = df.groupby("interval")["volume"]
    mean_ = grouped.mean()
    p90_  = grouped.quantile(0.90)
    gap_  = p90_ - mean_
    gap_pct = (gap_ / mean_ * 100).fillna(0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=mean_.index, y=mean_,
        name="Mean Volume",
        marker_color=CHART_COLORS["secondary"],
        opacity=0.8,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=p90_.index, y=p90_,
        mode="lines+markers",
        line=dict(color=CHART_COLORS["warning"], width=2, dash="dash"),
        marker=dict(size=4),
        name="P90 Volume",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=gap_pct.index, y=gap_pct,
        mode="lines",
        line=dict(color=CHART_COLORS["danger"], width=1.5, dash="dot"),
        name="Gap % (right axis)",
    ), secondary_y=True)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=340,
        xaxis_title="Interval",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=50),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Gap %", secondary_y=True)
    return fig


def poisson_fit_test(series: pd.Series) -> dict:
    """
    Test whether the volume series follows a Poisson distribution.
    (Erlang C assumes Poisson arrivals. If violated, queue models may underestimate staffing.)
    For Poisson: mean ≈ variance. The index of dispersion = variance/mean.
    """
    s = series.dropna()
    mean_ = s.mean()
    var_  = s.var()
    iod   = var_ / mean_ if mean_ != 0 else None

    # Chi-squared goodness of fit on rounded integers
    rounded = s.round().astype(int).clip(lower=0)
    observed_counts = rounded.value_counts().sort_index()
    max_val = min(observed_counts.index.max(), 200)
    bins = np.arange(0, max_val + 2)
    expected_freq = stats.poisson.pmf(bins[:-1], mean_) * len(s)
    # Group tail
    obs_arr = np.array([observed_counts.get(k, 0) for k in bins[:-1]])

    # Only use bins where expected >= 5
    mask = expected_freq >= 5
    if mask.sum() < 2:
        return {"success": False, "reason": "Insufficient data for chi-squared test."}

    chi2_stat, p_value = stats.chisquare(obs_arr[mask], f_exp=expected_freq[mask])

    return {
        "success": True,
        "mean": round(mean_, 2),
        "variance": round(var_, 2),
        "index_of_dispersion": round(iod, 4) if iod else None,
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "is_poisson": p_value > 0.05,
        "interpretation": (
            "Arrival pattern is consistent with Poisson distribution. Erlang C is valid."
            if p_value > 0.05
            else f"Arrival pattern deviates from Poisson (index of dispersion = {round(iod, 2)}). "
                 "Consider over-dispersed models (e.g., Negative Binomial) for queue calculations."
        ),
    }
