"""
utils/eda/decomposition.py
Trend and seasonality decomposition using STL and classical methods.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL, seasonal_decompose, MSTL
from scipy.fft import fft, fftfreq
from config import CHART_COLORS, PLOTLY_TEMPLATE


def run_stl_decomposition(series: pd.Series, period: int) -> dict:
    """
    Run STL (Seasonal-Trend decomposition using Loess).
    Returns dict with trend, seasonal, residual components and strength metrics.
    """
    s = series.dropna()
    try:
        stl = STL(s, period=period, robust=True)
        result = stl.fit()
        var_resid    = np.var(result.resid)
        var_detrended = np.var(result.seasonal + result.resid)
        var_deseasoned = np.var(result.trend + result.resid)
        seasonal_strength = max(0, 1 - var_resid / var_detrended) if var_detrended != 0 else 0
        trend_strength    = max(0, 1 - var_resid / var_deseasoned) if var_deseasoned != 0 else 0
        return {
            "success": True,
            "trend":    result.trend,
            "seasonal": result.seasonal,
            "resid":    result.resid,
            "seasonal_strength": round(seasonal_strength, 4),
            "trend_strength":    round(trend_strength, 4),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_classical_decomposition(series: pd.Series, period: int, model: str = "additive") -> dict:
    """
    Run classical additive or multiplicative decomposition.
    """
    s = series.dropna()
    try:
        result = seasonal_decompose(s, model=model, period=period, extrapolate_trend="freq")
        return {
            "success": True,
            "trend":    result.trend,
            "seasonal": result.seasonal,
            "resid":    result.resid,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def detect_dominant_periods_fft(series: pd.Series, top_n: int = 5) -> list:
    """
    Use FFT to detect dominant seasonal periods in the series.
    Returns a list of (period_in_samples, relative_power) tuples, sorted by power.
    """
    s = series.dropna().values
    n = len(s)
    yf = np.abs(fft(s - s.mean()))[:n // 2]
    xf = fftfreq(n)[:n // 2]

    # Exclude DC component (index 0)
    yf[0] = 0

    top_idx = np.argsort(yf)[::-1][:top_n * 3]
    results = []
    seen = set()
    for idx in top_idx:
        if xf[idx] == 0:
            continue
        period = round(1 / xf[idx])
        if period < 2 or period > n // 2:
            continue
        if period in seen:
            continue
        seen.add(period)
        results.append((period, round(yf[idx] / yf[1:].max(), 4)))
        if len(results) >= top_n:
            break

    return sorted(results, key=lambda x: x[1], reverse=True)


def plot_decomposition(components: dict, title: str = "STL Decomposition") -> go.Figure:
    """
    4-panel decomposition plot: Observed, Trend, Seasonal, Residual.
    """
    series_keys = ["observed", "trend", "seasonal", "resid"]
    panels = []

    observed = components.get("observed")
    trend    = components.get("trend")
    seasonal = components.get("seasonal")
    resid    = components.get("resid")

    panel_data = [
        ("Observed",  observed,  CHART_COLORS["primary"]),
        ("Trend",     trend,     CHART_COLORS["accent"]),
        ("Seasonal",  seasonal,  CHART_COLORS["secondary"]),
        ("Residual",  resid,     CHART_COLORS["neutral"]),
    ]
    panel_data = [(lbl, data, color) for lbl, data, color in panel_data if data is not None]

    fig = make_subplots(
        rows=len(panel_data), cols=1,
        subplot_titles=[p[0] for p in panel_data],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    for i, (label, data, color) in enumerate(panel_data, start=1):
        if label == "Residual":
            fig.add_trace(go.Bar(
                x=data.index, y=data,
                marker_color=color,
                opacity=0.7,
                name=label,
            ), row=i, col=1)
            # Zero line
            fig.add_hline(y=0, line_dash="dot", line_color="#CBD5E0", line_width=1, row=i, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=data.index, y=data,
                mode="lines",
                line=dict(color=color, width=1.5),
                name=label,
            ), row=i, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=160 * len(panel_data),
        showlegend=False,
        margin=dict(t=30, b=20, l=50, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
        title=dict(text=title, font=dict(family="Libre Baskerville", size=13)),
    )
    return fig


def seasonality_strength_gauge(strength: float, label: str = "Seasonal Strength") -> go.Figure:
    """
    Gauge chart showing seasonal or trend strength (0–1 scale).
    """
    color = (
        CHART_COLORS["success"] if strength >= 0.7
        else CHART_COLORS["warning"] if strength >= 0.4
        else CHART_COLORS["danger"]
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(strength * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "family": "IBM Plex Mono"}},
        title={"text": label, "font": {"size": 13, "family": "IBM Plex Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": CHART_COLORS["light"],
            "steps": [
                {"range": [0, 40],   "color": "#FED7D7"},
                {"range": [40, 70],  "color": "#FEEBC8"},
                {"range": [70, 100], "color": "#C6F6D5"},
            ],
            "threshold": {
                "line": {"color": CHART_COLORS["primary"], "width": 2},
                "thickness": 0.75,
                "value": 70,
            },
        },
    ))

    fig.update_layout(
        height=220,
        margin=dict(t=20, b=10, l=20, r=20),
        font=dict(family="IBM Plex Sans", color=CHART_COLORS["text"]),
    )
    return fig
