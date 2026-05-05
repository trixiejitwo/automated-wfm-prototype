"""
utils/eda/descriptive.py
Descriptive statistics and distributional analysis for WFM volume data.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from config import CHART_COLORS, PLOTLY_TEMPLATE


def compute_descriptive_stats(series: pd.Series) -> pd.DataFrame:
    """
    Compute a comprehensive descriptive statistics table including
    WFM-relevant percentiles and distribution shape metrics.
    """
    s = series.dropna()
    records = [
        ("Count",                   f"{len(s):,}"),
        ("Mean",                    f"{s.mean():,.2f}"),
        ("Median (P50)",            f"{s.median():,.2f}"),
        ("Std Deviation",           f"{s.std():,.2f}"),
        ("Coefficient of Variation",f"{(s.std() / s.mean() * 100):.1f}%"),
        ("Min",                     f"{s.min():,.2f}"),
        ("P10",                     f"{s.quantile(0.10):,.2f}"),
        ("P25",                     f"{s.quantile(0.25):,.2f}"),
        ("P75",                     f"{s.quantile(0.75):,.2f}"),
        ("P90",                     f"{s.quantile(0.90):,.2f}"),
        ("P95",                     f"{s.quantile(0.95):,.2f}"),
        ("P99",                     f"{s.quantile(0.99):,.2f}"),
        ("Max",                     f"{s.max():,.2f}"),
        ("Skewness",                f"{s.skew():.4f}"),
        ("Kurtosis (excess)",       f"{s.kurtosis():.4f}"),
        ("IQR",                     f"{(s.quantile(0.75) - s.quantile(0.25)):,.2f}"),
        ("Zero-volume rows",        f"{(s == 0).sum():,} ({(s == 0).mean()*100:.1f}%)"),
        ("Negative rows",           f"{(s < 0).sum():,}"),
    ]
    return pd.DataFrame(records, columns=["Metric", "Value"])


def plot_distribution(series: pd.Series) -> go.Figure:
    """
    Side-by-side histogram + box plot of the volume distribution.
    Marks mean, median, P90, and P95.
    """
    s = series.dropna()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Volume Distribution", "Box Plot"),
        column_widths=[0.65, 0.35],
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=s,
            nbinsx=50,
            marker_color=CHART_COLORS["secondary"],
            opacity=0.8,
            name="Frequency",
        ),
        row=1, col=1,
    )

    # KDE overlay
    kde_x = np.linspace(s.min(), s.max(), 300)
    kde = stats.gaussian_kde(s)
    kde_y = kde(kde_x) * len(s) * (s.max() - s.min()) / 50

    fig.add_trace(
        go.Scatter(
            x=kde_x, y=kde_y,
            mode="lines",
            line=dict(color=CHART_COLORS["accent"], width=2),
            name="KDE",
        ),
        row=1, col=1,
    )

    # Vertical lines for key percentiles
    for val, label, color in [
        (s.mean(),           "Mean",  CHART_COLORS["primary"]),
        (s.median(),         "P50",   CHART_COLORS["success"]),
        (s.quantile(0.90),   "P90",   CHART_COLORS["warning"]),
        (s.quantile(0.95),   "P95",   CHART_COLORS["danger"]),
    ]:
        fig.add_vline(
            x=val, line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=label,
            annotation_font_size=10,
            row=1, col=1,
        )

    # Box plot
    fig.add_trace(
        go.Box(
            y=s,
            marker_color=CHART_COLORS["primary"],
            boxmean="sd",
            name="Volume",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=380,
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def compute_ranked_intervals(df: pd.DataFrame, granularity: str) -> dict:
    """
    Return the top 5 busiest and quietest periods (intervals, days, etc.)
    based on mean volume.
    """
    results = {}

    if granularity in ("15T", "30T", "1H"):
        # By hour of day
        hourly = df["volume"].groupby(df.index.hour).mean()
        results["busiest_hours"] = hourly.nlargest(5).reset_index()
        results["busiest_hours"].columns = ["Hour", "Avg Volume"]
        results["quietest_hours"] = hourly.nsmallest(5).reset_index()
        results["quietest_hours"].columns = ["Hour", "Avg Volume"]

    # By day of week
    dow_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}
    dow = df["volume"].groupby(df.index.dayofweek).mean().rename(dow_map)
    results["busiest_days"] = dow.nlargest(3).reset_index()
    results["busiest_days"].columns = ["Day", "Avg Volume"]
    results["quietest_days"] = dow.nsmallest(3).reset_index()
    results["quietest_days"].columns = ["Day", "Avg Volume"]

    # By month
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = df["volume"].groupby(df.index.month).mean().rename(month_map)
    results["busiest_months"] = monthly.nlargest(3).reset_index()
    results["busiest_months"].columns = ["Month", "Avg Volume"]

    return results


def plot_rolling_stats(series: pd.Series, window: int = 7) -> go.Figure:
    """
    Plot raw series with rolling mean and rolling std band.
    Window unit = number of data points.
    """
    s = series.dropna()
    roll_mean = s.rolling(window).mean()
    roll_std  = s.rolling(window).std()

    fig = go.Figure()

    # Std band
    fig.add_trace(go.Scatter(
        x=list(s.index) + list(s.index[::-1]),
        y=list(roll_mean + 2 * roll_std) + list((roll_mean - 2 * roll_std)[::-1]),
        fill="toself",
        fillcolor=f"rgba(46,109,164,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±2 Std Band",
    ))

    # Raw series
    fig.add_trace(go.Scatter(
        x=s.index, y=s,
        mode="lines",
        line=dict(color=CHART_COLORS["neutral"], width=1, dash="dot"),
        name="Actual",
        opacity=0.6,
    ))

    # Rolling mean
    fig.add_trace(go.Scatter(
        x=roll_mean.index, y=roll_mean,
        mode="lines",
        line=dict(color=CHART_COLORS["primary"], width=2),
        name=f"{window}-period Rolling Mean",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=340,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig
