"""
utils/eda/holidays.py
Holiday and special event impact analysis.
Includes Philippine public holiday calendar as default.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import CHART_COLORS, PLOTLY_TEMPLATE

try:
    import holidays as holidays_lib
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False


def get_ph_holidays(years: list) -> dict:
    """
    Return a dict of {date: holiday_name} for Philippine public holidays.
    Falls back to a hardcoded list if the 'holidays' library is unavailable.
    """
    if HOLIDAYS_AVAILABLE:
        ph = {}
        for yr in years:
            ph.update(holidays_lib.PH(years=yr))
        return {str(k): v for k, v in ph.items()}

    # Hardcoded fallback for key Philippine holidays
    fallback = {}
    for yr in years:
        fallback.update({
            f"{yr}-01-01": "New Year's Day",
            f"{yr}-04-09": "Araw ng Kagitingan",
            f"{yr}-05-01": "Labor Day",
            f"{yr}-06-12": "Independence Day",
            f"{yr}-08-21": "Ninoy Aquino Day",
            f"{yr}-08-26": "National Heroes Day",
            f"{yr}-11-01": "All Saints Day",
            f"{yr}-11-02": "All Souls Day",
            f"{yr}-11-30": "Bonifacio Day",
            f"{yr}-12-08": "Feast of the Immaculate Conception",
            f"{yr}-12-24": "Christmas Eve",
            f"{yr}-12-25": "Christmas Day",
            f"{yr}-12-30": "Rizal Day",
            f"{yr}-12-31": "New Year's Eve",
        })
    return fallback


def compute_holiday_impact(df: pd.DataFrame, holiday_dict: dict) -> pd.DataFrame:
    """
    For each holiday, compare volume on the holiday vs. rolling 4-week average
    on the same day of week. Returns a DataFrame with impact metrics.
    """
    records = []
    df = df.copy()
    df["date_str"] = df.index.strftime("%Y-%m-%d")

    for date_str, name in holiday_dict.items():
        try:
            hol_date = pd.Timestamp(date_str)
        except Exception:
            continue

        hol_rows = df[df["date_str"] == date_str]["volume"]
        if hol_rows.empty:
            continue

        hol_volume = hol_rows.sum()
        dow = hol_date.dayofweek

        # Baseline: same DOW in 4 weeks before and after (excluding other holidays)
        window_start = hol_date - pd.Timedelta(weeks=8)
        window_end   = hol_date + pd.Timedelta(weeks=8)
        baseline_rows = df[
            (df.index >= window_start) &
            (df.index <= window_end) &
            (df.index.dayofweek == dow) &
            (~df["date_str"].isin(holiday_dict.keys()))
        ]["volume"]

        baseline = baseline_rows.sum() / max(baseline_rows.index.normalize().nunique(), 1)
        hol_daily = hol_volume / max(hol_rows.index.normalize().nunique(), 1)

        if baseline > 0:
            lift = (hol_daily - baseline) / baseline * 100
        else:
            lift = None

        records.append({
            "Date": date_str,
            "Holiday": name,
            "Holiday Volume": round(hol_daily, 1),
            "Baseline Volume": round(baseline, 1),
            "Lift %": round(lift, 1) if lift is not None else "N/A",
            "Impact": (
                "Surge" if lift and lift > 10
                else "Drop" if lift and lift < -10
                else "Neutral"
            ),
        })

    return pd.DataFrame(records).sort_values("Date")


def compute_halo_effect(df: pd.DataFrame, holiday_dict: dict, days_window: int = 3) -> pd.DataFrame:
    """
    Compute volume in the N days before and after each holiday
    versus baseline to identify pre/post holiday halo effects.
    """
    records = []
    df = df.copy()
    df["date_str"] = df.index.strftime("%Y-%m-%d")

    for date_str, name in holiday_dict.items():
        try:
            hol_date = pd.Timestamp(date_str)
        except Exception:
            continue

        for offset in range(-days_window, days_window + 1):
            if offset == 0:
                continue
            target_date = hol_date + pd.Timedelta(days=offset)
            target_str  = target_date.strftime("%Y-%m-%d")
            rows = df[df["date_str"] == target_str]["volume"]
            if rows.empty:
                continue

            dow = target_date.dayofweek
            baseline_rows = df[
                (df.index.dayofweek == dow) &
                (~df["date_str"].isin(holiday_dict.keys())) &
                (df.index >= hol_date - pd.Timedelta(weeks=8)) &
                (df.index <= hol_date + pd.Timedelta(weeks=8))
            ]["volume"]

            baseline = baseline_rows.mean() if not baseline_rows.empty else None
            actual   = rows.mean()
            lift     = (actual - baseline) / baseline * 100 if baseline else None

            records.append({
                "Holiday": name,
                "Offset (days)": offset,
                "Direction": "Before" if offset < 0 else "After",
                "Avg Volume": round(actual, 2),
                "Baseline": round(baseline, 2) if baseline else None,
                "Halo Lift %": round(lift, 1) if lift else None,
            })

    return pd.DataFrame(records)


def plot_holiday_impact_chart(impact_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of holiday lift percentages.
    """
    if impact_df.empty:
        return None

    plot_df = impact_df[impact_df["Lift %"] != "N/A"].copy()
    if plot_df.empty:
        return None

    plot_df["Lift %"] = pd.to_numeric(plot_df["Lift %"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Lift %"]).sort_values("Lift %")

    colors = [
        CHART_COLORS["danger"] if v > 10
        else CHART_COLORS["success"] if v < -10
        else CHART_COLORS["neutral"]
        for v in plot_df["Lift %"]
    ]

    fig = go.Figure(go.Bar(
        x=plot_df["Lift %"],
        y=plot_df["Holiday"] + " (" + plot_df["Date"] + ")",
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in plot_df["Lift %"]],
        textposition="outside",
    ))

    fig.add_vline(x=0, line_color=CHART_COLORS["text"], line_width=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(300, len(plot_df) * 35 + 60),
        xaxis_title="Volume Lift vs. Baseline (%)",
        yaxis_title="",
        showlegend=False,
        margin=dict(t=20, b=40, l=20, r=80),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig
