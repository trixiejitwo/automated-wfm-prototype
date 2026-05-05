"""
utils/eda/stationarity.py
Stationarity tests (ADF, KPSS) and ACF/PACF diagnostics.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from config import CHART_COLORS, PLOTLY_TEMPLATE, STATIONARITY_ALPHA


def run_adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test for unit root (stationarity).
    H0: series has a unit root (non-stationary)
    Reject H0 (p < alpha) => stationary
    """
    s = series.dropna()
    result = adfuller(s, autolag="AIC")
    stat, p_value, n_lags, n_obs, critical_values = result[0], result[1], result[2], result[3], result[4]
    is_stationary = p_value < STATIONARITY_ALPHA

    return {
        "test": "Augmented Dickey-Fuller",
        "statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "n_lags_used": n_lags,
        "n_observations": n_obs,
        "critical_1pct": round(critical_values["1%"], 4),
        "critical_5pct": round(critical_values["5%"], 4),
        "critical_10pct": round(critical_values["10%"], 4),
        "is_stationary": is_stationary,
        "conclusion": (
            f"The series appears STATIONARY (p={round(p_value,4)} < {STATIONARITY_ALPHA}). "
            "No differencing required."
            if is_stationary else
            f"The series appears NON-STATIONARY (p={round(p_value,4)} >= {STATIONARITY_ALPHA}). "
            "Consider first differencing or log transformation before modeling."
        ),
    }


def run_kpss_test(series: pd.Series) -> dict:
    """
    KPSS test for level stationarity.
    H0: series is stationary
    Reject H0 (p < alpha) => non-stationary

    KPSS and ADF complement each other:
    Both say stationary  => stationary
    ADF stationary, KPSS non-stationary => trend-stationary
    ADF non-stationary, KPSS stationary => difference-stationary
    Both say non-stationary => requires more treatment
    """
    s = series.dropna()
    try:
        stat, p_value, n_lags, critical_values = kpss(s, regression="c", nlags="auto")
        is_stationary = p_value > STATIONARITY_ALPHA

        return {
            "test": "KPSS",
            "statistic": round(stat, 4),
            "p_value": round(p_value, 4),
            "n_lags_used": n_lags,
            "critical_1pct": round(critical_values["1%"], 4),
            "critical_5pct": round(critical_values["5%"], 4),
            "critical_10pct": round(critical_values["10%"], 4),
            "is_stationary": is_stationary,
            "conclusion": (
                f"KPSS: Series is STATIONARY around a level (p={round(p_value,4)} > {STATIONARITY_ALPHA})."
                if is_stationary else
                f"KPSS: Series is NON-STATIONARY (p={round(p_value,4)} <= {STATIONARITY_ALPHA})."
            ),
        }
    except Exception as e:
        return {"test": "KPSS", "error": str(e), "is_stationary": None, "conclusion": f"KPSS test failed: {e}"}


def combined_stationarity_interpretation(adf: dict, kpss_res: dict) -> str:
    """
    Synthesize ADF and KPSS results into a plain-English recommendation.
    """
    adf_stat  = adf.get("is_stationary")
    kpss_stat = kpss_res.get("is_stationary")

    if adf_stat is True and kpss_stat is True:
        return ("Both tests agree: the series is stationary. "
                "ARIMA with d=0 is appropriate. No transformation needed.")
    elif adf_stat is True and kpss_stat is False:
        return ("ADF rejects the unit root, but KPSS detects non-stationarity. "
                "The series may be trend-stationary. "
                "Consider detrending or including a trend component in the model.")
    elif adf_stat is False and kpss_stat is True:
        return ("ADF cannot reject the unit root, but KPSS shows level stationarity. "
                "The series is likely difference-stationary. "
                "Apply first differencing (d=1) before modeling with ARIMA.")
    else:
        return ("Both tests indicate non-stationarity. "
                "Apply first differencing and/or log transformation. "
                "Consider seasonal differencing as well if strong seasonality is present.")


def compute_acf_pacf(series: pd.Series, n_lags: int = 48) -> dict:
    """
    Compute ACF and PACF values with confidence intervals.
    """
    s = series.dropna()
    n_lags = min(n_lags, len(s) // 2 - 1)

    acf_values, acf_ci  = acf(s,  nlags=n_lags, alpha=0.05)
    pacf_values, pacf_ci = pacf(s, nlags=n_lags, alpha=0.05, method="ols")

    ci_upper_acf  = acf_ci[:, 1]  - acf_values
    ci_lower_acf  = acf_values - acf_ci[:, 0]
    ci_upper_pacf = pacf_ci[:, 1] - pacf_values
    ci_lower_pacf = pacf_values - pacf_ci[:, 0]

    return {
        "lags": np.arange(len(acf_values)),
        "acf":  acf_values,
        "pacf": pacf_values,
        "acf_ci_upper":  ci_upper_acf,
        "acf_ci_lower":  ci_lower_acf,
        "pacf_ci_upper": ci_upper_pacf,
        "pacf_ci_lower": ci_lower_pacf,
        "conf_bound": 1.96 / np.sqrt(len(s)),
    }


def plot_acf_pacf(acf_pacf: dict) -> go.Figure:
    """
    Side-by-side ACF and PACF stem plots with confidence bands.
    """
    lags = acf_pacf["lags"]
    conf = acf_pacf["conf_bound"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Autocorrelation Function (ACF)",
                                        "Partial Autocorrelation Function (PACF)"))

    for col, key, title in [(1, "acf", "ACF"), (2, "pacf", "PACF")]:
        values = acf_pacf[key]

        # Stem lines
        for i, v in enumerate(values):
            fig.add_trace(go.Scatter(
                x=[lags[i], lags[i]], y=[0, v],
                mode="lines",
                line=dict(
                    color=CHART_COLORS["danger"] if abs(v) > conf else CHART_COLORS["secondary"],
                    width=1.5,
                ),
                showlegend=False,
            ), row=1, col=col)

        # Markers
        colors_stem = [
            CHART_COLORS["danger"] if abs(v) > conf else CHART_COLORS["secondary"]
            for v in values
        ]
        fig.add_trace(go.Scatter(
            x=lags, y=values,
            mode="markers",
            marker=dict(color=colors_stem, size=5),
            name=title,
            showlegend=False,
        ), row=1, col=col)

        # Confidence bands
        for sign in [1, -1]:
            fig.add_trace(go.Scatter(
                x=lags, y=[sign * conf] * len(lags),
                mode="lines",
                line=dict(color=CHART_COLORS["warning"], width=1, dash="dash"),
                showlegend=False,
            ), row=1, col=col)

        # Zero line
        fig.add_hline(y=0, line_color=CHART_COLORS["text"], line_width=0.8, row=1, col=col)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(t=40, b=20, l=40, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="Correlation")
    return fig


def recommend_arima_order(acf_pacf: dict) -> str:
    """
    Heuristic ARIMA order recommendation from ACF/PACF patterns.
    """
    conf = acf_pacf["conf_bound"]
    acf_vals  = acf_pacf["acf"][1:]   # Skip lag 0
    pacf_vals = acf_pacf["pacf"][1:]

    sig_acf  = sum(1 for v in acf_vals  if abs(v) > conf)
    sig_pacf = sum(1 for v in pacf_vals if abs(v) > conf)

    if sig_acf == 0 and sig_pacf == 0:
        return "No significant autocorrelation detected. The series may already be white noise. ARIMA(0,d,0) or no model needed."
    elif sig_pacf <= 3 and sig_acf > sig_pacf:
        return (f"PACF cuts off at lag ~{sig_pacf}, ACF tails off — suggests AR({sig_pacf}) component. "
                f"Try ARIMA(p={sig_pacf}, d=0 or 1, q=0).")
    elif sig_acf <= 3 and sig_pacf > sig_acf:
        return (f"ACF cuts off at lag ~{sig_acf}, PACF tails off — suggests MA({sig_acf}) component. "
                f"Try ARIMA(p=0, d=0 or 1, q={sig_acf}).")
    else:
        return (f"Both ACF ({sig_acf} significant lags) and PACF ({sig_pacf} significant lags) tail off — "
                "suggests ARMA model. Use auto_arima to determine optimal p and q.")
