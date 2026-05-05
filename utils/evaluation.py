"""
utils/evaluation.py
Model evaluation: metrics, leaderboard, actual vs. predicted plots,
residual diagnostics, and interval-level error heatmaps.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from config import CHART_COLORS, PLOTLY_TEMPLATE

DOW_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Core Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Full suite of WFM-relevant forecast error metrics.
    """
    actual    = actual.dropna()
    predicted = predicted.reindex(actual.index).dropna()
    actual    = actual.reindex(predicted.index)

    if len(actual) == 0:
        return {}

    errors     = actual - predicted
    abs_errors = errors.abs()
    n          = len(actual)

    mae   = abs_errors.mean()
    rmse  = np.sqrt((errors ** 2).mean())
    mape  = (abs_errors / actual.replace(0, np.nan)).mean() * 100
    smape = (2 * abs_errors / (actual.abs() + predicted.abs()).replace(0, np.nan)).mean() * 100
    wape  = abs_errors.sum() / actual.abs().sum() * 100
    bias  = errors.mean()
    r2    = 1 - (errors ** 2).sum() / ((actual - actual.mean()) ** 2).sum()

    return {
        "MAE":   round(mae,   3),
        "RMSE":  round(rmse,  3),
        "MAPE":  round(mape,  3),
        "sMAPE": round(smape, 3),
        "WAPE":  round(wape,  3),
        "Bias":  round(bias,  3),
        "R2":    round(r2,    4),
        "N":     n,
    }


def build_leaderboard(results: list, sort_by: str = "MAPE") -> pd.DataFrame:
    """
    Build a ranked leaderboard from a list of model result dicts.
    Each result dict must have keys: model_name, predictions, actual.
    """
    rows = []
    for res in results:
        if not res.get("success"):
            rows.append({
                "Model":  res.get("model_name", "Unknown"),
                "Status": f"Failed — {res.get('error', 'unknown error')}",
                "MAE": None, "RMSE": None, "MAPE": None,
                "sMAPE": None, "WAPE": None, "Bias": None, "R2": None,
            })
            continue

        metrics = compute_metrics(res["actual"], res["predictions"])
        rows.append({
            "Model":  res["model_name"],
            "Status": "OK",
            **metrics,
        })

    df = pd.DataFrame(rows)
    ok = df[df["Status"] == "OK"].copy()
    failed = df[df["Status"] != "OK"].copy()

    if sort_by in ok.columns and not ok.empty:
        ok = ok.sort_values(sort_by)

    return pd.concat([ok, failed], ignore_index=True)


# ── Actual vs. Predicted ──────────────────────────────────────────────────────

def plot_actual_vs_predicted(actual: pd.Series, model_results: list,
                             title: str = "Actual vs. Predicted") -> go.Figure:
    """
    Overlay actual series with predictions from multiple models.
    """
    colors = [
        CHART_COLORS["secondary"], CHART_COLORS["accent"],
        CHART_COLORS["success"],   CHART_COLORS["warning"],
        CHART_COLORS["danger"],    "#9B59B6", "#1ABC9C",
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual.index, y=actual,
        mode="lines",
        line=dict(color=CHART_COLORS["primary"], width=2),
        name="Actual",
    ))

    for i, res in enumerate(model_results):
        if not res.get("success"):
            continue
        pred = res["predictions"].reindex(actual.index)
        fig.add_trace(go.Scatter(
            x=pred.index, y=pred,
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=1.5, dash="dash"),
            name=res["model_name"],
            opacity=0.85,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=380,
        title=title,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def plot_single_model(actual: pd.Series, predicted: pd.Series,
                      model_name: str, show_train: pd.Series = None) -> go.Figure:
    """
    Detailed plot for a single model including optional training fit.
    """
    fig = go.Figure()

    if show_train is not None:
        fig.add_trace(go.Scatter(
            x=show_train.index, y=show_train,
            mode="lines",
            line=dict(color=CHART_COLORS["neutral"], width=1, dash="dot"),
            name="Training fit",
            opacity=0.5,
        ))

    fig.add_trace(go.Scatter(
        x=actual.index, y=actual,
        mode="lines",
        line=dict(color=CHART_COLORS["primary"], width=2),
        name="Actual",
    ))

    fig.add_trace(go.Scatter(
        x=predicted.index, y=predicted,
        mode="lines",
        line=dict(color=CHART_COLORS["accent"], width=2, dash="dash"),
        name=f"{model_name} — Predicted",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


# ── Residual Diagnostics ──────────────────────────────────────────────────────

def plot_residuals(actual: pd.Series, predicted: pd.Series, model_name: str) -> go.Figure:
    """
    4-panel residual diagnostic: residuals over time, histogram,
    Q-Q plot, and ACF of residuals.
    """
    from statsmodels.tsa.stattools import acf

    residuals = (actual - predicted.reindex(actual.index)).dropna()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residuals over Time",
            "Residual Distribution",
            "Q-Q Plot",
            "ACF of Residuals",
        ),
    )

    # Residuals over time
    fig.add_trace(go.Scatter(
        x=residuals.index, y=residuals,
        mode="lines",
        line=dict(color=CHART_COLORS["secondary"], width=1),
        name="Residuals",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=CHART_COLORS["muted"], line_width=1, row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=40,
        marker_color=CHART_COLORS["secondary"],
        opacity=0.75,
        name="Distribution",
    ), row=1, col=2)

    # Q-Q plot
    try:
        qq_residuals = residuals.dropna()
        if len(qq_residuals) >= 4:
            (osm, osr), (slope, intercept, _) = stats.probplot(qq_residuals, dist="norm")
            osm = list(osm)
            osr = list(osr)
            if len(osm) > 0:
                fig.add_trace(go.Scatter(
                    x=osm, y=osr,
                    mode="markers",
                    marker=dict(color=CHART_COLORS["secondary"], size=4),
                    name="Q-Q",
                ), row=2, col=1)
                line_x = [min(osm), max(osm)]
                line_y = [slope * x + intercept for x in line_x]
                fig.add_trace(go.Scatter(
                    x=line_x, y=line_y,
                    mode="lines",
                    line=dict(color=CHART_COLORS["danger"], dash="dash", width=1.5),
                    name="Normal line",
                    showlegend=False,
                ), row=2, col=1)
    except Exception:
        pass

    # ACF of residuals
    try:
        acf_residuals = residuals.dropna()
        n_lags = min(40, len(acf_residuals) // 2 - 1)
        if n_lags >= 1:
            acf_vals = acf(acf_residuals, nlags=n_lags)
            conf     = 1.96 / np.sqrt(len(acf_residuals))
            lags     = np.arange(len(acf_vals))

            for i, v in enumerate(acf_vals):
                fig.add_trace(go.Scatter(
                    x=[lags[i], lags[i]], y=[0, v],
                    mode="lines",
                    line=dict(color=CHART_COLORS["danger"] if abs(v) > conf and i > 0
                              else CHART_COLORS["secondary"], width=1.5),
                    showlegend=False,
                ), row=2, col=2)
            fig.add_trace(go.Scatter(
                x=lags, y=acf_vals,
                mode="markers",
                marker=dict(size=5, color=CHART_COLORS["secondary"]),
                name="ACF",
                showlegend=False,
            ), row=2, col=2)
            for sign in [1, -1]:
                fig.add_trace(go.Scatter(
                    x=lags, y=[sign * conf] * len(lags),
                    mode="lines",
                    line=dict(color=CHART_COLORS["warning"], width=1, dash="dash"),
                    showlegend=False,
                ), row=2, col=2)
            fig.add_hline(y=0, line_color=CHART_COLORS["text"], line_width=0.8, row=2, col=2)
    except Exception:
        pass

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=520,
        showlegend=False,
        title=f"Residual Diagnostics — {model_name}",
        margin=dict(t=60, b=20, l=40, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


def run_ljungbox(residuals: pd.Series, lags: int = 10) -> dict:
    """
    Ljung-Box test on residuals.
    H0: residuals are white noise (no autocorrelation).
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        result = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
        p_value = float(result["lb_pvalue"].iloc[0])
        return {
            "success":       True,
            "statistic":     round(float(result["lb_stat"].iloc[0]), 4),
            "p_value":       round(p_value, 4),
            "is_white_noise": p_value > 0.05,
            "interpretation": (
                "Residuals behave as white noise. The model has captured the series structure well."
                if p_value > 0.05 else
                "Residuals show significant autocorrelation. The model has not fully captured the structure — "
                "consider adding seasonal terms or switching to a more complex model."
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Error Heatmap ─────────────────────────────────────────────────────────────

def plot_error_heatmap(actual: pd.Series, predicted: pd.Series,
                       model_name: str, granularity: str) -> go.Figure:
    """
    Hour x DOW heatmap of absolute percentage errors.
    Shows where a model consistently under- or over-forecasts.
    Only for sub-daily granularity.
    """
    if granularity not in ("15T", "30T", "1H"):
        return None

    pred_aligned = predicted.reindex(actual.index).dropna()
    act_aligned  = actual.reindex(pred_aligned.index)
    ape = ((act_aligned - pred_aligned).abs() / act_aligned.replace(0, np.nan) * 100)

    df_err = pd.DataFrame({
        "ape":  ape,
        "hour": ape.index.hour,
        "dow":  ape.index.dayofweek,
    }).dropna()

    pivot = df_err.groupby(["hour", "dow"])["ape"].mean().unstack(fill_value=np.nan)
    pivot.columns = [DOW_LABELS[c] for c in pivot.columns if c < len(DOW_LABELS)]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=[
            [0.0,  "#C6F6D5"],
            [0.3,  "#FEFCBF"],
            [0.7,  "#FEEBC8"],
            [1.0,  "#FED7D7"],
        ],
        colorbar=dict(title="MAPE%"),
        hovertemplate="Day: %{x}<br>Hour: %{y}<br>MAPE: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        title=f"Error Heatmap — {model_name}",
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        margin=dict(t=50, b=30, l=60, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig


# ── Confidence Interval Comparison ───────────────────────────────────────────

def plot_forecast_with_intervals(actual: pd.Series, predicted: pd.Series,
                                  lower: pd.Series, upper: pd.Series,
                                  model_name: str) -> go.Figure:
    """
    Plot forecast with prediction intervals (P10/P90 or confidence bands).
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(predicted.index) + list(predicted.index[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor=f"rgba(46,109,164,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Prediction Interval",
    ))

    fig.add_trace(go.Scatter(
        x=actual.index, y=actual,
        mode="lines",
        line=dict(color=CHART_COLORS["primary"], width=2),
        name="Actual",
    ))

    fig.add_trace(go.Scatter(
        x=predicted.index, y=predicted,
        mode="lines",
        line=dict(color=CHART_COLORS["accent"], width=2, dash="dash"),
        name=f"{model_name} Forecast",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360,
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=30, l=30, r=20),
        font=dict(family="IBM Plex Sans", size=11, color=CHART_COLORS["text"]),
    )
    return fig