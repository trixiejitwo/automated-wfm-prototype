"""
pages/2_Modeling_and_Evaluation.py
Model selection, training, evaluation leaderboard, and residual diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Modeling and Evaluation — WFM Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.models.statistical import fit_arima, fit_ets, fit_holtwinters, fit_prophet
from utils.models.ml import build_features, split_features, fit_xgboost, fit_lightgbm, fit_random_forest, fit_ridge
from utils.evaluation import (
    compute_metrics, build_leaderboard, plot_actual_vs_predicted,
    plot_single_model, plot_residuals, run_ljungbox, plot_error_heatmap,
)
from config import CHART_COLORS

st.title("Modeling and Evaluation")
st.caption("Just In Time Workforce Solutions Inc. — WFM Analytics Platform")

if not st.session_state.get("data_loaded"):
    st.warning("No dataset loaded. Go to Ingestion and EDA first.")
    st.stop()

df          = st.session_state["raw_df"]
granularity = st.session_state["granularity"]
series      = df["volume"]
holiday_dict = st.session_state.get("holiday_dict", {})
detected_period = st.session_state.get("detected_period", 7)

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
st.subheader("Preprocessing")

c1, c2 = st.columns(2)
with c1:
    impute_method = st.selectbox(
        "Missing value imputation",
        ["None", "Forward fill", "Linear interpolation", "Seasonal mean"],
    )
with c2:
    outlier_treatment = st.selectbox(
        "Outlier treatment",
        ["None", "Cap at P99 / floor at P1", "Replace with seasonal median"],
    )

transform = "None"
adf_stat  = st.session_state.get("stationarity", {}).get("adf", {}).get("is_stationary", True)
kpss_stat = st.session_state.get("stationarity", {}).get("kpss", {}).get("is_stationary", True)
if not (adf_stat and kpss_stat):
    transform = st.selectbox(
        "Stationarity transformation",
        ["None", "First difference", "Log transform", "Log + first difference"],
    )

# Apply preprocessing
proc = series.copy()

if "Forward" in impute_method:
    proc = proc.ffill()
elif "interpolation" in impute_method:
    proc = proc.interpolate(method="linear")
elif "Seasonal" in impute_method:
    proc = proc.fillna(proc.rolling(7, min_periods=1).mean())

outlier_mask = st.session_state.get("outlier_mask", pd.Series(False, index=proc.index))
if "Cap" in outlier_treatment:
    proc = proc.clip(lower=proc.quantile(0.01), upper=proc.quantile(0.99))
elif "Replace" in outlier_treatment and outlier_mask.sum() > 0:
    seasonal_med = proc.rolling(7, center=True, min_periods=1).median()
    proc[outlier_mask] = seasonal_med[outlier_mask]

if "Log" in transform:
    proc = np.log1p(proc.clip(lower=0))
if "difference" in transform:
    proc = proc.diff().dropna()

st.divider()

# ── TRAIN / VAL / TEST SPLIT ──────────────────────────────────────────────────
st.subheader("Train / Validation / Test Split")

c1, c2, c3 = st.columns(3)
with c1:
    train_pct = st.slider("Train %", 50, 85, 70)
with c2:
    val_pct = st.slider("Validation %", 5, 25, 15)
with c3:
    test_pct = 100 - train_pct - val_pct
    st.metric("Test %", test_pct)

n         = len(proc)
train_end = int(n * train_pct / 100)
val_end   = int(n * (train_pct + val_pct) / 100)

train_s = proc.iloc[:train_end]
val_s   = proc.iloc[train_end:val_end]
test_s  = proc.iloc[val_end:]

import plotly.graph_objects as go
from config import PLOTLY_TEMPLATE

split_fig = go.Figure()
for s, label, color in [
    (train_s, f"Train ({train_pct}%)",     CHART_COLORS["primary"]),
    (val_s,   f"Validation ({val_pct}%)",  CHART_COLORS["accent"]),
    (test_s,  f"Test ({test_pct}%)",        CHART_COLORS["success"]),
]:
    split_fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=label,
                                   line=dict(color=color, width=1.5)))
split_fig.update_layout(
    template=PLOTLY_TEMPLATE, height=260,
    xaxis_title="Date", yaxis_title="Volume",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=20, b=20, l=30, r=20),
    font=dict(family="IBM Plex Sans", size=11),
)
st.plotly_chart(split_fig, width='stretch')

st.divider()

# ── MODEL SELECTION ───────────────────────────────────────────────────────────
st.subheader("Model Selection")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Statistical**")
    use_arima  = st.checkbox("ARIMA / SARIMA",  value=True)
    use_ets    = st.checkbox("ETS",             value=True)
    use_hw     = st.checkbox("Holt-Winters",    value=True)
    use_prophet= st.checkbox("Prophet",         value=True)
with c2:
    st.markdown("**Machine Learning**")
    use_xgb   = st.checkbox("XGBoost",          value=True)
    use_lgbm  = st.checkbox("LightGBM",         value=True)
    use_rf    = st.checkbox("Random Forest",    value=False)
    use_ridge = st.checkbox("Ridge Regression", value=False)
with c3:
    st.markdown("**Deep Learning** *(optional)*")
    st.caption("DL models require additional setup and longer training times.")
    use_lstm = st.checkbox("LSTM", value=False, disabled=True)
    use_gru  = st.checkbox("GRU",  value=False, disabled=True)
    st.caption("LSTM/GRU support coming in next iteration.")

st.divider()

# ── MODEL CONFIGURATION ───────────────────────────────────────────────────────
st.subheader("Model Configuration")

tabs_cfg = st.tabs(["ARIMA", "ETS / Holt-Winters", "Prophet", "ML Models", "Feature Engineering"])

with tabs_cfg[0]:
    c1, c2 = st.columns(2)
    with c1:
        arima_auto = st.checkbox("Use auto_arima", value=True)
    with c2:
        use_seasonal_arima = st.checkbox("Seasonal (SARIMA)", value=True)
    arima_period = st.number_input("ARIMA seasonal period (s)", 2, 1000, int(detected_period), key="arima_s")
    arima_config = {
        "auto": arima_auto, "seasonal": use_seasonal_arima,
        "period": arima_period,
    }

with tabs_cfg[1]:
    c1, c2 = st.columns(2)
    with c1:
        ets_trend    = st.selectbox("ETS trend",    ["add", "mul", "none"])
        ets_seasonal = st.selectbox("ETS seasonal", ["add", "mul", "none"])
    with c2:
        ets_period   = st.number_input("ETS seasonal period", 2, 1000, int(detected_period), key="ets_s")
        hw_damped    = st.checkbox("Holt-Winters damped trend", value=True)
    ets_config = {"trend": ets_trend, "seasonal": ets_seasonal, "period": ets_period}
    hw_config  = {"period": ets_period, "damped": hw_damped}

with tabs_cfg[2]:
    c1, c2 = st.columns(2)
    with c1:
        p_changepoint = st.slider("Changepoint prior scale", 0.001, 0.5, 0.05, key="p_cp")
        p_seasonality = st.slider("Seasonality prior scale", 0.01, 10.0, 1.0,  key="p_sp")
    with c2:
        p_holidays = st.checkbox("Include PH holidays", value=True)
        p_yearly   = st.checkbox("Yearly seasonality",  value=True)
        p_weekly   = st.checkbox("Weekly seasonality",  value=True)
        p_daily    = st.checkbox("Daily seasonality",   value=True)
    prophet_config = {
        "changepoint_prior": p_changepoint, "seasonality_prior": p_seasonality,
        "use_holidays": p_holidays, "yearly_seasonality": p_yearly,
        "weekly_seasonality": p_weekly, "daily_seasonality": p_daily,
    }

with tabs_cfg[3]:
    c1, c2, c3 = st.columns(3)
    with c1:
        n_estimators = st.slider("Estimators",     50,  500, 200)
    with c2:
        max_depth    = st.slider("Max depth",       2,   12,  5)
    with c3:
        learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.05)
    ml_config = {"n_estimators": n_estimators, "max_depth": max_depth, "lr": learning_rate}

with tabs_cfg[4]:
    c1, c2 = st.columns(2)
    with c1:
        lag_features = st.multiselect(
            "Lag features",
            ["lag_1", "lag_7", "lag_14", "lag_28", "lag_48", "lag_336"],
            default=["lag_1", "lag_7"],
        )
        rolling_features = st.multiselect(
            "Rolling statistics",
            ["rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std_7", "rolling_std_28"],
            default=["rolling_mean_7"],
        )
    with c2:
        calendar_features = st.multiselect(
            "Calendar features",
            ["hour", "day_of_week", "day_of_month", "week_of_month", "month", "is_holiday", "is_payday_week"],
            default=["hour", "day_of_week", "month", "is_holiday"],
        )

st.divider()

# ── TRAIN ─────────────────────────────────────────────────────────────────────
if st.button("Train Selected Models", type="primary"):

    selected_models = {
        "arima":        use_arima,
        "ets":          use_ets,
        "holtwinters":  use_hw,
        "prophet":      use_prophet,
        "xgboost":      use_xgb,
        "lightgbm":     use_lgbm,
        "randomforest": use_rf,
        "ridge":        use_ridge,
    }
    active = [k for k, v in selected_models.items() if v]

    results     = []
    progress    = st.progress(0)
    status_text = st.empty()

    # Build ML feature matrix once
    ml_needed = any(k in active for k in ["xgboost", "lightgbm", "randomforest", "ridge"])
    if ml_needed:
        feat_df   = build_features(proc, lag_features, rolling_features, calendar_features, holiday_dict)
        X_tr, y_tr, X_val, y_val, X_te, y_te, feat_cols = split_features(feat_df, train_end, val_end)

    for i, key in enumerate(active):
        status_text.text(f"Training {key} ({i+1}/{len(active)})...")

        if key == "arima":
            r = fit_arima(train_s, test_s, arima_config)
        elif key == "ets":
            r = fit_ets(train_s, test_s, ets_config)
        elif key == "holtwinters":
            r = fit_holtwinters(train_s, test_s, hw_config)
        elif key == "prophet":
            r = fit_prophet(train_s, test_s, prophet_config, holiday_dict)
        elif key == "xgboost":
            r = fit_xgboost(X_tr, y_tr, X_te, y_te, ml_config)
            if r["success"]:
                r["feature_cols"] = feat_cols
                r["residual_std"] = float((y_tr - r["fitted_values"].reindex(y_tr.index)).std())
        elif key == "lightgbm":
            r = fit_lightgbm(X_tr, y_tr, X_te, y_te, ml_config)
            if r["success"]:
                r["feature_cols"] = feat_cols
                r["residual_std"] = float((y_tr - r["fitted_values"].reindex(y_tr.index)).std())
        elif key == "randomforest":
            r = fit_random_forest(X_tr, y_tr, X_te, y_te, ml_config)
            if r["success"]:
                r["feature_cols"] = feat_cols
                r["residual_std"] = float((y_tr - r["fitted_values"].reindex(y_tr.index)).std())
        elif key == "ridge":
            r = fit_ridge(X_tr, y_tr, X_te, y_te, ml_config)
            if r["success"]:
                r["feature_cols"] = feat_cols
                r["residual_std"] = float((y_tr - r["fitted_values"].reindex(y_tr.index)).std())
        else:
            continue

        # Attach actual test series for evaluation
        if r.get("success"):
            r["actual"] = test_s if key not in ("xgboost", "lightgbm", "randomforest", "ridge") else y_te

        results.append(r)
        progress.progress((i + 1) / len(active))

    status_text.text("Training complete.")
    st.session_state["model_results"] = results
    st.session_state["test_series"]   = test_s
    st.session_state["train_series"]  = train_s
    st.session_state["proc_series"]   = proc
    st.session_state["granularity"]   = granularity
    st.rerun()

# ── EVALUATION ────────────────────────────────────────────────────────────────
results = st.session_state.get("model_results")
if not results:
    st.stop()

test_s  = st.session_state.get("test_series",  series.iloc[int(len(series)*0.85):])
train_s = st.session_state.get("train_series", series.iloc[:int(len(series)*0.70)])

st.divider()
st.subheader("Evaluation")

sort_metric = st.selectbox("Sort leaderboard by", ["MAPE", "MAE", "RMSE", "sMAPE", "WAPE"], index=0)

leaderboard = build_leaderboard(results, sort_by=sort_metric)
st.dataframe(leaderboard, width='stretch', hide_index=True)

st.caption(
    "MAPE and WAPE are most interpretable for WFM planning. "
    "Bias indicates systematic over- or under-forecasting. "
    "R2 closer to 1.0 is better."
)

st.markdown("#### Actual vs. Predicted — All Models")
st.plotly_chart(
    plot_actual_vs_predicted(test_s, results, title="Test Period — Actual vs. Predicted"),
    width='stretch',
)

# Per-model deep dive
st.markdown("#### Per-Model Diagnostics")

successful = [r for r in results if r.get("success")]
if not successful:
    st.info("No models trained successfully.")
    st.stop()

model_names  = [r["model_name"] for r in successful]
selected_mdl = st.selectbox("Select model for detailed diagnostics", model_names)
mdl_result   = next(r for r in successful if r["model_name"] == selected_mdl)

actual_eval  = mdl_result.get("actual", test_s)
pred_eval    = mdl_result["predictions"].reindex(actual_eval.index)

st.plotly_chart(
    plot_single_model(
        actual_eval, pred_eval, selected_mdl,
        show_train=mdl_result.get("fitted_values"),
    ),
    width='stretch',
)

metrics = compute_metrics(actual_eval, pred_eval)
if metrics:
    display_metrics = {k: v for k, v in metrics.items() if k != 'N'}
    m_cols = st.columns(max(1, len(display_metrics)))
    for col, (k, v) in zip(m_cols, display_metrics.items()):
        col.metric(k, v)
else:
    st.warning('Could not compute metrics — check that predictions align with the test series.')

st.markdown("#### Residual Diagnostics")
st.plotly_chart(plot_residuals(actual_eval, pred_eval, selected_mdl), width='stretch')

lb = run_ljungbox((actual_eval - pred_eval.reindex(actual_eval.index)).dropna())
if lb.get("success"):
    st.info(f"Ljung-Box test (lag 10): statistic={lb['statistic']}, p={lb['p_value']}. {lb['interpretation']}")

if granularity in ("15T", "30T", "1H"):
    st.markdown("#### Error Heatmap — Where Does the Model Struggle?")
    err_hm = plot_error_heatmap(actual_eval, pred_eval, selected_mdl, granularity)
    if err_hm:
        st.plotly_chart(err_hm, width='stretch')

# Feature importance (ML models)
if "feature_importances" in mdl_result:
    st.markdown("#### Feature Importances")
    fi = mdl_result["feature_importances"].head(15).reset_index()
    fi.columns = ["Feature", "Importance"]
    import plotly.express as px
    fi_fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                    color_discrete_sequence=[CHART_COLORS["secondary"]])
    fi_fig.update_layout(
        template="plotly_white", height=350,
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(family="IBM Plex Sans", size=11),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fi_fig, width='stretch')

# Save best model to session state
ok_results = [r for r in results if r.get("success")]
if ok_results:
    lb_df     = build_leaderboard(ok_results, sort_by="MAPE")
    best_name = lb_df.iloc[0]["Model"] if not lb_df.empty else ok_results[0]["model_name"]
    best_mdl  = next((r for r in ok_results if r["model_name"] == best_name), ok_results[0])
    st.session_state["best_model"]   = best_mdl
    st.session_state["all_models"]   = ok_results
    st.session_state["proc_series"]  = st.session_state.get("proc_series", proc)