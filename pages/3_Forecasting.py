"""
pages/3_Forecasting.py
Forecast generation, prediction intervals, Erlang C staffing conversion,
what-if scenario tool, and export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Forecasting — WFM Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.forecast_output import (
    generate_forecast, build_ensemble, build_staffing_plan,
    plot_forecast, plot_staffing_plan,
    export_forecast_csv, export_staffing_excel,
)
from config import CHART_COLORS, FORECAST_HORIZON_DEFAULTS, ERLANG_DEFAULTS

st.title("Forecasting")
st.caption("Just In Time Workforce Solutions Inc. — WFM Analytics Platform")

if not st.session_state.get("data_loaded"):
    st.warning("No dataset loaded. Go to Ingestion and EDA first.")
    st.stop()

if not st.session_state.get("model_results"):
    st.warning("No trained models found. Go to Modeling and Evaluation and train at least one model.")
    st.stop()

granularity  = st.session_state["granularity"]
series       = st.session_state["raw_df"]["volume"]
proc_series  = st.session_state.get("proc_series", series)
all_models   = st.session_state.get("all_models", [])
best_model   = st.session_state.get("best_model")

# ── MODEL SELECTION ───────────────────────────────────────────────────────────
st.subheader("Forecast Configuration")

model_options = ["Best model (auto)", "Ensemble — simple average", "Ensemble — weighted"] + \
                [r["model_name"] for r in all_models]

c1, c2 = st.columns(2)
with c1:
    model_choice = st.selectbox("Model", model_options)
with c2:
    default_horizon = FORECAST_HORIZON_DEFAULTS.get(granularity, 30)
    horizon = st.number_input("Forecast horizon (intervals)", min_value=1, max_value=5000, value=default_horizon)

# Ensemble weights
weights = None
if model_choice == "Ensemble — weighted" and len(all_models) > 1:
    st.markdown("**Ensemble Weights**")
    w_cols  = st.columns(len(all_models))
    weights = []
    for i, mdl in enumerate(all_models):
        with w_cols[i]:
            w = st.number_input(mdl["model_name"], min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=f"w_{i}")
            weights.append(w)

st.divider()

# ── GENERATE FORECAST ─────────────────────────────────────────────────────────
if st.button("Generate Forecast", type="primary"):

    with st.spinner("Generating forecast..."):

        freq = granularity

        if model_choice == "Best model (auto)":
            target_model = best_model
            fc = generate_forecast(target_model, horizon, proc_series, freq)
            if fc["success"]:
                fc["model_name"] = target_model["model_name"]
            forecasts_all = [fc]

        elif model_choice.startswith("Ensemble"):
            forecasts_all = []
            for mdl in all_models:
                fc_i = generate_forecast(mdl, horizon, proc_series, freq)
                if fc_i["success"]:
                    forecasts_all.append(fc_i)

            if forecasts_all:
                ensemble_point = build_ensemble(forecasts_all, weights)
                # Interval from spread of member forecasts
                all_points = pd.concat([f["point"] for f in forecasts_all], axis=1)
                lower = all_points.min(axis=1)
                upper = all_points.max(axis=1)
                fc = {
                    "success":    True,
                    "point":      ensemble_point,
                    "lower":      lower,
                    "upper":      upper,
                    "model_name": model_choice,
                }
            else:
                fc = {"success": False, "error": "All ensemble members failed."}

        else:
            target_model = next((r for r in all_models if r["model_name"] == model_choice), None)
            if target_model is None:
                st.error("Selected model not found.")
                st.stop()
            fc = generate_forecast(target_model, horizon, proc_series, freq)
            if fc["success"]:
                fc["model_name"] = model_choice
            forecasts_all = [fc]

        if not fc.get("success"):
            st.error(f"Forecast generation failed: {fc.get('error')}")
            st.stop()

        st.session_state["current_forecast"] = fc

st.divider()

fc = st.session_state.get("current_forecast")
if fc is None:
    st.info("Configure the forecast above and click Generate Forecast.")
    st.stop()

# ── FORECAST PLOT ─────────────────────────────────────────────────────────────
st.subheader("Forecast")

# Show last 20% of history + full forecast
tail_n = max(50, int(len(proc_series) * 0.20))
hist_tail = proc_series.iloc[-tail_n:]

st.plotly_chart(
    plot_forecast(
        hist_tail,
        fc["point"],
        fc.get("lower"),
        fc.get("upper"),
        model_name=fc.get("model_name", "Forecast"),
    ),
    width='stretch',
)

# Forecast summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Periods Forecast",  f"{len(fc['point']):,}")
c2.metric("Mean Forecast",     f"{fc['point'].mean():,.1f}")
c3.metric("Peak Forecast",     f"{fc['point'].max():,.1f}")
c4.metric("Min Forecast",      f"{fc['point'].min():,.1f}")

# What-if adjustment
st.divider()
st.subheader("What-If Scenario")
st.caption("Adjust the forecast volume and see the impact on staffing before committing to a plan.")

volume_adj_pct = st.slider("Volume adjustment (%)", -50, 100, 0, 5)
adjusted_point = fc["point"] * (1 + volume_adj_pct / 100)
if volume_adj_pct != 0:
    st.info(
        f"Adjusted forecast: {volume_adj_pct:+d}% — "
        f"new mean = {adjusted_point.mean():,.1f}, "
        f"new peak = {adjusted_point.max():,.1f}"
    )

forecast_for_staffing = adjusted_point

st.divider()

# ── ERLANG C STAFFING PLAN ────────────────────────────────────────────────────
st.subheader("Erlang C Staffing Conversion")
st.caption(
    "Converts each forecast interval into a required headcount using the Erlang C queueing model. "
    "Accounts for target service level, AHT, and shrinkage."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    aht_seconds = st.number_input("Average Handle Time (seconds)", min_value=1, value=ERLANG_DEFAULTS["aht_seconds"])
with c2:
    target_sl   = st.slider("Target Service Level (%)", 50, 99, int(ERLANG_DEFAULTS["target_sl"] * 100)) / 100
with c3:
    answer_time = st.number_input("Target Answer Time (seconds)", min_value=1, value=ERLANG_DEFAULTS["target_answer_time_seconds"])
with c4:
    shrinkage   = st.slider("Shrinkage (%)", 0, 60, int(ERLANG_DEFAULTS["shrinkage"] * 100)) / 100

# Interval duration in seconds
interval_seconds_map = {"15T": 900, "30T": 1800, "1H": 3600, "1D": 86400, "1W": 604800}
interval_seconds = interval_seconds_map.get(granularity, 1800)

if st.button("Run Erlang C Staffing Plan"):
    with st.spinner("Calculating staffing requirements..."):
        staffing_df = build_staffing_plan(
            forecast_series=forecast_for_staffing,
            aht_seconds=aht_seconds,
            interval_seconds=interval_seconds,
            target_sl=target_sl,
            target_answer_seconds=answer_time,
            shrinkage=shrinkage,
        )
        st.session_state["staffing_df"] = staffing_df

staffing_df = st.session_state.get("staffing_df")
if staffing_df is not None:
    st.plotly_chart(plot_staffing_plan(staffing_df), width='stretch')

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Agents Required", int(staffing_df["Agents Required"].max()))
    c2.metric("Mean Agents Required", f"{staffing_df['Agents Required'].mean():.1f}")
    c3.metric("Total Agent-Intervals", f"{staffing_df['Agents Required'].sum():,}")

    st.dataframe(staffing_df.head(100), width='stretch', hide_index=True)
    if len(staffing_df) > 100:
        st.caption(f"Showing first 100 of {len(staffing_df):,} intervals.")

    st.divider()

    # ── EXPORT ────────────────────────────────────────────────────────────────
    st.subheader("Export")

    forecast_export_df = fc["point"].reset_index()
    forecast_export_df.columns = ["Timestamp", "Forecast Volume"]
    if fc.get("lower") is not None:
        forecast_export_df["Lower Bound"] = fc["lower"].values
        forecast_export_df["Upper Bound"] = fc["upper"].values
    forecast_export_df["Model"] = fc.get("model_name", "")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="Download Forecast (CSV)",
            data=export_forecast_csv(forecast_export_df),
            file_name="wfm_forecast.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            label="Download Forecast + Staffing Plan (Excel)",
            data=export_staffing_excel(forecast_export_df, staffing_df),
            file_name="wfm_forecast_staffing.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )