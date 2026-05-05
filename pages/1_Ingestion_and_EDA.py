"""
pages/1_Ingestion_and_EDA.py
Data upload, validation, and full EDA pipeline on a single page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Ingestion and EDA — WFM Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.data_utils import (
    load_file, detect_datetime_columns, detect_numeric_columns,
    infer_granularity, validate_dataset, prepare_dataframe, data_quality_scorecard,
)
from utils.eda.descriptive  import compute_descriptive_stats, plot_distribution, compute_ranked_intervals, plot_rolling_stats
from utils.eda.patterns     import (plot_intraday_profile, plot_intraday_by_dow, compute_dow_index,
                                    plot_dow_bar, plot_dow_stability, plot_hour_dow_heatmap,
                                    plot_month_dom_heatmap, plot_yoy_comparison, compute_week_of_month_effect)
from utils.eda.decomposition import (run_stl_decomposition, run_classical_decomposition,
                                     detect_dominant_periods_fft, plot_decomposition, seasonality_strength_gauge)
from utils.eda.volatility   import (plot_rolling_cv, detect_outliers, plot_outlier_series,
                                    worst_case_profile, plot_p90_vs_mean_gap, poisson_fit_test)
from utils.eda.holidays     import get_ph_holidays, compute_holiday_impact, compute_halo_effect, plot_holiday_impact_chart
from utils.eda.stationarity import (run_adf_test, run_kpss_test, combined_stationarity_interpretation,
                                    compute_acf_pacf, plot_acf_pacf, recommend_arima_order)
from utils.eda.operational  import (check_available_metrics, plot_aht_trend, plot_aht_volume_scatter,
                                    plot_sl_heatmap, compute_occupancy_risk, compute_shrinkage_backfill)
from config import GRANULARITY_OPTIONS, CHART_COLORS

st.title("Ingestion and EDA")
st.caption("Just In Time Workforce Solutions Inc. — WFM Analytics Platform")

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────

st.subheader("Upload Dataset")
uploaded = st.file_uploader(
    "Upload a WFM dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="Supported exports from NICE, Verint, Genesys, Aspect, or any WFM platform.",
)

if uploaded is None:
    st.info("Upload a file above to begin.")
    st.stop()

df_raw, err = load_file(uploaded)
if err:
    st.error(err)
    st.stop()

st.success(f"File loaded — {len(df_raw):,} rows, {len(df_raw.columns)} columns.")

with st.expander("Preview raw data (first 50 rows)"):
    st.dataframe(df_raw.head(50), use_container_width=True)

# ── COLUMN MAPPING ────────────────────────────────────────────────────────────

st.subheader("Column Mapping")

datetime_cols = detect_datetime_columns(df_raw)
all_cols      = list(df_raw.columns)
numeric_cols  = detect_numeric_columns(df_raw)

c1, c2 = st.columns(2)
with c1:
    date_col = st.selectbox(
        "Date / time column",
        options=all_cols,
        index=all_cols.index(datetime_cols[0]) if datetime_cols else 0,
    )
with c2:
    vol_suggestions = [c for c in numeric_cols if any(
        kw in c.lower() for kw in ["volume", "calls", "contacts", "offered", "arrivals", "count"]
    )]
    default_vol = vol_suggestions[0] if vol_suggestions else (numeric_cols[0] if numeric_cols else all_cols[0])
    target_col = st.selectbox(
        "Volume / offered contacts column (target)",
        options=all_cols,
        index=all_cols.index(default_vol),
    )

optional_cols = st.multiselect(
    "Optional operational columns — AHT, abandoned, SL%, occupancy, headcount (used in operational metrics module)",
    options=[c for c in all_cols if c not in [date_col, target_col]],
)

# Granularity
try:
    dt_sample    = pd.to_datetime(df_raw[date_col].dropna())
    inferred_gran = infer_granularity(dt_sample)
except Exception:
    inferred_gran = "1D"

gran_label_map    = {v: k for k, v in GRANULARITY_OPTIONS.items()}
default_gran_label = gran_label_map.get(inferred_gran, "1 day")

c3, c4 = st.columns([1, 2])
with c3:
    gran_label = st.selectbox(
        "Interval granularity",
        options=list(GRANULARITY_OPTIONS.keys()),
        index=list(GRANULARITY_OPTIONS.keys()).index(default_gran_label)
              if default_gran_label in GRANULARITY_OPTIONS else 3,
    )
with c4:
    st.caption(f"Auto-detected: **{default_gran_label}**")

granularity = GRANULARITY_OPTIONS[gran_label]

# ── VALIDATION ────────────────────────────────────────────────────────────────

st.subheader("Data Quality Checks")

validation = validate_dataset(df_raw.copy(), date_col, target_col)

status_map = {"ok": "PASS", "warn": "WARNING", "error": "FAIL"}
check_rows = [
    {
        "Check":  c["check"],
        "Status": status_map[c["status"]],
        "Detail": c["detail"],
    }
    for c in validation["checks"]
]
st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)

if not validation["passed"]:
    st.error("One or more critical checks failed. Fix the issues above before continuing.")
    st.stop()

# ── SCORECARD ─────────────────────────────────────────────────────────────────

prepared_df = prepare_dataframe(df_raw.copy(), date_col, target_col, optional_cols)
scorecard   = data_quality_scorecard(prepared_df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Rows",     f"{scorecard['row_count']:,}")
c2.metric("Date Range",     scorecard["date_range"])
c3.metric("Mean Volume",    f"{scorecard['mean']:,.1f}")
c4.metric("Missing Values", f"{scorecard['missing_pct']}%", f"{scorecard['missing_count']} intervals")
c5.metric("CV",             f"{scorecard['cv']}%", "Coefficient of Variation")

# Save to session state for downstream pages
st.session_state["raw_df"]          = prepared_df
st.session_state["granularity"]     = granularity
st.session_state["data_loaded"]     = True
st.session_state["holiday_dict"]    = get_ph_holidays(sorted(prepared_df.index.year.unique().tolist()))

st.divider()

# ── EDA TABS ──────────────────────────────────────────────────────────────────

df     = prepared_df
series = df["volume"]

def default_period(gran):
    return {"15T": 96, "30T": 48, "1H": 24, "1D": 7, "1W": 52}.get(gran, 7)

period = default_period(granularity)

st.subheader("Exploratory Data Analysis")

tabs = st.tabs([
    "Descriptive",
    "Volume Patterns",
    "Decomposition",
    "Holidays and Events",
    "Volatility and Risk",
    "Operational Metrics",
    "Stationarity",
    "Summary",
])

# ── TAB 1: Descriptive ────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("#### Descriptive Statistics")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.dataframe(compute_descriptive_stats(series), use_container_width=True, hide_index=True, height=540)
    with c2:
        st.plotly_chart(plot_distribution(series), use_container_width=True)

    st.markdown("#### Rolling Mean and Volatility Band")
    window = st.slider("Rolling window (periods)", 2, 60, 7, key="roll_win")
    st.plotly_chart(plot_rolling_stats(series, window=window), use_container_width=True)

    st.markdown("#### Period Rankings")
    ranked = compute_ranked_intervals(df, granularity)
    cols = st.columns(3)
    if "busiest_hours" in ranked:
        with cols[0]:
            st.markdown("**Busiest hours**")
            st.dataframe(ranked["busiest_hours"].round(1), hide_index=True, use_container_width=True)
    with cols[1]:
        st.markdown("**Busiest days**")
        st.dataframe(ranked["busiest_days"].round(1), hide_index=True, use_container_width=True)
    with cols[2]:
        st.markdown("**Busiest months**")
        st.dataframe(ranked["busiest_months"].round(1), hide_index=True, use_container_width=True)


# ── TAB 2: Volume Patterns ────────────────────────────────────────────────────
with tabs[1]:
    if granularity in ("15T", "30T", "1H"):
        st.markdown("#### Intraday Volume Profile")
        st.plotly_chart(plot_intraday_profile(df, granularity), use_container_width=True)

        st.markdown("#### Intraday Profile by Day of Week")
        st.plotly_chart(plot_intraday_by_dow(df, granularity), use_container_width=True)

    st.markdown("#### Day-of-Week Analysis")
    dow_df = compute_dow_index(df)
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.dataframe(dow_df.round(3), hide_index=True, use_container_width=True)
        st.caption("Index > 1.0 means above weekly average.")
    with c2:
        st.plotly_chart(plot_dow_bar(dow_df), use_container_width=True)

    st.markdown("#### DOW Stability (Week-over-Week Spread)")
    st.plotly_chart(plot_dow_stability(df), use_container_width=True)

    if granularity in ("15T", "30T", "1H"):
        st.markdown("#### Hour x Day-of-Week Heatmap")
        h = plot_hour_dow_heatmap(df, granularity)
        if h:
            st.plotly_chart(h, use_container_width=True)

    st.markdown("#### Month x Day-of-Month Heatmap")
    st.plotly_chart(plot_month_dom_heatmap(df), use_container_width=True)

    yoy = plot_yoy_comparison(df)
    if yoy:
        st.markdown("#### Year-over-Year Comparison")
        st.plotly_chart(yoy, use_container_width=True)

    st.markdown("#### Week-of-Month Effect")
    st.dataframe(compute_week_of_month_effect(df).round(2), hide_index=True, use_container_width=True)
    st.caption("Significant variation may indicate billing cycle effects.")


# ── TAB 3: Decomposition ──────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### Trend and Seasonality Decomposition")
    c1, c2 = st.columns(2)
    with c1:
        decomp_method = st.selectbox(
            "Method",
            ["STL (recommended)", "Classical — Additive", "Classical — Multiplicative"],
        )
    with c2:
        period_input = st.number_input("Seasonal period (intervals)", min_value=2, max_value=1000, value=period)

    st.markdown("#### FFT Period Detection")
    fft_periods = detect_dominant_periods_fft(series, top_n=5)
    fft_df = pd.DataFrame(fft_periods, columns=["Period (intervals)", "Relative Power"])
    fft_df["Relative Power"] = fft_df["Relative Power"].apply(lambda x: f"{x:.4f}")
    st.dataframe(fft_df, hide_index=True, use_container_width=True)
    st.caption("Use the highest-power period as the seasonal period input above.")

    if decomp_method.startswith("STL"):
        result = run_stl_decomposition(series, int(period_input))
    elif "Additive" in decomp_method:
        result = run_classical_decomposition(series, int(period_input), "additive")
    else:
        result = run_classical_decomposition(series, int(period_input), "multiplicative")

    if result["success"]:
        components = {
            "observed": series,
            "trend":    result["trend"],
            "seasonal": result["seasonal"],
            "resid":    result["resid"],
        }
        st.plotly_chart(plot_decomposition(components, title=decomp_method), use_container_width=True)

        if "seasonal_strength" in result:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(seasonality_strength_gauge(result["seasonal_strength"], "Seasonal Strength"), use_container_width=True)
            with c2:
                st.plotly_chart(seasonality_strength_gauge(result["trend_strength"], "Trend Strength"), use_container_width=True)
            st.info(
                f"Seasonal strength: {result['seasonal_strength']:.1%}. "
                f"Trend strength: {result['trend_strength']:.1%}. "
                "Values above 70% indicate components the model must explicitly capture."
            )
            st.session_state["stl_season_strength"] = result["seasonal_strength"]
    else:
        st.error(f"Decomposition failed: {result.get('error')}")

    st.session_state["detected_period"] = fft_periods[0][0] if fft_periods else period


# ── TAB 4: Holidays and Events ────────────────────────────────────────────────
with tabs[3]:
    st.markdown("#### Holiday and Special Event Analysis")

    years        = sorted(df.index.year.unique().tolist())
    holiday_dict = get_ph_holidays(years)

    st.caption(
        f"Philippine public holiday calendar loaded for {min(years)}–{max(years)} "
        f"({len(holiday_dict)} dates). Add custom events below."
    )

    with st.expander("Add custom events"):
        custom_input = st.text_area(
            "One entry per line: date,event name — e.g. 2024-03-15,Product Launch",
            height=100,
        )
        if custom_input.strip():
            for line in custom_input.strip().split("\n"):
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    holiday_dict[parts[0].strip()] = parts[1].strip()

    st.session_state["holiday_dict"] = holiday_dict

    impact_df = compute_holiday_impact(df, holiday_dict)
    if not impact_df.empty:
        st.markdown("#### Holiday Volume Impact vs. Baseline")
        fig = plot_holiday_impact_chart(impact_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(impact_df, hide_index=True, use_container_width=True)

        st.markdown("#### Pre / Post Holiday Halo Effect")
        days_w = st.slider("Days before/after holiday", 1, 5, 2)
        halo_df = compute_halo_effect(df, holiday_dict, days_window=days_w)
        if not halo_df.empty:
            st.dataframe(halo_df, hide_index=True, use_container_width=True)
        else:
            st.info("Not enough data around holidays to compute halo effect.")
    else:
        st.info("No matching holiday dates found in the dataset date range.")


# ── TAB 5: Volatility and Risk ────────────────────────────────────────────────
with tabs[4]:
    st.markdown("#### Rolling Coefficient of Variation")
    st.caption("A rising CV over time signals increasing staffing risk.")
    st.plotly_chart(plot_rolling_cv(series, windows=[7, 14, 28]), use_container_width=True)

    st.markdown("#### Outlier Detection")
    c1, c2 = st.columns(2)
    with c1:
        outlier_method = st.selectbox("Method", ["zscore", "iqr"])
    with c2:
        z_thresh = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.5) if outlier_method == "zscore" else 3.0

    outlier_mask = detect_outliers(series, method=outlier_method, threshold=z_thresh)
    st.caption(f"{outlier_mask.sum()} outlier(s) detected ({outlier_mask.sum()/len(series)*100:.2f}% of series).")
    st.plotly_chart(plot_outlier_series(series, outlier_mask), use_container_width=True)
    st.session_state["outlier_mask"] = outlier_mask

    st.markdown("#### Worst-Case Day Profile (Top 5%)")
    wc = worst_case_profile(series, top_pct=0.05)
    c1, c2, c3 = st.columns(3)
    c1.metric("P95 Threshold",   f"{wc['threshold']:,.1f}")
    c2.metric("Worst-Case Count", f"{wc['count']:,}")
    c3.metric("Frequency",        f"{wc['frequency_pct']}%")

    wc_df = wc["dates"].reset_index()
    wc_df.columns = ["Timestamp", "Volume"]
    st.dataframe(wc_df, hide_index=True, use_container_width=True)

    if granularity in ("15T", "30T", "1H"):
        st.markdown("#### Mean vs. P90 Gap per Interval")
        st.caption("The gap represents the staffing buffer needed above the forecast to hit SL targets.")
        gap_fig = plot_p90_vs_mean_gap(df, granularity)
        if gap_fig:
            st.plotly_chart(gap_fig, use_container_width=True)

    st.markdown("#### Poisson Arrival Distribution Test")
    st.caption("Erlang C assumes Poisson arrivals. A violation means queue estimates may understaff.")
    p_result = poisson_fit_test(series)
    if p_result.get("success"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Index of Dispersion", f"{p_result['index_of_dispersion']}", "1.0 = perfect Poisson")
        c2.metric("Chi-squared p-value",  f"{p_result['p_value']}")
        c3.metric("Arrival Pattern",      "Poisson" if p_result["is_poisson"] else "Non-Poisson")
        st.info(p_result["interpretation"])
    else:
        st.warning(p_result.get("reason", "Test could not be completed."))


# ── TAB 6: Operational Metrics ────────────────────────────────────────────────
with tabs[5]:
    available = check_available_metrics(df)
    active    = [k for k, v in available.items() if v]

    if not active:
        st.info(
            "No operational columns were mapped. Return to the column mapping section above "
            "and add columns such as AHT, abandoned, SL%, occupancy, or headcount."
        )
    else:
        st.caption(f"Available operational metrics: {', '.join(active)}")

        if available["aht"]:
            st.markdown("#### AHT Trend")
            fig = plot_aht_trend(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### AHT vs. Volume")
            fig2 = plot_aht_volume_scatter(df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("A positive slope suggests AHT rises under load — watch for agent burnout.")

        if available["sl_pct"] and granularity in ("15T", "30T", "1H"):
            st.markdown("#### Service Level Achievement Heatmap")
            fig3 = plot_sl_heatmap(df)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)

        if available["occupancy"]:
            st.markdown("#### Occupancy Risk (threshold: 85%)")
            occ = compute_occupancy_risk(df)
            if occ:
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Occupancy",      f"{occ['mean_occupancy']}%")
                c2.metric("Intervals Above 85%", f"{occ['intervals_at_risk']:,}", f"{occ['pct_above_threshold']}% of total")
                c3.metric("Risk Level",          occ["risk_level"])
                if occ["risk_level"] != "Low":
                    st.warning(
                        f"{occ['pct_above_threshold']}% of intervals exceed 85% occupancy. "
                        "This is a leading indicator of agent burnout and quality issues."
                    )

        if available["headcount"] and available["aht"]:
            st.markdown("#### Shrinkage Back-Calculation")
            shrink = compute_shrinkage_backfill(df)
            if shrink:
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Shrinkage",   f"{shrink['mean_shrinkage']}%")
                c2.metric("Median Shrinkage", f"{shrink['median_shrinkage']}%")
                c3.metric("P90 Shrinkage",    f"{shrink['p90_shrinkage']}%", "Use for conservative planning")


# ── TAB 7: Stationarity ───────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("#### Stationarity Tests")

    adf_result  = run_adf_test(series)
    kpss_result = run_kpss_test(series)

    c1, c2 = st.columns(2)
    for col, res in [(c1, adf_result), (c2, kpss_result)]:
        with col:
            st.markdown(f"**{res['test']}**")
            is_stat = res.get("is_stationary")
            label   = "Stationary" if is_stat else "Non-Stationary"
            color   = "normal" if is_stat else "inverse"
            st.markdown(f"`{label}`")
            rows = [
                ("Test Statistic", res.get("statistic", "N/A")),
                ("p-value",        res.get("p_value",   "N/A")),
                ("Lags Used",      res.get("n_lags_used", "N/A")),
                ("Critical 1%",    res.get("critical_1pct", "N/A")),
                ("Critical 5%",    res.get("critical_5pct", "N/A")),
            ]
            st.dataframe(
                pd.DataFrame(rows, columns=["Metric", "Value"]),
                hide_index=True,
                use_container_width=True,
            )

    st.info(combined_stationarity_interpretation(adf_result, kpss_result))

    st.markdown("#### ACF and PACF")
    n_lags_input = st.slider("Number of lags", 12, 144, min(48, len(series) // 2 - 2))
    acf_pacf = compute_acf_pacf(series, n_lags=n_lags_input)
    st.plotly_chart(plot_acf_pacf(acf_pacf), use_container_width=True)
    st.info(recommend_arima_order(acf_pacf))

    st.session_state["stationarity"] = {
        "adf":            adf_result,
        "kpss":           kpss_result,
        "interpretation": combined_stationarity_interpretation(adf_result, kpss_result),
        "recommendation": recommend_arima_order(acf_pacf),
    }


# ── TAB 8: Summary ────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("#### WFM Data Intelligence Summary")

    cv             = series.std() / series.mean() * 100 if series.mean() != 0 else 0
    stl            = run_stl_decomposition(series, period)
    seas_strength  = stl.get("seasonal_strength", 0) if stl.get("success") else 0
    trend_strength = stl.get("trend_strength",    0) if stl.get("success") else 0

    trend_series = stl.get("trend") if stl.get("success") else None
    if trend_series is not None and len(trend_series.dropna()) > 1:
        tv    = trend_series.dropna()
        slope = (tv.iloc[-1] - tv.iloc[0]) / len(tv)
        trend_dir  = "upward" if slope > 0 else "downward"
        trend_rate = abs(slope / tv.iloc[0] * 100) if tv.iloc[0] != 0 else 0
    else:
        trend_dir, trend_rate = "indeterminate", 0

    fft_pds        = detect_dominant_periods_fft(series, top_n=3)
    top_period_str = ", ".join([str(p[0]) for p in fft_pds]) if fft_pds else "Not detected"
    dow_df_sum     = compute_dow_index(df)
    riskiest_dow   = dow_df_sum.loc[dow_df_sum["DOW Index"].idxmax(), "Day"] if not dow_df_sum.empty else "N/A"
    outlier_mask_s = detect_outliers(series)
    n_out          = outlier_mask_s.sum()
    adf_r          = run_adf_test(series)
    kpss_r         = run_kpss_test(series)

    if seas_strength >= 0.70:
        model_rec = "SARIMA or Prophet — strong seasonality detected."
    elif seas_strength >= 0.40:
        model_rec = "ETS or Prophet — moderate seasonality. XGBoost with calendar features also viable."
    else:
        model_rec = "ARIMA or ML models — low seasonality. Focus on trend and lag features."

    summary = pd.DataFrame([
        ("Dataset",           f"{len(df):,} rows | {df.index.min().date()} to {df.index.max().date()} | {gran_label} granularity"),
        ("Detected Periods",  f"{top_period_str} intervals"),
        ("Volume Trend",      f"{trend_dir.capitalize()} at {trend_rate:.2f}% per interval"),
        ("Seasonal Strength", f"{seas_strength:.1%} — {'dominant' if seas_strength > 0.70 else 'moderate' if seas_strength > 0.40 else 'weak'}"),
        ("Trend Strength",    f"{trend_strength:.1%}"),
        ("Volatility (CV)",   f"{cv:.1f}% — {'high' if cv > 30 else 'moderate' if cv > 15 else 'low'}"),
        ("Riskiest DOW",      riskiest_dow),
        ("Outliers",          f"{n_out} ({round(n_out/len(series)*100,2)}%)"),
        ("Stationarity",      combined_stationarity_interpretation(adf_r, kpss_r)),
        ("Recommended Model", model_rec),
    ], columns=["Dimension", "Finding"])

    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.session_state["eda_summary"]         = summary.to_dict("records")
    st.session_state["stl_season_strength"] = seas_strength
    st.session_state["detected_period"]     = fft_pds[0][0] if fft_pds else period
    st.session_state["model_recommendation"] = model_rec