"""
config.py
Global configuration for the Just In Time Workforce Solutions Inc. WFM Analytics Platform.
"""

APP_NAME     = "WFM Analytics Platform"
COMPANY_NAME = "Just In Time Workforce Solutions Inc."
APP_VERSION  = "1.0.0"

SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls"]

GRANULARITY_OPTIONS = {
    "15 minutes": "15T",
    "30 minutes": "30T",
    "1 hour":     "1H",
    "1 day":      "1D",
    "1 week":     "1W",
}

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

STATIONARITY_ALPHA   = 0.05
OUTLIER_Z_THRESHOLD  = 3.0
ROLLING_WINDOWS      = [7, 14, 28]

ERLANG_DEFAULTS = {
    "target_sl":                  0.80,
    "target_answer_time_seconds": 20,
    "aht_seconds":                300,
    "shrinkage":                  0.30,
}

MODEL_NAMES = {
    "arima":        "ARIMA / SARIMA",
    "ets":          "ETS",
    "holtwinters":  "Holt-Winters",
    "prophet":      "Prophet",
    "xgboost":      "XGBoost",
    "lightgbm":     "LightGBM",
    "randomforest": "Random Forest",
    "ridge":        "Ridge Regression",
    "lstm":         "LSTM",
    "gru":          "GRU",
}

EVAL_METRICS = ["MAE", "RMSE", "MAPE", "sMAPE", "WAPE"]

CHART_COLORS = {
    "primary":   "#1B3A5C",
    "secondary": "#2E6DA4",
    "accent":    "#C8974E",
    "success":   "#2E7D52",
    "warning":   "#B85C1A",
    "danger":    "#8B1A1A",
    "neutral":   "#4A5568",
    "light":     "#EDF2F7",
    "grid":      "#E2E8F0",
    "text":      "#1A202C",
    "muted":     "#718096",
}

PLOTLY_TEMPLATE = "plotly_white"

FORECAST_HORIZON_DEFAULTS = {
    "15T": 672,
    "30T": 336,
    "1H":  168,
    "1D":  30,
    "1W":  12,
}