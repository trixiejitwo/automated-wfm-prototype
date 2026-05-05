"""
utils/models/statistical.py
ARIMA/SARIMA, ETS, Holt-Winters, and Prophet model wrappers.
Each function returns a dict: {success, model_name, predictions, fitted_values, model_obj, error}
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def fit_arima(train: pd.Series, test: pd.Series, config: dict) -> dict:
    """
    Fit ARIMA or SARIMA using pmdarima auto_arima or manual order.
    """
    try:
        import pmdarima as pm

        seasonal = config.get("seasonal", True)
        period   = int(config.get("period", 7))

        if config.get("auto", True):
            model = pm.auto_arima(
                train,
                seasonal=seasonal,
                m=period if seasonal else 1,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                information_criterion="aic",
                max_p=5, max_q=5, max_P=2, max_Q=2,
            )
        else:
            p, d, q = config.get("p", 1), config.get("d", 1), config.get("q", 0)
            P, D, Q = config.get("P", 0), config.get("D", 0), config.get("Q", 0)
            order         = (p, d, q)
            seasonal_order = (P, D, Q, period) if seasonal else (0, 0, 0, 0)
            model = pm.ARIMA(order=order, seasonal_order=seasonal_order)
            model.fit(train)

        fitted      = pd.Series(model.predict_in_sample(), index=train.index)
        predictions = pd.Series(model.predict(n_periods=len(test)), index=test.index)

        return {
            "success":       True,
            "model_name":    "ARIMA / SARIMA",
            "key":           "arima",
            "predictions":   predictions,
            "fitted_values": fitted,
            "model_obj":     model,
            "order":         str(model.order),
        }
    except Exception as e:
        return {"success": False, "model_name": "ARIMA / SARIMA", "key": "arima", "error": str(e)}


def fit_ets(train: pd.Series, test: pd.Series, config: dict) -> dict:
    """
    Fit ETS (ExponentialSmoothing) model via statsmodels.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        error    = config.get("error", "add")
        trend    = config.get("trend", "add")
        seasonal = config.get("seasonal", "add")
        period   = int(config.get("period", 7))

        # statsmodels uses None instead of "none"
        trend_arg    = None if trend    == "none" else trend
        seasonal_arg = None if seasonal == "none" else seasonal

        model = ExponentialSmoothing(
            train,
            trend=trend_arg,
            seasonal=seasonal_arg,
            seasonal_periods=period if seasonal_arg else None,
            initialization_method="estimated",
        ).fit(optimized=True)

        fitted      = model.fittedvalues
        predictions = pd.Series(model.forecast(len(test)), index=test.index)

        return {
            "success":       True,
            "model_name":    "ETS",
            "key":           "ets",
            "predictions":   predictions,
            "fitted_values": fitted,
            "model_obj":     model,
        }
    except Exception as e:
        return {"success": False, "model_name": "ETS", "key": "ets", "error": str(e)}


def fit_holtwinters(train: pd.Series, test: pd.Series, config: dict) -> dict:
    """
    Fit Holt-Winters triple exponential smoothing.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        period = int(config.get("period", 7))
        damped = config.get("damped", True)

        model = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=damped,
            seasonal="add",
            seasonal_periods=period,
            initialization_method="estimated",
        ).fit(optimized=True)

        fitted      = model.fittedvalues
        predictions = pd.Series(model.forecast(len(test)), index=test.index)

        return {
            "success":       True,
            "model_name":    "Holt-Winters",
            "key":           "holtwinters",
            "predictions":   predictions,
            "fitted_values": fitted,
            "model_obj":     model,
        }
    except Exception as e:
        return {"success": False, "model_name": "Holt-Winters", "key": "holtwinters", "error": str(e)}


def fit_prophet(train: pd.Series, test: pd.Series, config: dict, holiday_dict: dict = None) -> dict:
    """
    Fit Facebook Prophet model.
    """
    try:
        from prophet import Prophet

        changepoint_prior  = config.get("changepoint_prior", 0.05)
        seasonality_prior  = config.get("seasonality_prior", 1.0)
        yearly_seasonality = config.get("yearly_seasonality", True)
        weekly_seasonality = config.get("weekly_seasonality", True)
        daily_seasonality  = config.get("daily_seasonality",  True)
        use_holidays       = config.get("use_holidays", True)

        # Build holidays dataframe
        holidays_df = None
        if use_holidays and holiday_dict:
            holiday_rows = []
            for date_str, name in holiday_dict.items():
                try:
                    holiday_rows.append({"ds": pd.Timestamp(date_str), "holiday": name})
                except Exception:
                    pass
            if holiday_rows:
                holidays_df = pd.DataFrame(holiday_rows)

        m = Prophet(
            changepoint_prior_scale=changepoint_prior,
            seasonality_prior_scale=seasonality_prior,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays_df,
        )

        train_df = train.reset_index().rename(columns={train.index.name or "ds": "ds", "volume": "y"})
        if "y" not in train_df.columns:
            train_df.columns = ["ds", "y"]

        m.fit(train_df)

        future       = m.make_future_dataframe(periods=len(test), freq=train.index.freq or "D")
        forecast_all = m.predict(future)

        fitted_vals = forecast_all.iloc[:len(train)][["ds", "yhat"]].set_index("ds")["yhat"]
        fitted_vals.index = train.index

        pred_vals = forecast_all.iloc[len(train):len(train)+len(test)][["ds", "yhat"]].set_index("ds")["yhat"]
        pred_vals.index = test.index

        return {
            "success":       True,
            "model_name":    "Prophet",
            "key":           "prophet",
            "predictions":   pred_vals,
            "fitted_values": fitted_vals,
            "model_obj":     m,
            "forecast_df":   forecast_all,
        }
    except Exception as e:
        return {"success": False, "model_name": "Prophet", "key": "prophet", "error": str(e)}