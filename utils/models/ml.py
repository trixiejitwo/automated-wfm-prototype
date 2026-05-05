"""
utils/models/ml.py
ML model wrappers: XGBoost, LightGBM, Random Forest, Ridge.
All models use lag + calendar features built from the time series index.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(series: pd.Series, lag_cols: list, rolling_cols: list,
                   calendar_cols: list, holiday_dict: dict = None) -> pd.DataFrame:
    """
    Build a feature matrix from a volume series.
    Returns DataFrame with features aligned on the same index as series.
    """
    df = pd.DataFrame({"volume": series})

    # Lag features
    lag_map = {
        "lag_1": 1, "lag_7": 7, "lag_14": 14, "lag_28": 28,
        "lag_48": 48, "lag_336": 336,
    }
    for col in lag_cols:
        if col in lag_map:
            df[col] = df["volume"].shift(lag_map[col])

    # Rolling features
    roll_map = {
        "rolling_mean_7":  (7,  "mean"),
        "rolling_mean_14": (14, "mean"),
        "rolling_mean_28": (28, "mean"),
        "rolling_std_7":   (7,  "std"),
        "rolling_std_28":  (28, "std"),
    }
    for col in rolling_cols:
        if col in roll_map:
            win, fn = roll_map[col]
            if fn == "mean":
                df[col] = df["volume"].rolling(win, min_periods=1).mean().shift(1)
            else:
                df[col] = df["volume"].rolling(win, min_periods=1).std().shift(1)

    # Calendar features
    cal_map = {
        "hour":           lambda idx: idx.hour,
        "day_of_week":    lambda idx: idx.dayofweek,
        "day_of_month":   lambda idx: idx.day,
        "week_of_month":  lambda idx: (idx.day - 1) // 7 + 1,
        "month":          lambda idx: idx.month,
        "is_holiday":     None,
        "is_payday_week": lambda idx: ((idx.day >= 14) & (idx.day <= 16)) | (idx.day >= 28),
    }
    for col in calendar_cols:
        if col == "is_holiday":
            if holiday_dict:
                hol_dates = set(holiday_dict.keys())
                df["is_holiday"] = df.index.strftime("%Y-%m-%d").isin(hol_dates).astype(int)
            else:
                df["is_holiday"] = 0
        elif col in cal_map and cal_map[col] is not None:
            df[col] = cal_map[col](df.index).astype(int)

    return df


def split_features(df: pd.DataFrame, train_end: int, val_end: int):
    """Split feature matrix into train/val/test, drop rows with NaN from lag creation."""
    feature_cols = [c for c in df.columns if c != "volume"]
    df_clean     = df.dropna()

    train = df_clean.iloc[:train_end]
    val   = df_clean.iloc[train_end:val_end]
    test  = df_clean.iloc[val_end:]

    X_train, y_train = train[feature_cols], train["volume"]
    X_val,   y_val   = val[feature_cols],   val["volume"]
    X_test,  y_test  = test[feature_cols],  test["volume"]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ── Model Fitters ─────────────────────────────────────────────────────────────

def fit_xgboost(X_train, y_train, X_test, y_test, config: dict) -> dict:
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", 5),
            learning_rate=config.get("lr", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        fitted      = pd.Series(model.predict(X_train), index=X_train.index)
        predictions = pd.Series(model.predict(X_test),  index=X_test.index)
        importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        return {
            "success": True, "model_name": "XGBoost", "key": "xgboost",
            "predictions": predictions, "fitted_values": fitted,
            "model_obj": model, "feature_importances": importances,
        }
    except Exception as e:
        return {"success": False, "model_name": "XGBoost", "key": "xgboost", "error": str(e)}


def fit_lightgbm(X_train, y_train, X_test, y_test, config: dict) -> dict:
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", 5),
            learning_rate=config.get("lr", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        fitted      = pd.Series(model.predict(X_train), index=X_train.index)
        predictions = pd.Series(model.predict(X_test),  index=X_test.index)
        importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        return {
            "success": True, "model_name": "LightGBM", "key": "lightgbm",
            "predictions": predictions, "fitted_values": fitted,
            "model_obj": model, "feature_importances": importances,
        }
    except Exception as e:
        return {"success": False, "model_name": "LightGBM", "key": "lightgbm", "error": str(e)}


def fit_random_forest(X_train, y_train, X_test, y_test, config: dict) -> dict:
    try:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", None),
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        fitted      = pd.Series(model.predict(X_train), index=X_train.index)
        predictions = pd.Series(model.predict(X_test),  index=X_test.index)
        importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        return {
            "success": True, "model_name": "Random Forest", "key": "randomforest",
            "predictions": predictions, "fitted_values": fitted,
            "model_obj": model, "feature_importances": importances,
        }
    except Exception as e:
        return {"success": False, "model_name": "Random Forest", "key": "randomforest", "error": str(e)}


def fit_ridge(X_train, y_train, X_test, y_test, config: dict) -> dict:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        model  = Ridge(alpha=config.get("alpha", 1.0))
        model.fit(X_tr_s, y_train)
        fitted      = pd.Series(model.predict(X_tr_s), index=X_train.index)
        predictions = pd.Series(model.predict(X_te_s), index=X_test.index)
        return {
            "success": True, "model_name": "Ridge Regression", "key": "ridge",
            "predictions": predictions, "fitted_values": fitted,
            "model_obj": model,
        }
    except Exception as e:
        return {"success": False, "model_name": "Ridge Regression", "key": "ridge", "error": str(e)}