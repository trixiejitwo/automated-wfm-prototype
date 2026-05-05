"""
utils/data_utils.py
Data ingestion, validation, and schema utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from config import GRANULARITY_OPTIONS


# ── File loading ────────────────────────────────────────────────────────────

def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load a CSV or Excel file into a DataFrame.
    Returns (df, error_message). On success error_message is empty string.
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Please upload a CSV or Excel file."
        return df, ""
    except Exception as e:
        return None, f"Failed to read file: {str(e)}"


# ── Column detection ─────────────────────────────────────────────────────────

def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of column names that are likely datetime columns."""
    candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
            continue
        if df[col].dtype == object:
            sample = df[col].dropna().head(50)
            try:
                pd.to_datetime(sample)
                candidates.append(col)
            except Exception:
                pass
    return candidates


def detect_numeric_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """Return numeric columns, optionally excluding specified columns."""
    exclude = exclude or []
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def infer_granularity(series: pd.Series) -> str:
    """
    Infer time series granularity from a sorted datetime Series.
    Returns a pandas offset alias string.
    """
    if len(series) < 2:
        return "1D"
    diffs = series.sort_values().diff().dropna()
    median_diff = diffs.median()
    minutes = median_diff.total_seconds() / 60

    if minutes <= 16:
        return "15T"
    elif minutes <= 31:
        return "30T"
    elif minutes <= 61:
        return "1H"
    elif minutes <= 1500:
        return "1D"
    else:
        return "1W"


# ── Validation ───────────────────────────────────────────────────────────────

def validate_dataset(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> Dict:
    """
    Run a suite of data quality checks.
    Returns a dict with keys:
        passed (bool), checks (list of check dicts), summary (str)
    """
    checks = []

    # 1. Datetime parse
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        checks.append({"check": "Datetime parse", "status": "ok", "detail": f"Column '{date_col}' parsed successfully."})
    except Exception as e:
        checks.append({"check": "Datetime parse", "status": "error", "detail": str(e)})
        return {"passed": False, "checks": checks, "summary": "Datetime column could not be parsed."}

    # 2. Duplicate timestamps
    dup_count = df[date_col].duplicated().sum()
    if dup_count > 0:
        checks.append({"check": "Duplicate timestamps", "status": "warn",
                        "detail": f"{dup_count} duplicate timestamp(s) found."})
    else:
        checks.append({"check": "Duplicate timestamps", "status": "ok", "detail": "No duplicates."})

    # 3. Missing target values
    missing_pct = df[target_col].isna().mean() * 100
    if missing_pct > 20:
        checks.append({"check": "Missing target values", "status": "error",
                        "detail": f"{missing_pct:.1f}% of target values are missing."})
    elif missing_pct > 0:
        checks.append({"check": "Missing target values", "status": "warn",
                        "detail": f"{missing_pct:.1f}% missing — will be imputed."})
    else:
        checks.append({"check": "Missing target values", "status": "ok", "detail": "No missing values."})

    # 4. Negative values
    neg_count = (df[target_col] < 0).sum()
    if neg_count > 0:
        checks.append({"check": "Negative volume values", "status": "warn",
                        "detail": f"{neg_count} negative value(s) detected."})
    else:
        checks.append({"check": "Negative volume values", "status": "ok", "detail": "None found."})

    # 5. Zero-volume rows
    zero_pct = (df[target_col] == 0).mean() * 100
    if zero_pct > 50:
        checks.append({"check": "Zero-volume rows", "status": "warn",
                        "detail": f"{zero_pct:.1f}% of rows are zero — check for closures/overnight gaps."})
    else:
        checks.append({"check": "Zero-volume rows", "status": "ok",
                        "detail": f"{zero_pct:.1f}% zero rows."})

    # 6. Minimum row count
    row_count = len(df)
    if row_count < 100:
        checks.append({"check": "Minimum data volume", "status": "warn",
                        "detail": f"Only {row_count} rows. Models may underfit."})
    else:
        checks.append({"check": "Minimum data volume", "status": "ok",
                        "detail": f"{row_count:,} rows available."})

    # 7. Sorted order
    is_sorted = df[date_col].is_monotonic_increasing
    if not is_sorted:
        checks.append({"check": "Chronological order", "status": "warn",
                        "detail": "Data is not sorted by date. Will be sorted automatically."})
    else:
        checks.append({"check": "Chronological order", "status": "ok", "detail": "Data is sorted."})

    any_error = any(c["status"] == "error" for c in checks)
    summary = "One or more critical issues found." if any_error else "Data quality checks passed."

    return {"passed": not any_error, "checks": checks, "summary": summary}


# ── Preparation ──────────────────────────────────────────────────────────────

def prepare_dataframe(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    optional_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Standardize the DataFrame:
    - Parse and set datetime index
    - Sort chronologically
    - Rename target to 'volume'
    - Keep optional operational columns
    """
    keep_cols = [date_col, target_col] + (optional_cols or [])
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "ds", target_col: "volume"})
    df = df.set_index("ds")
    return df


def data_quality_scorecard(df: pd.DataFrame, target_col: str = "volume") -> Dict:
    """
    Generate a summary scorecard dict from a prepared DataFrame.
    """
    series = df[target_col].dropna()
    return {
        "row_count": len(df),
        "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
        "missing_count": int(df[target_col].isna().sum()),
        "missing_pct": round(df[target_col].isna().mean() * 100, 2),
        "zero_count": int((df[target_col] == 0).sum()),
        "negative_count": int((df[target_col] < 0).sum()),
        "mean": round(series.mean(), 2),
        "std": round(series.std(), 2),
        "min": round(series.min(), 2),
        "max": round(series.max(), 2),
        "cv": round(series.std() / series.mean() * 100, 2) if series.mean() != 0 else None,
    }