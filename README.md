# WFM Analytics Platform

A Streamlit-based workforce management (WFM) analytics application for data ingestion, exploratory analysis, model training, forecasting, and staffing planning.

## Overview

This project is built for Just In Time Workforce Solutions Inc. and provides a multi-step WFM analytics pipeline with interactive data upload, quality checks, model comparison, forecast generation, and Erlang C staffing conversion.

## Key Features

- Upload and validate CSV / Excel WFM datasets
- Detect datetime and numeric columns automatically
- Infer data granularity and prepare time series data
- Data quality scorecard and validation checks
- Exploratory Data Analysis (EDA) with descriptive statistics, seasonality, volatility, operational metrics, and stationarity diagnostics
- Train and compare statistical, machine learning, and forecast models
- Generate forecasts using the best model or ensembles
- Adjust forecast volumes with what-if scenarios
- Convert forecasts into staffing requirements using Erlang C
- Export forecasts and staffing plans

## Installation

1. Create a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the App

From the project root:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Workflow

1. **Ingestion and EDA**
   - Upload a WFM dataset (`csv`, `xlsx`, or `xls`)
   - Map the date/time and volume columns
   - Select optional operational metrics such as AHT, SL%, occupancy, or headcount
   - Validate the dataset and review the quality scorecard
   - Explore descriptive analytics, seasonality, decomposition, holiday impact, volatility, and stationarity

2. **Modeling and Evaluation**
   - Preprocess the series with imputation, outlier treatment, and stationarity transforms
   - Define train / validation / test splits
   - Configure and train models such as ARIMA/SARIMA, ETS, Holt-Winters, Prophet, XGBoost, LightGBM, Random Forest, and Ridge Regression
   - Review evaluation metrics, leaderboard, and residual diagnostics

3. **Forecasting**
   - Select a forecast model or create an ensemble
   - Generate point forecasts with optional prediction intervals
   - Adjust the forecast via what-if volume scenarios
   - Convert forecast output into interval-based staffing requirements using Erlang C logic

## Project Structure

- `app.py` — Streamlit entry point and main page navigation
- `config.py` — Global configuration, model labels, granularity options, and defaults
- `requirements.txt` — Python dependency list
- `pages/1_Ingestion_and_EDA.py` — Data upload, validation, and EDA workflow
- `pages/2_Modeling_and_Evaluation.py` — Model training, tuning, and evaluation
- `pages/3_Forecasting.py` — Forecasting, ensembles, what-if scenarios, and staffing conversion
- `utils/` — Supporting modules for data handling, EDA, modeling, evaluation, and forecast exports

## Notes

- The app stores intermediate data in `st.session_state` to flow from ingestion to forecasting.
- Forecasting supports both single-model and ensemble workflows.
- Erlang C staffing uses configurable service level, AHT, answer time, and shrinkage.

## Future Improvements

- Add dedicated support for LSTM / GRU model training
- Expand export formats and reporting templates
- Add dashboard summarization for multiple scenarios
