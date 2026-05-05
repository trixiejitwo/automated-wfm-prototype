"""
app.py
Just In Time Workforce Solutions Inc. — WFM Analytics Platform
Entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="WFM Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("WFM Analytics Platform")
st.caption("Just In Time Workforce Solutions Inc.")
st.markdown(
    "Use the sidebar to navigate through the pipeline: "
    "**Ingestion and EDA**, **Modeling and Evaluation**, and **Forecasting**."
)