import streamlit as st

def hotspot_placeholder():
    st.subheader("Hotspot Detection (Placeholder)")
    st.info("Later: ML classification model (High / Medium / Low risk).")
    st.write(
        "- Select date/state\n"
        "- Compute features (growth rate, MA7, spike score)\n"
        "- Run model → risk category\n"
        "- Display top hotspot counties + charts\n"
    )

def forecasting_placeholder():
    st.subheader("Forecasting (Placeholder)")
    st.info("Later: Time-series forecasting (Prophet / ARIMA / LSTM).")
    st.write(
        "- Select county\n"
        "- Choose horizon (7/14/30 days)\n"
        "- Train/load model\n"
        "- Plot historical vs forecast + summary metrics\n"
    )