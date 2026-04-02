import streamlit as st

from db import connect_db, table_exists
from visuals import render_home, render_county_analysis
from ml_model import hotspot_placeholder, forecasting_placeholder

# ----------------------------
# SETTINGS
# ----------------------------
APP_TITLE = "Covid 19 Risk Predictor"
DB_PATH = "covid_counties.db"
TABLE_NAME = "counties_daily"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# ----------------------------
# SIDEBAR (final: only navigation)
# ----------------------------
page = st.sidebar.radio(
    "Menu",
    ["Home", "County Wise", "Hotspot", "Prediction"]
)

# ----------------------------
# DB CONNECTION
# ----------------------------
conn = connect_db(DB_PATH)

if not table_exists(conn, TABLE_NAME):
    st.error(
        "Database not found / table missing.\n\n"
        "Run your ETL script once to create the SQLite DB, then restart the app."
    )
    st.stop()

# ----------------------------
# ROUTING
# ----------------------------
if page == "Home":
    render_home(conn, TABLE_NAME)

elif page == "County Wise":
    render_county_analysis(conn, TABLE_NAME)

elif page == "Hotspot":
    hotspot_placeholder(conn, TABLE_NAME)

elif page == "Prediction":
    forecasting_placeholder()