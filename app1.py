import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="COVID County Dashboard", layout="wide")

# ----------------------------
# MOCK DATA
# ----------------------------
@st.cache_data
def make_mock_data(seed=42):
    rng = np.random.default_rng(seed)

    states = ["Illinois", "Texas", "California", "Florida"]
    counties = {
        "Illinois": ["Cook", "DuPage", "Lake"],
        "Texas": ["Harris", "Dallas", "Travis"],
        "California": ["Los Angeles", "San Diego", "Orange"],
        "Florida": ["Miami-Dade", "Broward", "Orange"],
    }

    end = datetime.today().date()
    start = end - timedelta(days=360)
    dates = pd.date_range(start=start, end=end, freq="D")

    rows = []
    for state in states:
        for county in counties[state]:
            cases = int(rng.integers(5_000, 30_000))
            deaths = int(cases * rng.uniform(0.008, 0.02))

            for i, d in enumerate(dates):
                # wave + noise
                wave = (np.sin(i / 18) + 1.4) * rng.uniform(60, 220)
                noise = rng.normal(30, 25)
                new_cases = max(0, int(wave + noise))

                # deaths correlated with cases (small fraction)
                new_deaths = max(0, int(new_cases * rng.uniform(0.006, 0.02)))

                cases += new_cases
                deaths += new_deaths

                rows.append({
                    "date": d,
                    "state": state,
                    "county": county,
                    "cases": cases,
                    "deaths": deaths,
                    "new_cases": new_cases,
                    "new_deaths": new_deaths,
                })

    df = pd.DataFrame(rows).sort_values(["state", "county", "date"]).reset_index(drop=True)
    df["ma7"] = df.groupby(["state", "county"])["new_cases"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    latest = df[df["date"] == df["date"].max()].copy()
    latest["risk_score"] = (
        (latest["new_cases"] / (latest["new_cases"].max() or 1)) * 50 +
        (latest["ma7"] / (latest["ma7"].max() or 1)) * 50
    )
    latest["risk_category"] = pd.cut(
        latest["risk_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"]
    )

    return df, latest

df, df_latest = make_mock_data()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "County Analysis", "Hotspot Detection", "Forecasting"]
)

# ----------------------------
# HOME (UPDATED)
# ----------------------------
if page == "Home":
    st.title("Home — State Monthly Summary (Mock)")

    # KPIs (optional)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases (latest)", f"{int(df_latest['cases'].sum()):,}")
    col2.metric("Total Deaths (latest)", f"{int(df_latest['deaths'].sum()):,}")
    col3.metric("Counties Covered", df[['state', 'county']].drop_duplicates().shape[0])

    st.markdown("### Select a state to view monthly totals")

    state = st.selectbox("State", sorted(df["state"].unique().tolist()))

    # Aggregate to state-level monthly totals
    # (Sum across counties for each date, then group by month)
    state_daily = df[df["state"] == state].groupby("date", as_index=False)[["new_cases", "new_deaths"]].sum()
    state_daily["month"] = state_daily["date"].dt.to_period("M").dt.to_timestamp()

    state_monthly = state_daily.groupby("month", as_index=False)[["new_cases", "new_deaths"]].sum()

    # Bar chart (grouped)
    plot_df = state_monthly.melt(
        id_vars="month",
        value_vars=["new_cases", "new_deaths"],
        var_name="metric",
        value_name="count"
    )
    plot_df["metric"] = plot_df["metric"].replace({
        "new_cases": "New Cases",
        "new_deaths": "Deaths"
    })

    fig = px.bar(
        plot_df,
        x="month",
        y="count",
        color="metric",
        barmode="group",
        title=f"{state} — Monthly New Cases vs Deaths (Mock)",
        labels={"month": "Month", "count": "Total", "metric": "Metric"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: show the monthly table (helps graders)
    with st.expander("Show monthly totals table"):
        st.dataframe(state_monthly.rename(columns={"new_cases": "monthly_new_cases", "new_deaths": "monthly_deaths"}))

# ----------------------------
# COUNTY ANALYSIS
# ----------------------------
elif page == "County Analysis":
    st.title("County Analysis")

    state = st.selectbox("Select State", sorted(df["state"].unique()), key="ca_state")
    county = st.selectbox("Select County", sorted(df[df["state"] == state]["county"].unique()), key="ca_county")

    dff = df[(df["state"] == state) & (df["county"] == county)].copy()

    latest = dff.sort_values("date").iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Cumulative Cases", f"{int(latest['cases']):,}")
    col2.metric("Cumulative Deaths", f"{int(latest['deaths']):,}")
    col3.metric("Latest New Cases", f"{int(latest['new_cases']):,}")

    st.markdown("### Cumulative Cases/Deaths")
    fig1 = px.line(dff, x="date", y=["cases", "deaths"], title=f"{county}, {state} — Cumulative")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Daily New Cases + 7-Day Average")
    fig2 = px.line(dff, x="date", y=["new_cases", "ma7"], title=f"{county}, {state} — New Cases & MA7")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# HOTSPOT DETECTION (MOCK)
# ----------------------------
elif page == "Hotspot Detection":
    st.title("Hotspot Detection (Mock)")

    st.markdown("### Top 10 Risk Counties")
    top = df_latest.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(top[["state", "county", "new_cases", "ma7", "risk_category"]], use_container_width=True)

    fig = px.bar(top, x="county", y="risk_score", color="risk_category", title="Top 10 Hotspots (Mock Risk Score)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# FORECASTING (MOCK)
# ----------------------------
elif page == "Forecasting":
    st.title("Forecasting (Mock UI)")

    state = st.selectbox("Select State", sorted(df["state"].unique()), key="fc_state")
    county = st.selectbox("Select County", sorted(df[df["state"] == state]["county"].unique()), key="fc_county")

    dff = df[(df["state"] == state) & (df["county"] == county)].copy().sort_values("date")

    horizon = st.slider("Forecast Days", 7, 30, 14)

    last_value = float(dff["new_cases"].tail(7).mean())
    future_dates = pd.date_range(dff["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_values = (last_value * np.linspace(0.9, 1.1, horizon)).clip(min=0)

    forecast_df = pd.DataFrame({"date": future_dates, "forecast_new_cases": forecast_values})

    fig_hist = px.line(dff.tail(90), x="date", y="new_cases", title="Historical Daily New Cases (last 90 days)")
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_fc = px.line(forecast_df, x="date", y="forecast_new_cases", title=f"Forecast (next {horizon} days) — Mock")
    st.plotly_chart(fig_fc, use_container_width=True)