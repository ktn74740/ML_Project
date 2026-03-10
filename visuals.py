import pandas as pd
import plotly.express as px
import streamlit as st

from db import read_sql


# get list of states from database
def get_states(conn, table_name: str):
    q = f"SELECT DISTINCT state FROM {table_name} ORDER BY state"
    return read_sql(conn, q)["state"].tolist()


def render_home(conn, table_name: str):
    st.subheader("Home")

    # get latest available date in dataset
    latest = read_sql(conn, f"SELECT MAX(date) AS max_date FROM {table_name}")
    max_date = latest.loc[0, "max_date"]

    # aggregate total country data from all counties for latest date
    us_totals = read_sql(
        conn,
        f"""
        SELECT
            SUM(cases)      AS total_cases,
            SUM(deaths)     AS total_deaths,
            SUM(new_cases)  AS total_new_cases,
            SUM(new_deaths) AS total_new_deaths
        FROM {table_name}
        WHERE date = ?
        """,
        (max_date,)
    )

    # count total number of counties present in data
    counties_count = read_sql(
        conn,
        f"SELECT COUNT(DISTINCT state || '|' || county) AS counties FROM {table_name}"
    ).loc[0, "counties"]

    # display main country level metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data Date", str(max_date))
    c2.metric("Total US Cases", f"{int(us_totals.loc[0, 'total_cases']):,}")
    c3.metric("Total US Deaths", f"{int(us_totals.loc[0, 'total_deaths']):,}")
    c4.metric("New Cases (that day)", f"{int(us_totals.loc[0, 'total_new_cases']):,}")
    c5.metric("Counties Covered", f"{int(counties_count):,}")

    st.markdown("---")

    st.markdown("### State Summary")

    # select state to see detailed analysis
    states = get_states(conn, table_name)
    state = st.selectbox("Select State", states)

    # sum daily new cases and deaths for selected state
    daily = read_sql(
        conn,
        f"""
        SELECT date,
               SUM(new_cases)  AS new_cases,
               SUM(new_deaths) AS new_deaths
        FROM {table_name}
        WHERE state = ?
        GROUP BY date
        ORDER BY date
        """,
        (state,)
    )

    daily["date"] = pd.to_datetime(daily["date"])

    # calculate 7 day moving average for state level daily cases
    daily["ma7_new_cases"] = daily["new_cases"].rolling(7, min_periods=1).mean()

    # ----------------------------
    # monthly bar chart
    # ----------------------------
    st.markdown("#### Monthly New Cases vs Deaths")

    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    monthly = daily.groupby("month", as_index=False)[["new_cases", "new_deaths"]].sum()

    plot_monthly = monthly.melt(
        id_vars="month",
        value_vars=["new_cases", "new_deaths"],
        var_name="metric",
        value_name="count"
    )

    plot_monthly["metric"] = plot_monthly["metric"].replace(
        {"new_cases": "New Cases", "new_deaths": "Deaths"}
    )

    fig_month = px.bar(
        plot_monthly,
        x="month",
        y="count",
        color="metric",
        barmode="group",
        title=f"{state} — Monthly New Cases vs Deaths",
    )

    st.plotly_chart(fig_month, use_container_width=True)

    # show monthly data in expandable table
    with st.expander("Monthly totals table"):
        st.dataframe(
            monthly.rename(columns={
                "new_cases": "monthly_new_cases",
                "new_deaths": "monthly_deaths"
            }),
            use_container_width=True
        )

    # ----------------------------
    # daily new cases + 7 day avg graph
    # ----------------------------
    st.markdown("#### Daily New Cases with 7-Day Average")

    plot_daily = daily[["date", "new_cases", "ma7_new_cases"]].copy()
    plot_daily = plot_daily.melt(
        id_vars="date",
        value_vars=["new_cases", "ma7_new_cases"],
        var_name="metric",
        value_name="count"
    )

    plot_daily["metric"] = plot_daily["metric"].replace(
        {"new_cases": "Daily New Cases", "ma7_new_cases": "7-Day Average"}
    )

    fig_daily = px.line(
        plot_daily,
        x="date",
        y="count",
        color="metric",
        title=f"{state} — Daily New Cases and 7-Day Average",
    )

    st.plotly_chart(fig_daily, use_container_width=True)


def render_county_analysis(conn, table_name: str):
    st.subheader("County Wise Analysis")

    # select state and county
    states = get_states(conn, table_name)
    state = st.selectbox("Select State", states, key="cw_state")

    counties = read_sql(
        conn,
        f"SELECT DISTINCT county FROM {table_name} WHERE state = ? ORDER BY county",
        (state,)
    )["county"].tolist()

    county = st.selectbox("Select County", counties, key="cw_county")

    # get full time series data for selected county
    q = f"""
    SELECT date, cases, deaths, new_cases, new_deaths, ma7_new_cases
    FROM {table_name}
    WHERE state = ? AND county = ?
    ORDER BY date
    """

    dff = read_sql(conn, q, (state, county))
    dff["date"] = pd.to_datetime(dff["date"])

    if dff.empty:
        st.warning("No data found for this county selection.")
        return

    # date range filter
    min_d = dff["date"].min().to_pydatetime()
    max_d = dff["date"].max().to_pydatetime()

    start_d, end_d = st.slider(
        "Date Range",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d)
    )

    dff = dff[(dff["date"] >= pd.to_datetime(start_d)) &
              (dff["date"] <= pd.to_datetime(end_d))]

    # show key metrics for selected county
    latest = dff.sort_values("date").tail(1).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative Cases", f"{int(latest['cases']):,}")
    c2.metric("Cumulative Deaths", f"{int(latest['deaths']):,}")
    c3.metric("Latest New Cases", f"{int(latest['new_cases']):,}")
    c4.metric("7-Day Avg New Cases", f"{float(latest['ma7_new_cases']):,.0f}")

    tab1, tab2 = st.tabs(["Cumulative Trend", "Daily New + MA7"])

    with tab1:
        fig1 = px.line(
            dff,
            x="date",
            y=["cases", "deaths"],
            title=f"{county}, {state} — Cumulative Cases/Deaths"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = px.bar(
            dff,
            x="date",
            y="new_cases",
            title="Daily New Cases"
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(
            dff,
            x="date",
            y="ma7_new_cases",
            title="7-Day Moving Average of New Cases"
        )
        st.plotly_chart(fig3, use_container_width=True)