import json
import ssl
import urllib.request

import pandas as pd
import plotly.express as px
import streamlit as st

from classification import predict_current_hotspots, get_county_history
from forecasting import prepare_forecast_artifacts


# ============================================================
# MAP SUPPORT
# ============================================================

@st.cache_data
def load_us_counties_geojson():
    """
    Load USA county boundary GeoJSON.

    SSL verification is bypassed here because some local Python
    environments fail certificate verification while downloading
    the file from GitHub.
    """
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ssl_context = ssl._create_unverified_context()

    with urllib.request.urlopen(url, context=ssl_context) as response:
        return json.load(response)


# ============================================================
# HOTSPOT PAGE
# ============================================================

def hotspot_placeholder(conn, table_name):
    """
    Hotspot page:
    - summary cards
    - top 10 risky counties
    - USA county risk map
    - selected county detail view
    """
    st.subheader("Hotspot Detection")

    results, metrics = predict_current_hotspots(conn, table_name)

    if results.empty:
        st.warning("No hotspot results available.")
        return

    # --------------------------------------------------------
    # Summary cards
    # --------------------------------------------------------
    total_counties = len(results)
    high_count = int((results["predicted_risk"] == "High").sum())
    medium_count = int((results["predicted_risk"] == "Medium").sum())
    low_count = int((results["predicted_risk"] == "Low").sum())

    top_county = results.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Counties Analyzed", f"{total_counties:,}")
    c2.metric("High Risk", f"{high_count:,}")
    c3.metric("Medium Risk", f"{medium_count:,}")
    c4.metric("Low Risk", f"{low_count:,}")
    c5.metric("Top Risk County", f"{top_county['county']}, {top_county['state']}")

    if metrics is not None:
        st.caption(
            f"Model: Random Forest Classifier | "
            f"Training rows: {metrics['training_rows']:,} | "
            f"OOB score: {metrics['oob_score']:.3f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Top 10 risky counties
    # --------------------------------------------------------
    st.markdown("### Top 10 Risky Counties")

    top10 = results.head(10)[
        [
            "rank",
            "state",
            "county",
            "predicted_risk",
            "confidence",
            "new_cases",
            "avg_7day_cases",
            "growth_7",
            "trend",
        ]
    ].copy()

    top10["confidence"] = top10["confidence"].round(3)
    top10["avg_7day_cases"] = top10["avg_7day_cases"].round(1)
    top10["growth_7"] = (top10["growth_7"] * 100).round(2)

    st.dataframe(
        top10.rename(
            columns={
                "predicted_risk": "risk_level",
                "growth_7": "growth_7_percent",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # USA risk map
    # --------------------------------------------------------
    st.markdown("### USA Risk Map")

    try:
        geojson = load_us_counties_geojson()

        map_df = results.copy()

        # Keep only rows that have a valid county FIPS code
        map_df = map_df[map_df["fips"].notna()].copy()
        map_df["fips"] = map_df["fips"].astype(str)
        map_df = map_df[map_df["fips"].str.len() == 5].copy()

        if map_df.empty:
            st.warning("No valid county FIPS values found for the map.")
        else:
            fig_map = px.choropleth(
                map_df,
                geojson=geojson,
                locations="fips",
                featureidkey="id",
                color="predicted_risk",
                scope="usa",
                hover_name="county",
                hover_data={
                    "state": True,
                    "confidence": ":.2f",
                    "avg_7day_cases": ":.1f",
                    "fips": True,
                },
                color_discrete_map={
                    "Low": "green",
                    "Medium": "orange",
                    "High": "red",
                },
                title="County-Level Predicted Risk",
            )

            fig_map.update_geos(
                visible=False,
                scope="usa",
                projection_type="albers usa",
            )

            fig_map.update_layout(
                margin={"r": 0, "t": 50, "l": 0, "b": 0}
            )

            st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.warning(f"Map could not be loaded: {e}")

    st.markdown("---")

    # --------------------------------------------------------
    # County detail section
    # --------------------------------------------------------
    st.markdown("### County Details")

    results["display_name"] = results["state"] + " — " + results["county"]

    selected_display = st.selectbox(
        "Select County",
        results["display_name"].tolist(),
        index=0,
        key="hotspot_county_select",
    )

    selected_row = results.loc[results["display_name"] == selected_display].iloc[0]
    selected_state = selected_row["state"]
    selected_county = selected_row["county"]

    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
    cc1.metric("Risk Level", selected_row["predicted_risk"])
    cc2.metric("Confidence", f"{selected_row['confidence']:.2f}")
    cc3.metric("Cumulative Cases", f"{int(selected_row['cases']):,}")
    cc4.metric("Cumulative Deaths", f"{int(selected_row['deaths']):,}")
    cc5.metric("Latest New Cases", f"{int(selected_row['new_cases']):,}")
    cc6.metric("7-Day Avg Cases", f"{selected_row['avg_7day_cases']:.1f}")

    st.info(selected_row["reason"])

    st.markdown(
        f"**Trend:** {selected_row['trend']}  |  "
        f"**7-Day Growth:** {selected_row['growth_7'] * 100:.2f}%"
    )

    history = get_county_history(conn, table_name, selected_state, selected_county)

    if not history.empty:
        plot_df = history[["date", "new_cases", "ma7_new_cases"]].copy()
        plot_df = plot_df.melt(
            id_vars="date",
            value_vars=["new_cases", "ma7_new_cases"],
            var_name="metric",
            value_name="count",
        )

        plot_df["metric"] = plot_df["metric"].replace(
            {
                "new_cases": "Daily New Cases",
                "ma7_new_cases": "7-Day Moving Average",
            }
        )

        fig_trend = px.line(
            plot_df,
            x="date",
            y="count",
            color="metric",
            title=f"{selected_county}, {selected_state} — Historical Trend",
        )

        st.plotly_chart(fig_trend, use_container_width=True)


# ============================================================
# PREDICTION PAGE
# ============================================================

def forecasting_placeholder(conn, table_name):
    """
    Prediction page:
    - choose horizon
    - select state and county
    - show predicted value
    - show validation metrics
    - plot history + forecast
    """
    st.subheader("Forecasting")

    # User chooses prediction horizon
    horizon = st.radio(
        "Select Forecast Horizon",
        options=[1, 7, 14],
        horizontal=True,
        format_func=lambda x: f"{x}-Day",
    )

    # Build forecast model + latest predictions
    model, latest_df, metrics = prepare_forecast_artifacts(conn, table_name, horizon)

    if latest_df.empty or model is None:
        st.warning("No forecast data available.")
        return

    # --------------------------------------------------------
    # State and county selection
    # --------------------------------------------------------
    states = sorted(latest_df["state"].dropna().unique().tolist())

    selected_state = st.selectbox(
        "Select State",
        states,
        key=f"forecast_state_{horizon}",
    )

    county_options = (
        latest_df.loc[latest_df["state"] == selected_state, "county"]
        .dropna()
        .sort_values()
        .tolist()
    )

    selected_county = st.selectbox(
        "Select County",
        county_options,
        key=f"forecast_county_{horizon}",
    )

    selected_row = latest_df[
        (latest_df["state"] == selected_state) &
        (latest_df["county"] == selected_county)
    ].iloc[0]

    forecast_value = float(selected_row["predicted_value"])

    # --------------------------------------------------------
    # Summary section
    # --------------------------------------------------------
    if horizon == 1:
        forecast_label = "Next-Day Forecast"
        forecast_desc = "Predicted new cases for the next day"
    elif horizon == 7:
        forecast_label = "Next 7-Day Avg Forecast"
        forecast_desc = "Predicted average daily new cases over the next 7 days"
    else:
        forecast_label = "Next 14-Day Avg Forecast"
        forecast_desc = "Predicted average daily new cases over the next 14 days"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(forecast_label, f"{forecast_value:,.1f}")
    c2.metric("Latest New Cases", f"{float(selected_row['new_cases']):,.0f}")
    c3.metric("Current 7-Day Avg", f"{float(selected_row['avg_7day_cases']):,.1f}")
    c4.metric("7-Day Growth", f"{float(selected_row['growth_7']) * 100:,.2f}%")

    if metrics is not None:
        st.caption(
            f"Model: Random Forest Regressor | "
            f"Training rows: {metrics['training_rows']:,} | "
            f"Validation rows: {metrics['validation_rows']:,} | "
            f"MAE: {metrics['mae']:.2f} | "
            f"RMSE: {metrics['rmse']:.2f} | "
            f"R²: {metrics['r2']:.3f}"
        )

    st.info(forecast_desc)

    st.markdown("---")

    # --------------------------------------------------------
    # Historical + forecast chart
    # --------------------------------------------------------
    history = get_county_history(conn, table_name, selected_state, selected_county)

    if history.empty:
        st.warning("No historical data found for this county.")
        return

    history_plot = history[["date", "new_cases", "ma7_new_cases"]].copy()
    history_plot = history_plot.melt(
        id_vars="date",
        value_vars=["new_cases", "ma7_new_cases"],
        var_name="metric",
        value_name="count",
    )

    history_plot["metric"] = history_plot["metric"].replace(
        {
            "new_cases": "Daily New Cases",
            "ma7_new_cases": "7-Day Moving Average",
        }
    )

    last_date = history["date"].max()

    if horizon == 1:
        future_dates = [last_date + pd.Timedelta(days=1)]
        future_metric_name = "Forecasted Next-Day Cases"
    else:
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        future_metric_name = f"Forecasted Daily Avg ({horizon}-Day Horizon)"

    forecast_plot = pd.DataFrame({
        "date": future_dates,
        "metric": future_metric_name,
        "count": forecast_value,
    })

    combined_plot = pd.concat([history_plot, forecast_plot], ignore_index=True)

    fig_forecast = px.line(
        combined_plot,
        x="date",
        y="count",
        color="metric",
        title=f"{selected_county}, {selected_state} — Forecast View",
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    if horizon in [7, 14]:
        st.caption(
            f"For the {horizon}-day horizon, the forecast line represents the "
            f"predicted average daily new cases during that upcoming period."
        )