import json
import ssl
import urllib.request

import plotly.express as px
import streamlit as st

from classification import predict_current_hotspots, get_county_history


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
# PREDICTION PAGE PLACEHOLDER
# ============================================================

def forecasting_placeholder():
    st.subheader("Forecasting")
    st.info("Next step: county-level ML prediction page.")