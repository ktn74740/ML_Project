import re
import streamlit as st

from classification import predict_current_hotspots, get_county_history
from forecasting import prepare_forecast_artifacts


# ------------------------------------------------------------
# BASIC TEXT HELPERS
# ------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Lowercase the text and collapse extra spaces so matching is easier.
    """
    return re.sub(r"\s+", " ", text.lower()).strip()


def find_phrase_in_query(query: str, candidates):
    """
    Find the first candidate phrase that appears in the user's query.
    We sort by length so longer names like 'New York' match before 'York'.
    """
    query_norm = normalize_text(query)
    cleaned = [c for c in candidates if isinstance(c, str) and c.strip()]

    for candidate in sorted(set(cleaned), key=len, reverse=True):
        candidate_norm = normalize_text(candidate)
        pattern = rf"(?<!\w){re.escape(candidate_norm)}(?!\w)"
        if re.search(pattern, query_norm):
            return candidate

    return None


def extract_horizon(query: str) -> int:
    """
    Detect whether the user wants 1-day, 7-day, or 14-day prediction.
    Default to 7-day if they ask for forecast without a specific horizon.
    """
    q = normalize_text(query)

    if "14" in q or "14-day" in q or "14 day" in q or "two week" in q or "2 week" in q:
        return 14

    if "1-day" in q or "1 day" in q or "next day" in q or "tomorrow" in q:
        return 1

    if "7" in q or "7-day" in q or "7 day" in q or "week" in q:
        return 7

    return 7


# ------------------------------------------------------------
# ENTITY RESOLUTION
# ------------------------------------------------------------

def resolve_state_and_county(query: str, hotspot_df):
    """
    Try to identify the state and county mentioned in the question.
    If county is ambiguous across multiple states, return that info
    so we can ask the user to specify the state.
    """
    states = hotspot_df["state"].dropna().unique().tolist()
    state = find_phrase_in_query(query, states)

    # If state is already found, only search counties inside that state
    if state:
        county_candidates = (
            hotspot_df.loc[hotspot_df["state"] == state, "county"]
            .dropna()
            .unique()
            .tolist()
        )
        county = find_phrase_in_query(query, county_candidates)
        return state, county, None

    # Otherwise search all counties
    all_counties = hotspot_df["county"].dropna().unique().tolist()
    county = find_phrase_in_query(query, all_counties)

    if county is None:
        return None, None, None

    matches = hotspot_df.loc[hotspot_df["county"].str.lower() == county.lower()].copy()
    matching_states = matches["state"].dropna().unique().tolist()

    # County exists in multiple states and user did not specify which one
    if len(matching_states) > 1:
        return None, county, matching_states

    if len(matching_states) == 1:
        return matching_states[0], county, None

    return None, county, None


# ------------------------------------------------------------
# RESPONSE BUILDERS
# ------------------------------------------------------------

def answer_greeting():
    return (
        "Hi! I can help with hotspot risk, county forecasts, and trend questions.\n\n"
        "Try asking things like:\n"
        "- Which counties are highest risk?\n"
        "- What is the risk for Cook County Illinois?\n"
        "- Forecast Harris County Texas for 7 days\n"
        "- Is Los Angeles County improving?"
    )


def answer_help():
    return (
        "I can answer questions from your current COVID-19 dashboard and ML outputs.\n\n"
        "Supported question types:\n"
        "1. Top hotspots\n"
        "2. County risk lookup\n"
        "3. County forecast lookup\n"
        "4. County trend summary\n\n"
        "Examples:\n"
        "- Show top 10 risky counties\n"
        "- What is the risk for Cook County Illinois?\n"
        "- Forecast Cook County Illinois for 14 days\n"
        "- Is Harris County Texas improving?"
    )


def answer_top_hotspots(query: str, hotspot_df):
    """
    Return top risky counties overall or within a specific state.
    """
    states = hotspot_df["state"].dropna().unique().tolist()
    state = find_phrase_in_query(query, states)

    if state:
        filtered = hotspot_df.loc[hotspot_df["state"] == state].copy()
        filtered = filtered.sort_values(
            ["risk_rank", "confidence", "avg_7day_cases"],
            ascending=[False, False, False],
        )
        top_rows = filtered.head(5)

        if top_rows.empty:
            return f"I could not find hotspot results for {state}."

        lines = [f"Top risky counties in {state}:"]
        for idx, (_, row) in enumerate(top_rows.iterrows(), start=1):
            lines.append(
                f"{idx}. {row['county']} — {row['predicted_risk']} risk "
                f"(confidence {row['confidence']:.2f}, 7-day avg {row['avg_7day_cases']:.1f})"
            )
        return "\n".join(lines)

    top_rows = hotspot_df.head(10)
    lines = ["Top 10 risky counties right now:"]
    for _, row in top_rows.iterrows():
        lines.append(
            f"{int(row['rank'])}. {row['county']}, {row['state']} — "
            f"{row['predicted_risk']} risk "
            f"(confidence {row['confidence']:.2f}, 7-day avg {row['avg_7day_cases']:.1f})"
        )
    return "\n".join(lines)


def answer_county_risk(query: str, hotspot_df):
    """
    Return the risk classification for a specific county.
    """
    state, county, ambiguous_states = resolve_state_and_county(query, hotspot_df)

    if county and ambiguous_states:
        return (
            f"I found {county} County in multiple states: "
            f"{', '.join(ambiguous_states[:6])}. "
            f"Please include the state name."
        )

    if not county:
        return (
            "Please mention a county name for the risk lookup.\n"
            "Example: What is the risk for Cook County Illinois?"
        )

    row = hotspot_df.loc[
        (hotspot_df["state"] == state) & (hotspot_df["county"] == county)
    ]

    if row.empty:
        return f"I could not find risk data for {county}, {state}."

    row = row.iloc[0]

    return (
        f"{county}, {state} is currently classified as **{row['predicted_risk']} risk**.\n\n"
        f"- Confidence: {row['confidence']:.2f}\n"
        f"- Latest new cases: {int(row['new_cases']):,}\n"
        f"- 7-day average: {row['avg_7day_cases']:.1f}\n"
        f"- 7-day growth: {row['growth_7'] * 100:.2f}%\n"
        f"- Trend: {row['trend']}\n\n"
        f"Reason: {row['reason']}"
    )


def answer_forecast(query: str, conn, table_name, hotspot_df):
    """
    Return a 1-day / 7-day / 14-day forecast for a specific county.
    """
    horizon = extract_horizon(query)
    _, latest_df, _ = prepare_forecast_artifacts(conn, table_name, horizon)

    if latest_df.empty:
        return "Forecast data is not available right now."

    state, county, ambiguous_states = resolve_state_and_county(query, hotspot_df)

    if county and ambiguous_states:
        return (
            f"I found {county} County in multiple states: "
            f"{', '.join(ambiguous_states[:6])}. "
            f"Please include the state name."
        )

    # If no county is given, show the top forecasts instead of failing
    if not county:
        top_rows = latest_df.sort_values("predicted_value", ascending=False).head(5)

        if horizon == 1:
            lines = ["Top next-day forecast counties:"]
        elif horizon == 7:
            lines = ["Top next 7-day average forecast counties:"]
        else:
            lines = ["Top next 14-day average forecast counties:"]

        for idx, (_, row) in enumerate(top_rows.iterrows(), start=1):
            lines.append(
                f"{idx}. {row['county']}, {row['state']} — {row['predicted_value']:.1f}"
            )

        return "\n".join(lines)

    row = latest_df.loc[
        (latest_df["state"] == state) & (latest_df["county"] == county)
    ]

    if row.empty:
        return f"I could not find forecast data for {county}, {state}."

    row = row.iloc[0]

    if horizon == 1:
        forecast_text = f"Predicted next-day new cases: **{row['predicted_value']:.1f}**"
    elif horizon == 7:
        forecast_text = (
            f"Predicted average daily new cases over the next 7 days: "
            f"**{row['predicted_value']:.1f}**"
        )
    else:
        forecast_text = (
            f"Predicted average daily new cases over the next 14 days: "
            f"**{row['predicted_value']:.1f}**"
        )

    return (
        f"Forecast for {county}, {state}:\n\n"
        f"{forecast_text}\n\n"
        f"- Latest new cases: {int(row['new_cases']):,}\n"
        f"- Current 7-day average: {row['avg_7day_cases']:.1f}\n"
        f"- 7-day growth: {row['growth_7'] * 100:.2f}%"
    )


def answer_trend(query: str, conn, table_name, hotspot_df):
    """
    Summarize whether a county is improving, stable, or rising.
    """
    state, county, ambiguous_states = resolve_state_and_county(query, hotspot_df)

    if county and ambiguous_states:
        return (
            f"I found {county} County in multiple states: "
            f"{', '.join(ambiguous_states[:6])}. "
            f"Please include the state name."
        )

    if not county:
        return (
            "Please mention a county name for the trend lookup.\n"
            "Example: Is Cook County Illinois improving?"
        )

    history = get_county_history(conn, table_name, state, county)

    if history.empty or len(history) < 14:
        return f"I do not have enough history to summarize the trend for {county}, {state}."

    recent_avg = history["new_cases"].tail(7).mean()
    prev_avg = history["new_cases"].tail(14).head(7).mean()

    if prev_avg == 0:
        growth = 0.0
    else:
        growth = (recent_avg - prev_avg) / prev_avg

    if growth > 0.10:
        direction = "rising"
    elif growth < -0.10:
        direction = "improving"
    else:
        direction = "stable"

    latest_row = history.iloc[-1]

    return (
        f"{county}, {state} looks **{direction}** right now.\n\n"
        f"- Latest new cases: {int(latest_row['new_cases']):,}\n"
        f"- Current 7-day average: {recent_avg:.1f}\n"
        f"- Previous 7-day average: {prev_avg:.1f}\n"
        f"- Weekly change: {growth * 100:.2f}%"
    )


def answer_fallback():
    return (
        "I couldn’t confidently map that question yet.\n\n"
        "Try one of these:\n"
        "- Show top 10 risky counties\n"
        "- What is the risk for Cook County Illinois?\n"
        "- Forecast Harris County Texas for 7 days\n"
        "- Is Los Angeles County improving?"
    )


# ------------------------------------------------------------
# MAIN ROUTER
# ------------------------------------------------------------

def generate_bot_response(query: str, conn, table_name: str) -> str:
    """
    Main chatbot router.

    We first load hotspot results, then decide which type of answer
    the user likely wants.
    """
    q = normalize_text(query)
    hotspot_df, _ = predict_current_hotspots(conn, table_name)

    if hotspot_df.empty:
        return "I could not load the current project data."

    # Greetings
    greeting_patterns = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    ]
    if q in greeting_patterns:
        return answer_greeting()

    # Help / capability questions
    if "what can you do" in q or "help" in q or "capabilities" in q:
        return answer_help()

    # Forecast questions
    forecast_keywords = ["forecast", "predict", "prediction", "next day", "next 7", "next 14"]
    if any(keyword in q for keyword in forecast_keywords):
        return answer_forecast(query, conn, table_name, hotspot_df)

    # Trend questions
    trend_keywords = ["trend", "improving", "rising", "declining", "increase", "decrease"]
    if any(keyword in q for keyword in trend_keywords):
        return answer_trend(query, conn, table_name, hotspot_df)

    # Hotspot / risk questions
    hotspot_keywords = ["risk", "hotspot", "risky", "high risk", "top counties", "top 10"]
    if any(keyword in q for keyword in hotspot_keywords):
        state, county, _ = resolve_state_and_county(query, hotspot_df)

        if county:
            return answer_county_risk(query, hotspot_df)

        return answer_top_hotspots(query, hotspot_df)

    return answer_fallback()


# ------------------------------------------------------------
# STREAMLIT PAGE
# ------------------------------------------------------------

def render_chatbot(conn, table_name):
    """
    Render the chatbot page using Streamlit chat UI.
    """
    st.subheader("COVID-19 Data Chatbot")
    st.caption(
        "Ask about hotspot risk, forecasts, and county trends using natural language."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I’m your COVID-19 project bot.\n\n"
                    "You can ask me things like:\n"
                    "- Show top 10 risky counties\n"
                    "- What is the risk for Cook County Illinois?\n"
                    "- Forecast Harris County Texas for 7 days\n"
                    "- Is Los Angeles County improving?"
                ),
            }
        ]

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about risk, forecast, or trend...")

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_bot_response(prompt, conn, table_name)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)