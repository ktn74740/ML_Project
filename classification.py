import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from db import read_sql


# ============================================================
# CONFIG
# ============================================================

# These are the model input features we will train on.
# They are built from recent county-level COVID activity.
FEATURE_COLS = [
    "new_cases",
    "new_deaths",
    "cases",
    "deaths",
    "avg_3day_cases",
    "avg_7day_cases",
    "prev_7day_cases",
    "growth_7",
    "avg_3day_deaths",
    "avg_7day_deaths",
    "volatility_7",
    "acceleration",
]

# Used later for sorting High > Medium > Low
RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_growth(current_series, previous_series):
    """
    Compute percentage growth safely.

    We avoid divide-by-zero problems by replacing 0 with NaN first,
    then filling missing values back with 0.
    """
    previous_series = previous_series.replace(0, np.nan)
    growth = (current_series - previous_series) / previous_series
    growth = growth.replace([np.inf, -np.inf], np.nan).fillna(0)
    return growth


def future_average(series, horizon=7):
    """
    Build the average of the next N future values.

    Example:
    If horizon=7, then for each date we compute the average of
    the next 7 days of new_cases.
    """
    total = 0
    for i in range(1, horizon + 1):
        total += series.shift(-i)
    return total / horizon


def label_from_future(future_avg_7, medium_threshold, high_threshold):
    """
    Convert future outbreak intensity into a class label.

    This gives us a true supervised-learning target:
    we use the future 7-day average to decide if a historical row
    should be treated as Low / Medium / High risk.
    """
    if future_avg_7 >= high_threshold:
        return "High"
    elif future_avg_7 >= medium_threshold:
        return "Medium"
    return "Low"


# ============================================================
# DATA LOADING
# ============================================================

def load_county_data(conn, table_name: str) -> pd.DataFrame:
    """
    Load county-level data from SQLite and do basic cleanup.
    """
    query = f"""
    SELECT date, state, county, fips, cases, deaths, new_cases, new_deaths, ma7_new_cases
    FROM {table_name}
    ORDER BY state, county, date
    """
    df = read_sql(conn, query)

    if df.empty:
        return df

    # Convert date and numeric columns into clean types
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = ["cases", "deaths", "new_cases", "new_deaths", "ma7_new_cases"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # FIPS needs to be a 5-digit string for the USA county map
    if "fips" in df.columns:
        df["fips"] = (
            pd.to_numeric(df["fips"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
    else:
        df["fips"] = "00000"

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_training_and_latest_frames(conn, table_name: str):
    """
    Build two datasets:

    1. train_df
       Historical county snapshots with features + future-based label
       Used to train the classifier.

    2. latest_df
       Most recent valid row per county
       Used to predict today's current hotspot risk.
    """
    df = load_county_data(conn, table_name)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    training_parts = []
    latest_parts = []

    for (state, county), group_df in df.groupby(["state", "county"], sort=False):
        group_df = group_df.sort_values("date").reset_index(drop=True)

        # We skip counties with too little history,
        # because the rolling features and future labels need enough rows.
        if len(group_df) < 30:
            continue

        # Recent case activity features
        group_df["avg_3day_cases"] = group_df["new_cases"].rolling(3).mean()
        group_df["avg_7day_cases"] = group_df["new_cases"].rolling(7).mean()
        group_df["prev_7day_cases"] = group_df["avg_7day_cases"].shift(7)
        group_df["growth_7"] = safe_growth(
            group_df["avg_7day_cases"],
            group_df["prev_7day_cases"]
        )

        # Recent death activity features
        group_df["avg_3day_deaths"] = group_df["new_deaths"].rolling(3).mean()
        group_df["avg_7day_deaths"] = group_df["new_deaths"].rolling(7).mean()

        # Volatility tells us how unstable the recent case pattern is
        group_df["volatility_7"] = group_df["new_cases"].rolling(7).std().fillna(0)

        # Acceleration tells us whether the short-term trend
        # is rising faster than the weekly baseline
        group_df["acceleration"] = (
            group_df["avg_3day_cases"] - group_df["avg_7day_cases"]
        )

        # This is the future target signal we will learn from
        group_df["future_avg_7"] = future_average(group_df["new_cases"], horizon=7)

        # Save the latest valid row for live/current prediction
        latest_valid = group_df.dropna(subset=FEATURE_COLS).tail(1).copy()
        if not latest_valid.empty:
            latest_parts.append(latest_valid)

        # Save historical rows for training
        train_valid = group_df.dropna(subset=FEATURE_COLS + ["future_avg_7"]).copy()
        if train_valid.empty:
            continue

        # Sampling every 7th row keeps training faster
        # while still giving enough historical coverage.
        train_valid = train_valid.iloc[::7].copy()
        training_parts.append(train_valid)

    if not training_parts or not latest_parts:
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.concat(training_parts, ignore_index=True)
    latest_df = pd.concat(latest_parts, ignore_index=True)

    # We derive class thresholds from the actual future case distribution.
    # This is more data-driven than using hardcoded labels.
    positive_future = train_df.loc[train_df["future_avg_7"] > 0, "future_avg_7"]

    if len(positive_future) >= 10:
        medium_threshold = float(positive_future.quantile(0.50))
        high_threshold = float(positive_future.quantile(0.85))
    else:
        medium_threshold = 10.0
        high_threshold = 50.0

    # Safety check in case the quantiles collapse
    if high_threshold <= medium_threshold:
        high_threshold = medium_threshold + 1.0

    # Create supervised class labels from actual future outcomes
    train_df["risk_label"] = train_df["future_avg_7"].apply(
        lambda x: label_from_future(x, medium_threshold, high_threshold)
    )

    return train_df, latest_df


# ============================================================
# MODEL TRAINING + PREDICTION
# ============================================================

def train_hotspot_model(train_df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on historical county snapshots.
    """
    X = train_df[FEATURE_COLS].fillna(0)
    y = train_df["risk_label"]

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def make_reason(row):
    """
    Create a short human-readable explanation for the selected county.
    """
    if row["predicted_risk"] == "High":
        return "High projected short-term case level with elevated recent activity."
    elif row["predicted_risk"] == "Medium":
        return "Moderate projected short-term case level with noticeable recent activity."
    return "Lower projected short-term case level with limited recent activity."


def predict_current_hotspots(conn, table_name: str):
    """
    Full hotspot workflow:
    - build training set from history
    - train classifier
    - predict current/latest county risk
    """
    train_df, latest_df = build_training_and_latest_frames(conn, table_name)

    if train_df.empty or latest_df.empty:
        return pd.DataFrame(), None

    model = train_hotspot_model(train_df)

    X_latest = latest_df[FEATURE_COLS].fillna(0)
    predicted_labels = model.predict(X_latest)
    predicted_probs = model.predict_proba(X_latest)

    classes = list(model.classes_)
    prob_df = pd.DataFrame(
        predicted_probs,
        columns=[f"prob_{cls_name}" for cls_name in classes]
    )

    results = latest_df.reset_index(drop=True).copy()
    results["predicted_risk"] = predicted_labels
    results = pd.concat([results, prob_df], axis=1)

    # Make sure all probability columns exist even if one class is missing
    if "prob_High" not in results.columns:
        results["prob_High"] = 0.0
    if "prob_Medium" not in results.columns:
        results["prob_Medium"] = 0.0
    if "prob_Low" not in results.columns:
        results["prob_Low"] = 0.0

    # Confidence = highest class probability
    results["confidence"] = results.apply(
        lambda row: max(row["prob_High"], row["prob_Medium"], row["prob_Low"]),
        axis=1,
    )

    results["risk_rank"] = results["predicted_risk"].map(RISK_ORDER)
    results["reason"] = results.apply(make_reason, axis=1)

    # Trend indicator from 7-day growth
    results["trend"] = np.where(
        results["growth_7"] > 0.10,
        "Rising",
        np.where(results["growth_7"] < -0.10, "Declining", "Stable"),
    )

    # Sort to show highest-risk counties first
    results = results.sort_values(
        by=["risk_rank", "prob_High", "confidence", "avg_7day_cases", "growth_7"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    results["rank"] = np.arange(1, len(results) + 1)

    metrics = {
        "training_rows": int(len(train_df)),
        "latest_counties": int(len(results)),
        "oob_score": float(getattr(model, "oob_score_", 0.0)),
    }

    return results, metrics


# ============================================================
# DETAIL VIEW SUPPORT
# ============================================================

def get_county_history(conn, table_name: str, state: str, county: str) -> pd.DataFrame:
    """
    Fetch full historical time series for one selected county.
    Used for the detail chart on the Hotspot page.
    """
    query = f"""
    SELECT date, cases, deaths, new_cases, new_deaths, ma7_new_cases
    FROM {table_name}
    WHERE state = ? AND county = ?
    ORDER BY date
    """
    history_df = read_sql(conn, query, (state, county))

    if history_df.empty:
        return history_df

    history_df["date"] = pd.to_datetime(history_df["date"])

    numeric_cols = ["cases", "deaths", "new_cases", "new_deaths", "ma7_new_cases"]
    for col in numeric_cols:
        history_df[col] = pd.to_numeric(history_df[col], errors="coerce").fillna(0)

    return history_df