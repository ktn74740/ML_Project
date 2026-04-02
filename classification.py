import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from db import read_sql


# ============================================================
# CONFIG
# ============================================================

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

RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_growth(current_series, previous_series):
    previous_series = previous_series.replace(0, np.nan)
    growth = (current_series - previous_series) / previous_series
    growth = growth.replace([np.inf, -np.inf], np.nan).fillna(0)
    return growth


def future_average(series, horizon=7):
    total = 0
    for i in range(1, horizon + 1):
        total += series.shift(-i)
    return total / horizon


def label_from_future(future_avg_7, medium_threshold, high_threshold):
    if future_avg_7 >= high_threshold:
        return "High"
    elif future_avg_7 >= medium_threshold:
        return "Medium"
    return "Low"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_county_data(_conn, table_name: str) -> pd.DataFrame:
    """
    Load county-level data from SQLite and clean the key columns.
    """
    query = f"""
    SELECT date, state, county, fips, cases, deaths, new_cases, new_deaths, ma7_new_cases
    FROM {table_name}
    ORDER BY state, county, date
    """
    df = read_sql(_conn, query)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = ["cases", "deaths", "new_cases", "new_deaths", "ma7_new_cases"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Convert only valid FIPS values. Missing ones stay as None.
    if "fips" in df.columns:
        fips_num = pd.to_numeric(df["fips"], errors="coerce")

        df["fips"] = None
        valid_mask = fips_num.notna()

        df.loc[valid_mask, "fips"] = (
            fips_num.loc[valid_mask]
            .astype(int)
            .astype(str)
            .str.zfill(5)
        )
    else:
        df["fips"] = None

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_training_and_latest_frames(_conn, table_name: str):
    """
    Build:
    1. train_df  -> historical county snapshots used for training
    2. latest_df -> most recent valid row per county for current prediction
    """
    df = load_county_data(_conn, table_name)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    training_parts = []
    latest_parts = []

    for (state, county), group_df in df.groupby(["state", "county"], sort=False):
        group_df = group_df.sort_values("date").reset_index(drop=True)

        if len(group_df) < 30:
            continue

        # Recent case features
        group_df["avg_3day_cases"] = group_df["new_cases"].rolling(3).mean()
        group_df["avg_7day_cases"] = group_df["new_cases"].rolling(7).mean()
        group_df["prev_7day_cases"] = group_df["avg_7day_cases"].shift(7)
        group_df["growth_7"] = safe_growth(
            group_df["avg_7day_cases"],
            group_df["prev_7day_cases"]
        )

        # Recent death features
        group_df["avg_3day_deaths"] = group_df["new_deaths"].rolling(3).mean()
        group_df["avg_7day_deaths"] = group_df["new_deaths"].rolling(7).mean()

        # Volatility and acceleration
        group_df["volatility_7"] = group_df["new_cases"].rolling(7).std().fillna(0)
        group_df["acceleration"] = (
            group_df["avg_3day_cases"] - group_df["avg_7day_cases"]
        )

        # Future target signal
        group_df["future_avg_7"] = future_average(group_df["new_cases"], horizon=7)

        # Latest valid row for live prediction
        latest_valid = group_df.dropna(subset=FEATURE_COLS).tail(1).copy()
        if not latest_valid.empty:
            latest_parts.append(latest_valid)

        # Historical training rows
        train_valid = group_df.dropna(subset=FEATURE_COLS + ["future_avg_7"]).copy()
        if train_valid.empty:
            continue

        # Weekly sampling keeps training lighter
        train_valid = train_valid.iloc[::7].copy()
        training_parts.append(train_valid)

    if not training_parts or not latest_parts:
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.concat(training_parts, ignore_index=True)
    latest_df = pd.concat(latest_parts, ignore_index=True)

    positive_future = train_df.loc[train_df["future_avg_7"] > 0, "future_avg_7"]

    if len(positive_future) >= 10:
        medium_threshold = float(positive_future.quantile(0.50))
        high_threshold = float(positive_future.quantile(0.85))
    else:
        medium_threshold = 10.0
        high_threshold = 50.0

    if high_threshold <= medium_threshold:
        high_threshold = medium_threshold + 1.0

    train_df["risk_label"] = train_df["future_avg_7"].apply(
        lambda x: label_from_future(x, medium_threshold, high_threshold)
    )

    return train_df, latest_df


# ============================================================
# MODEL TRAINING
# ============================================================

def train_hotspot_model(train_df: pd.DataFrame) -> RandomForestClassifier:
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
    if row["predicted_risk"] == "High":
        return "High projected short-term case level with elevated recent activity."
    elif row["predicted_risk"] == "Medium":
        return "Moderate projected short-term case level with noticeable recent activity."
    return "Lower projected short-term case level with limited recent activity."


# ============================================================
# HOTSPOT PREDICTION
# ============================================================

@st.cache_data(show_spinner=False)
def predict_current_hotspots(_conn, table_name: str):
    """
    Train the classifier and predict the latest risk for each county.
    Cached so the hotspot page does not retrain on every rerun.
    """
    train_df, latest_df = build_training_and_latest_frames(_conn, table_name)

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

    if "prob_High" not in results.columns:
        results["prob_High"] = 0.0
    if "prob_Medium" not in results.columns:
        results["prob_Medium"] = 0.0
    if "prob_Low" not in results.columns:
        results["prob_Low"] = 0.0

    results["confidence"] = results.apply(
        lambda row: max(row["prob_High"], row["prob_Medium"], row["prob_Low"]),
        axis=1,
    )

    results["risk_rank"] = results["predicted_risk"].map(RISK_ORDER)
    results["reason"] = results.apply(make_reason, axis=1)

    results["trend"] = np.where(
        results["growth_7"] > 0.10,
        "Rising",
        np.where(results["growth_7"] < -0.10, "Declining", "Stable"),
    )

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
# COUNTY HISTORY
# ============================================================

@st.cache_data(show_spinner=False)
def get_county_history(_conn, table_name: str, state: str, county: str) -> pd.DataFrame:
    """
    Fetch full historical series for one county.
    Used by both Hotspot and Prediction pages.
    """
    query = f"""
    SELECT date, cases, deaths, new_cases, new_deaths, ma7_new_cases
    FROM {table_name}
    WHERE state = ? AND county = ?
    ORDER BY date
    """
    history_df = read_sql(_conn, query, (state, county))

    if history_df.empty:
        return history_df

    history_df["date"] = pd.to_datetime(history_df["date"])

    numeric_cols = ["cases", "deaths", "new_cases", "new_deaths", "ma7_new_cases"]
    for col in numeric_cols:
        history_df[col] = pd.to_numeric(history_df[col], errors="coerce").fillna(0)

    return history_df