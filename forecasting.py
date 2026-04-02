import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from classification import load_county_data


# ============================================================
# MODEL INPUTS
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

TARGET_COLS = {
    1: "target_1day",
    7: "target_7day_avg",
    14: "target_14day_avg",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_growth(current_series, previous_series):
    previous_series = previous_series.replace(0, np.nan)
    growth = (current_series - previous_series) / previous_series
    growth = growth.replace([np.inf, -np.inf], np.nan).fillna(0)
    return growth


def future_average(series, horizon):
    total = 0
    for i in range(1, horizon + 1):
        total += series.shift(-i)
    return total / horizon


# ============================================================
# BUILD TRAINING DATA
# ============================================================

def build_forecast_frames(_conn, table_name: str, horizon: int):
    """
    Build:
    1. train_df  -> historical rows used for forecasting model training
    2. latest_df -> latest valid row per county for current forecast
    """
    df = load_county_data(_conn, table_name)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    training_parts = []
    latest_parts = []

    for (state, county), group_df in df.groupby(["state", "county"], sort=False):
        group_df = group_df.sort_values("date").reset_index(drop=True)

        if len(group_df) < 30 + horizon:
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

        # Stability and momentum
        group_df["volatility_7"] = group_df["new_cases"].rolling(7).std().fillna(0)
        group_df["acceleration"] = (
            group_df["avg_3day_cases"] - group_df["avg_7day_cases"]
        )

        # Forecast targets
        group_df["target_1day"] = group_df["new_cases"].shift(-1)
        group_df["target_7day_avg"] = future_average(group_df["new_cases"], 7)
        group_df["target_14day_avg"] = future_average(group_df["new_cases"], 14)

        latest_valid = group_df.dropna(subset=FEATURE_COLS).tail(1).copy()
        if not latest_valid.empty:
            latest_parts.append(latest_valid)

        target_col = TARGET_COLS[horizon]
        train_valid = group_df.dropna(subset=FEATURE_COLS + [target_col]).copy()

        if train_valid.empty:
            continue

        # Weekly sampling keeps training lighter
        train_valid = train_valid.iloc[::7].copy()
        training_parts.append(train_valid)

    if not training_parts or not latest_parts:
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.concat(training_parts, ignore_index=True)
    latest_df = pd.concat(latest_parts, ignore_index=True)

    return train_df, latest_df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_forecast_model(train_df: pd.DataFrame, horizon: int):
    """
    Train a Random Forest regressor and evaluate it on a time-based split.
    """
    target_col = TARGET_COLS[horizon]

    model_df = train_df.sort_values("date").reset_index(drop=True)

    X = model_df[FEATURE_COLS].fillna(0)
    y = model_df[target_col].fillna(0)

    split_index = int(len(model_df) * 0.80)

    if split_index <= 0 or split_index >= len(model_df):
        split_index = max(1, len(model_df) - 1)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_valid = X.iloc[split_index:]
    y_valid = y.iloc[split_index:]

    # Validation model
    eval_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    eval_model.fit(X_train, y_train)

    valid_preds = eval_model.predict(X_valid)

    mae = mean_absolute_error(y_valid, valid_preds)
    rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
    r2 = r2_score(y_valid, valid_preds) if len(y_valid) > 1 else 0.0

    # Final model on all available rows
    final_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    metrics = {
        "training_rows": int(len(model_df)),
        "validation_rows": int(len(y_valid)),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    return final_model, metrics


# ============================================================
# MAIN FORECAST WORKFLOW
# ============================================================

@st.cache_data(show_spinner=False)
def prepare_forecast_artifacts(_conn, table_name: str, horizon: int):
    """
    Train the forecasting model and generate latest predictions for all counties.
    Cached per horizon so 1-day / 7-day / 14-day each build once.
    """
    train_df, latest_df = build_forecast_frames(_conn, table_name, horizon)

    if train_df.empty or latest_df.empty:
        return None, pd.DataFrame(), None

    model, metrics = train_forecast_model(train_df, horizon)

    latest_df = latest_df.reset_index(drop=True).copy()
    X_latest = latest_df[FEATURE_COLS].fillna(0)

    latest_df["predicted_value"] = model.predict(X_latest)
    latest_df["predicted_value"] = latest_df["predicted_value"].clip(lower=0)

    latest_df = latest_df.sort_values(["state", "county"]).reset_index(drop=True)

    return model, latest_df, metrics