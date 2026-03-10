import pandas as pd

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load raw CSV."""
    return pd.read_csv(csv_path)

def clean_county_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean + engineer features for county-level COVID data.

    Output columns stored to DB:
    date (YYYY-MM-DD), state, county, fips, cases, deaths,
    new_cases, new_deaths, ma7_new_cases
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Numeric conversion
    for col in ["cases", "deaths"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing critical fields
    df = df.dropna(subset=["date", "state", "county", "cases"])

    # Remove Unknown counties (common non-county bucket)
    df = df[df["county"].astype(str).str.lower() != "unknown"]

    # Fill deaths if missing
    if "deaths" in df.columns:
        df["deaths"] = df["deaths"].fillna(0)
    else:
        df["deaths"] = 0

    # Sort for time-series
    df = df.sort_values(["state", "county", "date"]).reset_index(drop=True)


    df["new_cases"] = df.groupby(["state", "county"])["cases"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby(["state", "county"])["deaths"].diff().fillna(0).clip(lower=0)
    df["ma7_new_cases"] = df.groupby(["state", "county"])["new_cases"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )

    # Keep only what we need
    keep = ["date", "state", "county", "fips", "cases", "deaths", "new_cases", "new_deaths", "ma7_new_cases"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # Store date as string for SQLite
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df